#!/usr/bin/env python3
"""
Read result JSONs produced by `vllm bench serve` (via sweep_official.py)
and print a comparison table + $/M-tokens. Same shape as analyze.py.

Usage:
    python disagg-exp/analyze_official.py --log-dir ./results --configs A1 B C1 D --plot

Cross-check against the custom sweep.py JSONL output:
    python disagg-exp/analyze_official.py --log-dir ./results \
        --configs A1 --compare-custom ./results
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPLOT = True
except ImportError:
    HAS_MPLOT = False

# On-demand $/hr per config. Mirrors analyze.py.
COST_PER_HR = {
    "A":  3.91, "A1": 3.91, "A2": 3.91, "A3": 3.91,
    "B":  1.86,
    "C":  3.91, "C1": 3.91, "C2": 3.91,
    "D":  1.61,
}

# How many of the leading requests to discard as warmup before computing
# percentile stats (analogous to sweep.py's WARMUP_N).
DEFAULT_WARMUP_SKIP = int(os.environ.get("ANALYZE_WARMUP_SKIP", "10"))


# ── load ──────────────────────────────────────────────────────────────────────
def load_points(log_dir: Path, config: str) -> dict[str, dict]:
    """{point_id → result_json_dict} for one config."""
    config_dir = log_dir / config
    if not config_dir.exists():
        return {}
    points: dict[str, dict] = {}
    for jf in sorted(config_dir.glob("p*_d*_r*.json")):
        try:
            with open(jf) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
        # Some result JSONs may be wrapped — handle both shapes defensively.
        if isinstance(data, list):
            data = data[-1] if data else {}
        if isinstance(data, dict):
            points[jf.stem] = data
    return points


# ── stats ─────────────────────────────────────────────────────────────────────
def _p(arr: list[float], pct: float) -> float:
    if not arr:
        return float("nan")
    return float(np.percentile(arr, pct))


def _from_detailed(data: dict, skip: int) -> dict | None:
    """Recompute percentiles from per-request arrays, skipping the first `skip`."""
    ttfts = data.get("ttfts")
    itls = data.get("itls")
    if not ttfts:
        return None

    # ttfts/itls are seconds; convert to ms.
    n = len(ttfts)
    if skip >= n:
        skip = 0  # not enough to skip

    ttfts_ms = [t * 1000.0 for t in ttfts[skip:] if t is not None]

    # TPOT per request = mean of inter-token latencies for that request.
    tpots_ms: list[float] = []
    if itls is not None:
        for itl_list in itls[skip:]:
            if itl_list:
                tpots_ms.append((sum(itl_list) / len(itl_list)) * 1000.0)

    output_lens = data.get("output_lens") or []
    input_lens = data.get("input_lens") or []
    output_lens_skip = output_lens[skip:]
    input_lens_skip  = input_lens[skip:]

    duration = data.get("duration")
    # Approximate post-warmup throughput: assume rate is uniform.
    if duration and n > skip and n > 0:
        warmup_frac = skip / n
        eff_duration = duration * (1.0 - warmup_frac) if warmup_frac < 1.0 else duration
    else:
        eff_duration = duration

    out_throughput = (
        sum(output_lens_skip) / eff_duration if eff_duration and output_lens_skip else float("nan")
    )

    return {
        "n_ok": len(ttfts_ms),
        "n_total": n,
        "ttft_p50_ms": _p(ttfts_ms, 50),
        "ttft_p99_ms": _p(ttfts_ms, 99),
        "tpot_p50_ms": _p(tpots_ms, 50),
        "tpot_p99_ms": _p(tpots_ms, 99),
        "thr_tok_s_e2e": out_throughput,
        "completed": len(ttfts_ms),
        "input_tokens": sum(input_lens_skip),
        "output_tokens": sum(output_lens_skip),
    }


def _from_summary(data: dict) -> dict:
    """Fallback: use the summary fields the benchmark already wrote."""
    return {
        "n_ok": data.get("completed", 0),
        "n_total": data.get("completed", 0) + data.get("failed", 0),
        "ttft_p50_ms": data.get("p50_ttft_ms") or data.get("median_ttft_ms"),
        "ttft_p99_ms": data.get("p99_ttft_ms"),
        "tpot_p50_ms": data.get("p50_tpot_ms") or data.get("median_tpot_ms"),
        "tpot_p99_ms": data.get("p99_tpot_ms"),
        "thr_tok_s_e2e": data.get("output_throughput"),
        "completed": data.get("completed", 0),
        "input_tokens": data.get("total_input_tokens", 0),
        "output_tokens": data.get("total_output_tokens", 0),
    }


def analyze_point(data: dict, skip_warmup: int) -> dict:
    detailed = _from_detailed(data, skip_warmup) if skip_warmup > 0 else None
    stats = detailed if detailed is not None else _from_summary(data)
    failed = data.get("failed", 0) or 0
    n_total = stats.get("n_total", 0) or 0
    stats["fail_rate"] = failed / n_total if n_total else 0.0
    return stats


def dollar_per_m_tokens(stats: dict, config: str) -> float:
    thr = stats.get("thr_tok_s_e2e")
    if not thr or thr != thr:  # NaN check
        return float("nan")
    cost_hr = COST_PER_HR.get(config)
    if cost_hr is None:
        return float("nan")
    m_tok_hr = thr * 3600 / 1e6
    return cost_hr / m_tok_hr if m_tok_hr else float("nan")


# ── table ─────────────────────────────────────────────────────────────────────
def print_table(all_stats: dict[str, dict[str, dict]]) -> None:
    header = (
        f"{'config':<8} {'point':<32} {'n_ok':>6} {'fail%':>6}"
        f" {'ttft_p50ms':>11} {'ttft_p99ms':>11}"
        f" {'tpot_p50ms':>11} {'tpot_p99ms':>11}"
        f" {'thr_tok_s':>10} {'$/Mtok':>8}"
    )
    print(header)
    print("-" * len(header))

    for config in sorted(all_stats):
        for point_id in sorted(all_stats[config]):
            s = all_stats[config][point_id]
            if not s.get("n_ok"):
                print(f"{config:<8} {point_id:<32} {'NO DATA':>6}")
                continue
            dpm = dollar_per_m_tokens(s, config)

            def _fmt(v: float | None, w: int, prec: int = 1) -> str:
                if v is None or (isinstance(v, float) and v != v):
                    return f"{'-':>{w}}"
                return f"{v:>{w}.{prec}f}"

            print(
                f"{config:<8} {point_id:<32} {s['n_ok']:>6} {s.get('fail_rate', 0) * 100:>5.1f}%"
                f" {_fmt(s.get('ttft_p50_ms'), 11)} {_fmt(s.get('ttft_p99_ms'), 11)}"
                f" {_fmt(s.get('tpot_p50_ms'), 11)} {_fmt(s.get('tpot_p99_ms'), 11)}"
                f" {_fmt(s.get('thr_tok_s_e2e'), 10)} {_fmt(dpm, 8, 3)}"
            )


# ── plots (same layout as analyze.py) ────────────────────────────────────────
def plot_comparison(all_stats: dict[str, dict[str, dict]], out_dir: Path) -> None:
    if not HAS_MPLOT:
        print("matplotlib not available, skipping plots")
        return

    by_pd: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for config, points in all_stats.items():
        for point_id, s in points.items():
            try:
                parts = point_id.split("_")
                pl = int(parts[0][1:])
                dl = int(parts[1][1:])
                r  = float(parts[2][1:])
            except Exception:
                continue
            by_pd[(pl, dl)][config].append((r, s))

    for (pl, dl), config_data in by_pd.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"prefill={pl} decode={dl}  (official)")
        for config, rate_stats in sorted(config_data.items()):
            rate_stats.sort(key=lambda x: x[0])
            rates = [x[0] for x in rate_stats]
            ttft50 = [x[1].get("ttft_p50_ms", float("nan")) for x in rate_stats]
            ttft99 = [x[1].get("ttft_p99_ms", float("nan")) for x in rate_stats]
            tpot50 = [x[1].get("tpot_p50_ms", float("nan")) for x in rate_stats]
            dpm    = [dollar_per_m_tokens(x[1], config) for x in rate_stats]
            axes[0].plot(rates, ttft50, marker="o", label=f"{config} p50")
            axes[0].plot(rates, ttft99, marker="x", linestyle="--", label=f"{config} p99")
            axes[1].plot(rates, tpot50, marker="o", label=config)
            axes[2].plot(rates, dpm,    marker="o", label=config)
        axes[0].set_title("TTFT (ms)");      axes[0].set_xlabel("rate (req/s)"); axes[0].legend(fontsize=7)
        axes[1].set_title("TPOT p50 (ms)");  axes[1].set_xlabel("rate (req/s)"); axes[1].legend(fontsize=7)
        axes[2].set_title("$/M tokens");     axes[2].set_xlabel("rate (req/s)"); axes[2].legend(fontsize=7)
        fig.tight_layout()
        fname = out_dir / f"plot_p{pl}_d{dl}.png"
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"  saved {fname}")


# ── cross-check against sweep.py JSONL ───────────────────────────────────────
def load_custom_points(log_dir: Path, config: str) -> dict[str, dict]:
    """Re-use analyze.py's per-request JSONL loader for side-by-side comparison."""
    config_dir = log_dir / config
    if not config_dir.exists():
        return {}
    out: dict[str, dict] = {}
    for jsonl in sorted(config_dir.glob("p*_d*_r*.jsonl")):
        ttfts, tpots, comps = [], [], []
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("phase") != "measured" or r.get("status") != "success":
                    continue
                ttft = r.get("ttft_s")
                e2e  = r.get("e2e_s")
                comp = r.get("completion_tokens")
                if ttft is not None:
                    ttfts.append(ttft * 1000.0)
                if ttft is not None and e2e is not None and comp and comp > 1:
                    tpots.append(((e2e - ttft) / (comp - 1)) * 1000.0)
                if comp:
                    comps.append(comp)
        if ttfts:
            out[jsonl.stem] = {
                "n_ok": len(ttfts),
                "ttft_p50_ms": _p(ttfts, 50),
                "ttft_p99_ms": _p(ttfts, 99),
                "tpot_p50_ms": _p(tpots, 50),
                "tpot_p99_ms": _p(tpots, 99),
            }
    return out


def print_comparison(
    official: dict[str, dict[str, dict]],
    custom: dict[str, dict[str, dict]],
) -> None:
    print("\n=== Side-by-side: official vs custom sweep ===")
    print(
        f"{'config':<8} {'point':<28} "
        f"{'off_ttft50':>11} {'cus_ttft50':>11}   "
        f"{'off_ttft99':>11} {'cus_ttft99':>11}   "
        f"{'off_tpot50':>11} {'cus_tpot50':>11}"
    )
    for config in sorted(set(official) | set(custom)):
        opts = official.get(config, {})
        cpts = custom.get(config, {})
        for point_id in sorted(set(opts) | set(cpts)):
            o = opts.get(point_id, {})
            c = cpts.get(point_id, {})

            def _f(v):
                if v is None or (isinstance(v, float) and v != v):
                    return "-"
                return f"{v:.1f}"

            print(
                f"{config:<8} {point_id:<28} "
                f"{_f(o.get('ttft_p50_ms')):>11} {_f(c.get('ttft_p50_ms')):>11}   "
                f"{_f(o.get('ttft_p99_ms')):>11} {_f(c.get('ttft_p99_ms')):>11}   "
                f"{_f(o.get('tpot_p50_ms')):>11} {_f(c.get('tpot_p50_ms')):>11}"
            )


# ── main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    configs = args.configs or [
        d.name for d in log_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]

    all_stats: dict[str, dict[str, dict]] = {}
    for config in configs:
        points = load_points(log_dir, config)
        if not points:
            print(f"[analyze] no official results for config {config} in {log_dir / config}")
            continue
        all_stats[config] = {
            pid: analyze_point(data, args.skip_warmup)
            for pid, data in points.items()
        }

    if not all_stats:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    print_table(all_stats)

    if args.plot:
        out_dir = log_dir / "plots_official"
        out_dir.mkdir(exist_ok=True)
        plot_comparison(all_stats, out_dir)

    if args.compare_custom:
        custom_dir = Path(args.compare_custom)
        custom_stats: dict[str, dict[str, dict]] = {}
        for config in configs:
            cs = load_custom_points(custom_dir, config)
            if cs:
                custom_stats[config] = cs
        if custom_stats:
            print_comparison(all_stats, custom_stats)
        else:
            print(f"\n[compare] no custom sweep JSONL found under {custom_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("EXP_LOG_DIR", "./results"),
    )
    parser.add_argument(
        "--configs", nargs="*",
        help="Which configs to analyze (default: every subdirectory of --log-dir).",
    )
    parser.add_argument(
        "--skip-warmup", type=int, default=DEFAULT_WARMUP_SKIP,
        help="Discard the first N requests as warmup when recomputing percentiles. "
             "0 = use the benchmark's own aggregates as-is.",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--compare-custom", default=None,
        help="Path to a sweep.py log dir; print side-by-side with the official numbers.",
    )
    args = parser.parse_args()
    main(args)
