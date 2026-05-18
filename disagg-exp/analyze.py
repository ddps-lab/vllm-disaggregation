#!/usr/bin/env python3
"""
Post-experiment analysis for disagg-exp tier-1.

Usage:
    python analyze.py --log-dir ./results [--configs A B C D] [--plot]

Reads per-request JSONL files from $LOG_DIR/<config>/<point>.jsonl.
Prints a table and optionally saves matplotlib figures.
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

# On-demand cost ($/hr) per config. Override via env or edit here.
COST_PER_HR = {
    "A":  3.91,  # g4dn.12xlarge OD (alias for A1)
    "A1": 3.91,  # g4dn.12xlarge OD, TP=2 PP=2
    "A2": 3.91,  # g4dn.12xlarge OD, TP=4 PP=1
    "A3": 3.91,  # g4dn.12xlarge OD, TP=1 PP=4
    "B":  1.86,  # g6e.xlarge OD
    "C":  3.91,  # g4dn.12xlarge OD (alias for C1)
    "C1": 3.91,  # g4dn.12xlarge OD, same-node PD, TP=2 PP=1
    "C2": 3.91,  # g4dn.12xlarge OD, same-node PD, TP=1 PP=2
    "D":  1.61,  # 2× g6.xlarge OD
}


def load_points(log_dir: Path, config: str) -> dict:
    """Returns dict: point_id → list[dict] (measured phase only)."""
    config_dir = log_dir / config
    if not config_dir.exists():
        return {}
    points = {}
    for jsonl in sorted(config_dir.glob("*.jsonl")):
        rows = []
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        measured = [r for r in rows if r.get("phase") == "measured"]
        if measured:
            points[jsonl.stem] = measured
    return points


def _p(arr: list[float], pct: float) -> float:
    if not arr:
        return float("nan")
    return float(np.percentile(arr, pct))


def _warn_pt_delta(rows: list[dict], point_id: str) -> None:
    """Warn if prompt_tokens deviates from prefill_len by more than 1."""
    prefill_len = rows[0].get("prefill_len", 0)
    pts = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens") is not None]
    if not pts:
        return
    deltas = [abs(p - prefill_len) for p in pts]
    bad = [d for d in deltas if d > 1]
    if bad:
        print(
            f"  WARN [{point_id}]: {len(bad)}/{len(pts)} requests have"
            f" |prompt_tokens - prefill_len| > 1 (max={max(bad)})"
        )


def analyze_point(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "success"]
    if not ok:
        return {"n_ok": 0}

    ttfts  = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    e2es   = [r["e2e_s"]  for r in ok if r.get("e2e_s")  is not None]
    comp_t = [r["completion_tokens"] for r in ok if r.get("completion_tokens")]
    decode_len = ok[0].get("decode_len", 1)
    rate   = ok[0].get("rate", 1.0)

    # TPOT: (e2e - ttft) / (completion_tokens - 1), per request, then aggregate
    tpots = []
    for r in ok:
        if r.get("ttft_s") and r.get("e2e_s") and r.get("completion_tokens", 1) > 1:
            tpots.append((r["e2e_s"] - r["ttft_s"]) / (r["completion_tokens"] - 1))

    # Throughput: two definitions
    sends = [r["send_ts"] for r in ok]
    recvs = [r["send_ts"] + r["e2e_s"] for r in ok if r.get("e2e_s")]
    total_tok = sum(comp_t) + sum(r.get("prompt_tokens", 0) for r in ok)

    thr_e2e  = total_tok / (max(recvs) - min(sends)) if recvs else float("nan")
    thr_send = total_tok / (max(sends) - min(sends)) if len(sends) > 1 else float("nan")
    achieved_rate = len(ok) / ((max(sends) - min(sends)) or 1)

    return {
        "n_ok": len(ok),
        "n_total": len(rows),
        "fail_rate": 1.0 - len(ok) / len(rows),
        "ttft_p50_ms": _p(ttfts, 50) * 1000,
        "ttft_p99_ms": _p(ttfts, 99) * 1000,
        "tpot_p50_ms": _p(tpots, 50) * 1000,
        "tpot_p99_ms": _p(tpots, 99) * 1000,
        "thr_tok_s_e2e":  thr_e2e,
        "thr_tok_s_send": thr_send,
        "achieved_rate":  achieved_rate,
    }


def dollar_per_m_tokens(stats: dict, config: str) -> float:
    """$/M output tokens using e2e throughput and on-demand price."""
    thr = stats.get("thr_tok_s_e2e", 0)
    if not thr or thr != thr:
        return float("nan")
    cost_hr = COST_PER_HR.get(config, float("nan"))
    # tokens/s → M tokens/hr → $/M tokens
    m_tok_hr = thr * 3600 / 1e6
    return cost_hr / m_tok_hr if m_tok_hr else float("nan")


def print_table(all_stats: dict[str, dict[str, dict]]) -> None:
    header = (
        f"{'config':<8} {'point':<32} {'n_ok':>6} {'fail%':>6}"
        f" {'ttft_p50ms':>11} {'ttft_p99ms':>11}"
        f" {'tpot_p50ms':>11} {'tpot_p99ms':>11}"
        f" {'thr_e2e':>9} {'$/Mtok':>8}"
    )
    print(header)
    print("-" * len(header))

    for config in sorted(all_stats):
        for point_id in sorted(all_stats[config]):
            s = all_stats[config][point_id]
            if s.get("n_ok", 0) == 0:
                print(f"{config:<8} {point_id:<32} {'NO DATA':>6}")
                continue
            dpm = dollar_per_m_tokens(s, config)
            print(
                f"{config:<8} {point_id:<32} {s['n_ok']:>6} {s['fail_rate']*100:>5.1f}%"
                f" {s['ttft_p50_ms']:>11.1f} {s['ttft_p99_ms']:>11.1f}"
                f" {s['tpot_p50_ms']:>11.1f} {s['tpot_p99_ms']:>11.1f}"
                f" {s['thr_tok_s_e2e']:>9.1f} {dpm:>8.3f}"
            )


def plot_comparison(all_stats: dict[str, dict[str, dict]], out_dir: Path) -> None:
    """One plot per (prefill_len, decode_len) pair: TTFT p50 vs rate for all configs."""
    if not HAS_MPLOT:
        print("matplotlib not available, skipping plots")
        return

    # Collect (prefill, decode, rate) → config → stats
    by_pd: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for config, points in all_stats.items():
        for point_id, s in points.items():
            # point_id: p{prefill}_d{decode}_r{rate}
            try:
                parts = point_id.split("_")
                pl = int(parts[0][1:])
                dl = int(parts[1][1:])
                r  = float(parts[2][1:])
            except Exception:
                continue
            by_pd[(pl, dl)][f"{config}_{r}"] = s
            by_pd[(pl, dl)][config] = by_pd[(pl, dl)].get(config, {})
            # store as list for plotting
            if not isinstance(by_pd[(pl, dl)].get(config), list):
                by_pd[(pl, dl)][config] = []
            by_pd[(pl, dl)][config].append((r, s))

    for (pl, dl), config_data in by_pd.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"prefill={pl} decode={dl}")

        for config, rate_stats in sorted(config_data.items()):
            if not isinstance(rate_stats, list):
                continue
            rate_stats.sort(key=lambda x: x[0])
            rates = [x[0] for x in rate_stats]
            ttft50 = [x[1].get("ttft_p50_ms", float("nan")) for x in rate_stats]
            ttft99 = [x[1].get("ttft_p99_ms", float("nan")) for x in rate_stats]
            tpot50 = [x[1].get("tpot_p50_ms", float("nan")) for x in rate_stats]
            dpm = [dollar_per_m_tokens(x[1], config) for x in rate_stats]

            axes[0].plot(rates, ttft50, marker="o", label=f"{config} p50")
            axes[0].plot(rates, ttft99, marker="x", linestyle="--", label=f"{config} p99")
            axes[1].plot(rates, tpot50, marker="o", label=config)
            axes[2].plot(rates, dpm, marker="o", label=config)

        axes[0].set_title("TTFT (ms)")
        axes[0].set_xlabel("rate (req/s)")
        axes[0].legend(fontsize=7)

        axes[1].set_title("TPOT p50 (ms/tok)")
        axes[1].set_xlabel("rate (req/s)")
        axes[1].legend(fontsize=7)

        axes[2].set_title("$/M tokens (OD)")
        axes[2].set_xlabel("rate (req/s)")
        axes[2].legend(fontsize=7)

        fig.tight_layout()
        fname = out_dir / f"plot_p{pl}_d{dl}.png"
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"  saved {fname}")


def main(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    configs = args.configs or ["A1", "A2", "A3", "B", "C1", "C2", "D"]

    all_stats: dict[str, dict[str, dict]] = {}
    for config in configs:
        points = load_points(log_dir, config)
        if not points:
            print(f"[analyze] no data for config {config} in {log_dir / config}")
            continue
        all_stats[config] = {}
        for point_id, rows in sorted(points.items()):
            _warn_pt_delta(rows, point_id)
            all_stats[config][point_id] = analyze_point(rows)

    if not all_stats:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    print_table(all_stats)

    if args.plot:
        out_dir = log_dir / "plots"
        out_dir.mkdir(exist_ok=True)
        plot_comparison(all_stats, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=os.environ.get("EXP_LOG_DIR", "./results"))
    parser.add_argument("--configs", nargs="*", help="Which configs to analyze (default: all found)")
    parser.add_argument("--plot", action="store_true", help="Save matplotlib figures")
    args = parser.parse_args()
    main(args)
