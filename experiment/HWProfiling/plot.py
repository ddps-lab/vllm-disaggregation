"""Line plots of HWProfiling results.

For each results/<model>/<instance_type>/ (latest results_*.json) and each
expert_kind, renders three separate figures under .../plots/:
  <kind>.png          per-stage achieved TFLOPS (left axis, blue, solid) and
                      achieved GB/s (right axis, orange, dashed); dotted lines
                      mark datasheet peaks on their own axis
  <kind>_latency.png  median latency per stage vs M (log y)
  <kind>_memory.png   per-expert memory: weight vs activation buffers vs M
Requires matplotlib only.

Usage: python plot.py [--results-dir PATH]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


def human_gb(v: float) -> str:
    """Format a GB value as a readable KB/MB/GB tick label."""
    if v >= 1:
        return f"{v:.3g} GB"
    if v >= 1e-3:
        return f"{v * 1e3:.3g} MB"
    return f"{v * 1e6:.3g} KB"


def human_ms(v: float) -> str:
    """Format an ms tick as a plain number (0.01, 0.1, 1, 10, ...)."""
    return f"{v:g}"

C_TFLOPS = "#2a78d6"  # blue
C_GBPS = "#eb6834"  # orange
C_MEM_ACT = "#008300"  # green

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

STAGES = ["w13_gemm", "silu_and_mul", "w2_gemm", "full_chain"]
STAGE_LABELS = {
    "w13_gemm": "up&gate (w13)",
    "silu_and_mul": "activation (silu_and_mul)",
    "w2_gemm": "down (w2)",
    "full_chain": "full chain",
}
STAGE_COLORS = {
    "w13_gemm": "#1baf7a",  # aqua
    "silu_and_mul": "#eda100",  # yellow
    "w2_gemm": "#e87ba4",  # magenta
    "full_chain": "#4a3aa7",  # violet
}


def read_measurements_csv(csv_path: Path) -> list[dict]:
    """Load measurement rows from a results CSV into plot-ready dicts."""
    def num(v):
        return float(v) if v not in ("", None) else None

    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "expert_kind": r["expert_kind"],
                "M": int(r["M"]),
                "stage": r["stage"],
                "latency_ms": {"median": float(r["median_ms"])},
                "achieved_tflops": num(r.get("achieved_tflops")),
                "achieved_gbps": num(r.get("achieved_gbps")),
            })
    return rows


def latest_results(results_dir: Path):
    """Yields (model, instance, payload) for the newest config json per
    directory, with measurements loaded from the sibling results_<ts>.csv
    (jsons that still embed measurements are read as-is)."""
    latest: dict[tuple[str, str], tuple[str, dict, Path]] = {}
    for f in results_dir.glob("*/*/config_*.json"):
        payload = json.loads(f.read_text())
        key = (f.parts[-3], f.parts[-2])
        if key not in latest or f.name > latest[key][0]:
            latest[key] = (f.name, payload, f)
    for (model, instance), (_, payload, f) in sorted(latest.items()):
        if "measurements" not in payload:
            ts = f.stem.removeprefix("config_")
            payload["measurements"] = read_measurements_csv(
                f.parent / f"results_{ts}.csv"
            )
        yield model, instance, payload


def series(payload: dict, kind: str, stage: str, field: str):
    """Extract (Ms, values) for one metric of one (expert_kind, stage), M-sorted."""
    pts = sorted(
        (m["M"], m[field])
        for m in payload["measurements"]
        if m["expert_kind"] == kind and m["stage"] == stage and m[field] is not None
    )
    return [p[0] for p in pts], [p[1] for p in pts]


def expert_dims(payload: dict, kind: str) -> tuple[int, int]:
    """Return (K, I) for the expert kind; shared experts widen I by n_shared."""
    info = payload["model"]
    K, I = info["hidden_size"], info["intermediate_size"]
    if kind == "shared":
        I *= max(info.get("n_shared_experts", 1), 1)
    return K, I


def style_axis(ax, title: str):
    """Apply the shared look: log2 x-axis, recessive grid, muted spines."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11, color=TEXT)
    ax.set_xlabel("M (tokens routed to expert)", fontsize=9, color=TEXT_2)
    ax.set_xscale("log", base=2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    for spine in ("left", "bottom", "right"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)


def save(fig, out_path: Path):
    """Save the figure as PNG and close it."""
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_perf(payload: dict, kind: str, title: str, out_path: Path) -> bool:
    """Per-stage subplots: achieved TFLOPS (left axis) + GB/s (right axis) vs M."""
    peaks = payload["environment"].get("peaks")
    fig, axes = plt.subplots(1, len(STAGES), figsize=(4.4 * len(STAGES), 3.8),
                             facecolor=SURFACE)
    drew = False
    for ax, stage in zip(axes, STAGES):
        style_axis(ax, STAGE_LABELS[stage])
        ax_bw = ax.twinx()
        ax_bw.tick_params(colors=TEXT_2, labelsize=8)
        for spine in ax_bw.spines.values():
            spine.set_color(GRID)

        # TFLOPS is meaningless for the pointwise activation — bandwidth only.
        if stage != "silu_and_mul":
            xs, ys = series(payload, kind, stage, "achieved_tflops")
            if xs:
                drew = True
                ax.plot(xs, ys, color=C_TFLOPS, linewidth=2, marker="o",
                        markersize=5)
                if peaks:
                    ax.axhline(peaks["tensor_tflops_dense"], color=C_TFLOPS,
                               linewidth=1, linestyle=":", alpha=0.5)
        xs, ys = series(payload, kind, stage, "achieved_gbps")
        if xs:
            drew = True
            ax_bw.plot(xs, ys, color=C_GBPS, linewidth=2, linestyle="--",
                       marker="s", markersize=4.5)
            if peaks:
                ax_bw.axhline(peaks["hbm_gbps"], color=C_GBPS, linewidth=1,
                              linestyle=":", alpha=0.5)

        if stage == "silu_and_mul":
            ax.set_yticks([])  # pointwise op: bandwidth only, no TFLOPS axis
        else:
            ax.set_ylabel("achieved TFLOPS", fontsize=9, color=C_TFLOPS)
            ax.set_ylim(bottom=0)
        ax_bw.set_ylabel("achieved GB/s", fontsize=9, color=C_GBPS)
        ax_bw.set_ylim(bottom=0)
    if not drew:
        plt.close(fig)
        return False

    fig.legend(
        handles=[
            Line2D([], [], color=C_TFLOPS, linewidth=2, marker="o",
                   markersize=5, label="achieved TFLOPS (left)"),
            Line2D([], [], color=C_GBPS, linewidth=2, linestyle="--",
                   marker="s", markersize=4.5, label="achieved GB/s (right)"),
            Line2D([], [], color=TEXT_2, linewidth=1, linestyle=":",
                   label="datasheet peak"),
        ],
        loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 1.02), fontsize=9, labelcolor=TEXT,
    )
    fig.suptitle(title, fontsize=12, color=TEXT, y=1.08)
    fig.text(0.01, -0.04, "full data in results_*.csv", fontsize=8, color=TEXT_2)
    fig.tight_layout()
    save(fig, out_path)
    return True


def plot_latency(payload: dict, kind: str, title: str, out_path: Path) -> bool:
    """One log-y chart: median latency of all four stages vs M."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style_axis(ax, "median latency per stage")
    ax.set_yscale("log")
    ax.set_ylabel("latency (ms)", fontsize=9, color=TEXT_2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: human_ms(v)))
    drew = False
    for stage in STAGES:
        xs, ys = [], []
        for m in payload["measurements"]:
            if m["expert_kind"] == kind and m["stage"] == stage:
                xs.append(m["M"])
                ys.append(m["latency_ms"]["median"])
        if xs:
            drew = True
            xs, ys = zip(*sorted(zip(xs, ys)))
            ax.plot(xs, ys, color=STAGE_COLORS[stage], linewidth=2,
                    marker="o", markersize=4, label=STAGE_LABELS[stage])
    if not drew:
        plt.close(fig)
        return False
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle(title, fontsize=12, color=TEXT)
    fig.tight_layout()
    save(fig, out_path)
    return True


def plot_memory(payload: dict, kind: str, title: str, out_path: Path) -> bool:
    """Memory needed for ONE expert: its weights (constant) vs the activation
    buffers x/h/a/y that scale with M."""
    Ms = sorted({m["M"] for m in payload["measurements"]
                 if m["expert_kind"] == kind})
    if not Ms:
        return False
    K, I = expert_dims(payload, kind)
    elem = 2  # bf16/fp16
    w_gb = elem * 3 * I * K / 1e9
    act_gb = [elem * M * (2 * K + 3 * I) / 1e9 for M in Ms]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style_axis(ax, "memory for one expert: weight vs activation")
    ax.set_yscale("log")
    ax.set_ylabel("memory", fontsize=9, color=TEXT_2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: human_gb(v)))
    ax.plot(Ms, [w_gb] * len(Ms), color=TEXT_2, linewidth=2, linestyle="--",
            label=f"expert weight = {human_gb(w_gb)}")
    ax.plot(Ms, act_gb, color=C_MEM_ACT, linewidth=2, marker="o", markersize=4,
            label="activation (input + intermediate + output)")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle(title, fontsize=12, color=TEXT)
    fig.tight_layout()
    save(fig, out_path)
    return True


def main() -> None:
    """Render the three figures per (model, instance, expert_kind) found in results/."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path,
                   default=Path(__file__).parent / "results")
    args = p.parse_args()

    wrote = 0
    for model, instance, payload in latest_results(args.results_dir):
        out_dir = args.results_dir / model / instance / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        kinds = sorted({m["expert_kind"] for m in payload["measurements"]})
        for kind in kinds:
            info = payload["model"]
            detail = (
                f"top_k={info['top_k']} of {info['num_experts']} experts"
                if kind == "routed" else "all tokens"
            )
            title = f"{model} — {instance} — {kind} expert ({detail})"
            wrote += plot_perf(payload, kind, title, out_dir / f"{kind}.png")
            wrote += plot_latency(
                payload, kind, title, out_dir / f"{kind}_latency.png"
            )
            wrote += plot_memory(
                payload, kind, title, out_dir / f"{kind}_memory.png"
            )
    if not wrote:
        raise SystemExit(f"no results under {args.results_dir}")


if __name__ == "__main__":
    main()
