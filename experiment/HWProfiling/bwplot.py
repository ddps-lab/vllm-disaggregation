"""Plot bwprofile results: all GPUs on one chart.

y = achieved streaming bandwidth (GB/s, c = a + b), x = accessed memory per
iteration (log scale).
Reads the newest results_*.csv per results/bwprofile/<instance>/.

Usage: python bwplot.py [--results-dir PATH]
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

INSTANCE_COLORS = {
    "g4dn.xlarge": "#2a78d6",  # blue
    "g5.xlarge": "#eb6834",  # orange
    "g6.xlarge": "#1baf7a",  # aqua
    "g6e.xlarge": "#eda100",  # yellow
}

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def human_bytes(b: float) -> str:
    """Format bytes as a readable KB/MB/GB tick label."""
    if b >= 1e9:
        return f"{b / 1e9:g} GB"
    if b >= 1e6:
        return f"{b / 1e6:g} MB"
    return f"{b / 1e3:g} KB"


def main() -> None:
    """Render results/bwprofile/bandwidth_ramp.png from all instance CSVs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path,
                   default=Path(__file__).parent / "results")
    args = p.parse_args()
    base = args.results_dir / "bwprofile"

    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(17, 5.6), facecolor=SURFACE)
    for a in (ax, ax_t):
        a.set_facecolor(SURFACE)
        a.set_xlabel("Accessed Memory Size", fontsize=14, color=TEXT_2)
        a.set_xscale("log")
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: human_bytes(v)))
        a.grid(True, color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        for spine in ("left", "bottom"):
            a.spines[spine].set_color(GRID)
        a.tick_params(colors=TEXT_2, labelsize=13)
    ax.set_title("Achieved Bandwidth", fontsize=15, color=TEXT)
    ax.set_ylabel("Achieved Bandwidth (GB/s)", fontsize=14, color=TEXT_2)
    ax_t.set_title("Latency", fontsize=15, color=TEXT)
    ax_t.set_ylabel("Latency (ms)", fontsize=14, color=TEXT_2)
    ax_t.set_yscale("log")
    ax_t.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    drew = False
    for inst_dir in sorted(base.iterdir() if base.is_dir() else []):
        files = sorted(inst_dir.glob("results_*.csv"))
        if not files:
            continue
        drew = True
        inst = inst_dir.name
        rows = list(csv.DictReader(open(files[-1])))
        xs = [int(r["bytes"]) for r in rows]
        c = INSTANCE_COLORS.get(inst, TEXT_2)
        ax.plot(xs, [float(r["gbps"]) for r in rows], color=c, linewidth=2,
                marker="o", markersize=4, label=inst)
        ax_t.plot(xs, [float(r["median_ms"]) for r in rows], color=c,
                  linewidth=2, marker="o", markersize=4)
    if not drew:
        raise SystemExit(f"no results under {base}")

    ax.legend(frameon=False, fontsize=13, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    out = base / "bandwidth_ramp.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
