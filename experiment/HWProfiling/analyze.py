"""Extract effective roofline parameters from HWProfiling results.

The split between regimes is read directly off the latency curve: its flat
floor is the memory-bound regime (weight streaming, M-independent) and the
rising part the compute-bound regime. The knee — the M where latency first
exceeds KNEE_FACTOR x floor (log-interpolated) — is the empirical boundary.
  BW_eff = median implied bandwidth (bytes/t) over points below the knee
  F_eff  = median implied FLOPS (flops/t) over points above the knee
No fitting; one readable knee per curve. Also reports the effective ridge
F_eff/BW_eff and crossovers: M*_knee (the knee itself), M*_calc (closed form
AI(M*) = effective ridge via the saturation constant M_c), M*_peak (datasheet
ridge). Requires numpy only.

Usage: python analyze.py [--results-dir PATH]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

STAGES = ["w13_gemm", "w2_gemm", "full_chain"]  # silu is bandwidth-only


def stage_flops_bytes(stage: str, M: int, K: int, I: int, elem: int = 2):
    """Same formulas as profiler.stage_flops_bytes (duplicated: no torch here)."""
    w13 = (2 * M * (2 * I) * K, elem * (M * K + 2 * I * K + M * 2 * I))
    act = (5 * M * I, elem * (M * 2 * I + M * I))
    w2 = (2 * M * K * I, elem * (M * I + K * I + M * K))
    table = {
        "w13_gemm": w13,
        "silu_and_mul": act,
        "w2_gemm": w2,
        "full_chain": tuple(sum(v) for v in zip(w13, act, w2)),
    }
    return table[stage]


def ai_saturation(stage: str, K: int, I: int) -> float:
    """M_c such that AI(M) = M*M_c/(M+M_c): the large-M intensity ceiling."""
    if stage == "w13_gemm":
        return 2 * I * K / (K + 2 * I)
    if stage == "w2_gemm":
        return I * K / (I + K)
    return 3 * I * K / (2 * K + 6 * I)  # full_chain


KNEE_FACTOR = 1.5  # latency leaves the floor once it exceeds this multiple


def latency_knee(Ms, lats_ms):
    """Empirical regime boundary read directly off the latency curve: the M
    (log-interpolated) where latency first exceeds KNEE_FACTOR x its floor.
    Returns inf if latency never leaves the floor in range."""
    t = np.array(lats_ms)
    floor = t.min()
    above = np.nonzero(t >= KNEE_FACTOR * floor)[0]
    above = above[above > np.argmin(t)]  # only the rising tail
    if len(above) == 0:
        return float("inf")
    i = above[0]
    lm = np.interp(np.log2(KNEE_FACTOR * floor), np.log2(t[i - 1:i + 1]),
                   np.log2(np.array(Ms[i - 1:i + 1], dtype=float)))
    return 2 ** lm


def effective_roofline(Ms, lats_ms, K, I, stage):
    """Split points at the latency knee; take per-side medians.

    Returns (BW_eff GB/s, F_eff TFLOPS, knee, n_mem, n_cmp); inf when a side
    has no measured points.
    """
    t = np.array(lats_ms) * 1e-3  # seconds
    B = np.array([stage_flops_bytes(stage, M, K, I)[1] for M in Ms], dtype=float)
    F = np.array([stage_flops_bytes(stage, M, K, I)[0] for M in Ms], dtype=float)
    knee = latency_knee(Ms, lats_ms)
    mem = np.array(Ms) < knee
    bw = np.median(B[mem] / t[mem]) / 1e9 if mem.any() else float("inf")
    fl = np.median(F[~mem] / t[~mem]) / 1e12 if (~mem).any() else float("inf")
    return bw, fl, knee, int(mem.sum()), int((~mem).sum())


def latest_runs(results_dir: Path):
    """Newest (config json, measurement csv) per (model, instance)."""
    latest = {}
    for f in results_dir.glob("*/*/config_*.json"):
        key = (f.parts[-3], f.parts[-2])
        if key not in latest or f.name > latest[key].name:
            latest[key] = f
    for (model, instance), f in sorted(latest.items()):
        cfg = json.loads(f.read_text())
        ts = f.stem.removeprefix("config_")
        rows = list(csv.DictReader(open(f.parent / f"results_{ts}.csv")))
        yield model, instance, cfg, rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path,
                   default=Path(__file__).parent / "results")
    args = p.parse_args()

    header = (f"{'model':18s} {'instance':12s} {'kind':6s} {'stage':10s} "
              f"{'n_mem':>5s} {'n_cmp':>5s} {'BW_eff':>7s} {'F_eff':>6s} "
              f"{'ridge':>6s} {'M*_knee':>7s} {'M*_calc':>7s} {'M*_peak':>7s}")
    print(header)
    print("-" * len(header))
    for model, instance, cfg, rows in latest_runs(args.results_dir):
        K = cfg["model"]["hidden_size"]
        peaks = cfg["environment"].get("peaks") or {}
        r_peak = (peaks.get("tensor_tflops_dense", 0) * 1e3 /
                  peaks.get("hbm_gbps", 1)) if peaks else None
        for kind in sorted({r["expert_kind"] for r in rows}):
            I = cfg["model"]["intermediate_size"]
            if kind == "shared":
                I *= max(cfg["model"].get("n_shared_experts", 1), 1)
            for stage in STAGES:
                pts = [(int(r["M"]), float(r["median_ms"])) for r in rows
                       if r["expert_kind"] == kind and r["stage"] == stage]
                if len(pts) < 4:
                    continue
                Ms, lats = zip(*sorted(pts))
                bw, fl, knee, n_mem, n_cmp = effective_roofline(
                    Ms, lats, K, I, stage
                )
                ridge = fl * 1e3 / bw  # FLOP/byte
                mc = ai_saturation(stage, K, I)

                def crossover(r):
                    return r * mc / (mc - r) if r and mc > r else float("inf")

                print(f"{model:18s} {instance:12s} {kind:6s} {stage:10s} "
                      f"{n_mem:5d} {n_cmp:5d} {bw:7.1f} {fl:6.1f} "
                      f"{ridge:6.0f} {knee:7.0f} {crossover(ridge):7.0f} "
                      f"{crossover(r_peak) if r_peak else float('nan'):7.0f}")


if __name__ == "__main__":
    main()
