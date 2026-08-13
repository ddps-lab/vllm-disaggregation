"""Pure streaming bandwidth profile: c = a + b over [N, 2048], N doubling.

Maps achieved GB/s vs working-set size — where does each GPU's bandwidth ramp
saturate, and where do expert weight sizes (qwen3 9.5MB vs mixtral 604MB) sit
on that ramp. Buffer sets are rotated to defeat L2 (L40S L2 = 96MB).
Run on EC2 GPU instances, never locally.

Usage: python bwprofile.py [--warmup 10] [--iters 50] [--instance-type L]
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

import profiler

FIXED = 2048  # fixed second dimension


def bench_add(sets, warmup, iters):
    """Median ms of c = a + b, rotating across buffer sets."""
    R = len(sets)
    for i in range(warmup):
        a, b, c = sets[i % R]
        torch.add(a, b, out=c)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        a, b, c = sets[i % R]
        starts[i].record()
        torch.add(a, b, out=c)
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--instance-type", default=None)
    p.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA GPU required")
        return 1
    profiler.setup_console_logging(False)
    instance, _ = profiler.detect_instance_type(args.instance_type)
    info = profiler.gpu_info()
    dtype, _ = profiler.select_dtype("auto")
    peaks = profiler.peaks_for(instance, info["gpu_name"])
    peak_bw = peaks["hbm_gbps"] if peaks else None
    l2 = info["l2_cache_bytes"]
    budget = int(0.85 * torch.cuda.mem_get_info()[0])

    print(f"{instance} {info['gpu_name']} L2={l2/1e6:.0f}MB peak={peak_bw}GB/s")
    print(f"{'N':>9} {'bytes':>9} {'sets':>4} {'t(ms)':>9} {'GB/s':>7} {'%peak':>6}")
    rows = []
    N = 1
    while True:
        set_bytes = 3 * N * FIXED * 2  # read a + read b + write c, bf16/fp16
        R = max(1, min(64, 4 * l2 // set_bytes + 1, budget // set_bytes))
        if set_bytes > budget:
            break
        sets = [
            (torch.randn(N, FIXED, dtype=dtype, device="cuda"),
             torch.randn(N, FIXED, dtype=dtype, device="cuda"),
             torch.empty(N, FIXED, dtype=dtype, device="cuda"))
            for _ in range(R)
        ]
        t = bench_add(sets, args.warmup, args.iters)
        gbps = set_bytes / (t * 1e-3) / 1e9
        pct = f"{gbps / peak_bw * 100:5.1f}%" if peak_bw else "     -"
        print(f"{N:>9} {set_bytes/1e6:8.1f}M {R:>4} {t:9.4f} {gbps:7.1f} {pct}")
        rows.append({"N": N, "bytes": set_bytes, "sets": R,
                     "median_ms": round(t, 6), "gbps": round(gbps, 1)})
        del sets
        torch.cuda.empty_cache()
        N *= 2

    out_dir = args.results_dir / "bwprofile" / instance
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"results_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
