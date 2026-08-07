import logging
import statistics
from dataclasses import dataclass
from typing import Callable

import torch

logger = logging.getLogger("hwprofiling")


@dataclass
class Replica:
    """One layer's worth of expert weights + activations.

    Rotating across replicas each iteration mimics AFD-style serving where
    every layer streams a different expert's weights through the GPU — this
    defeats L2 residency (e.g. Qwen3's 6 MB w13 fits entirely in L4's 48 MB
    L2, which would otherwise measure L2 bandwidth instead of HBM).
    """

    w13: torch.Tensor  # [2I, K] gate first, up second (MergedColumnParallelLinear)
    w2: torch.Tensor  # [K, I]
    x: torch.Tensor  # [M, K]
    h: torch.Tensor  # [M, 2I]
    a: torch.Tensor  # [M, I]
    y: torch.Tensor  # [M, K]


def weight_nbytes(K: int, I: int, elem: int) -> int:
    return elem * (2 * I * K + K * I)


def replica_nbytes(M: int, K: int, I: int, elem: int) -> int:
    return weight_nbytes(K, I, elem) + elem * (M * K + M * 2 * I + M * I + M * K)


def _new_replica(M: int, K: int, I: int, dtype, device) -> Replica:
    # randn * 0.02: values don't affect GEMM timing, but NaN/inf/subnormals
    # from uninitialized memory can perturb pointwise kernels — never use empty
    # for inputs/weights.
    def rand(*shape):
        return torch.randn(*shape, dtype=dtype, device=device) * 0.02

    return Replica(
        w13=rand(2 * I, K),
        w2=rand(K, I),
        x=rand(M, K),
        h=rand(M, 2 * I),
        a=rand(M, I),
        y=torch.zeros(M, K, dtype=dtype, device=device),
    )


def make_replicas(
    M: int,
    K: int,
    I: int,
    dtype: torch.dtype,
    num_moe_layers: int,
    mem_budget_bytes: int,
    l2_bytes: int,
    device: str = "cuda",
) -> tuple[list[Replica], bool]:
    """Allocate up to num_moe_layers distinct weight+input sets.

    Returns (replicas, rotation_ok). rotation_ok=False means the rotating
    weight footprint does not exceed 2x L2 and bandwidth numbers may reflect
    cache, not HBM.
    """
    elem = torch.finfo(dtype).bits // 8
    per = replica_nbytes(M, K, I, elem)
    target = max(1, min(num_moe_layers, mem_budget_bytes // per))
    if target < num_moe_layers:
        logger.warning(
            "M=%d: clamping replicas %d -> %d (%.2f GB each, budget %.2f GB)",
            M, num_moe_layers, target, per / 1e9, mem_budget_bytes / 1e9,
        )

    replicas: list[Replica] = []
    try:
        for _ in range(target):
            replicas.append(_new_replica(M, K, I, dtype, device))
    except torch.cuda.OutOfMemoryError:
        if replicas:
            replicas.pop()
        logger.warning(
            "M=%d: OOM while allocating replicas; proceeding with %d",
            M, len(replicas),
        )
        torch.cuda.empty_cache()
    if not replicas:
        raise torch.cuda.OutOfMemoryError(
            f"M={M}: cannot fit even one replica ({per / 1e9:.2f} GB)"
        )

    rotation_ok = len(replicas) * weight_nbytes(K, I, elem) >= 2 * l2_bytes
    if not rotation_ok:
        logger.warning(
            "M=%d: rotating weight footprint %.1f MB < 2x L2 (%.1f MB) — "
            "bandwidth may be L2-inflated (rotation_ok=false)",
            M,
            len(replicas) * weight_nbytes(K, I, elem) / 1e6,
            2 * l2_bytes / 1e6,
        )
    return replicas, rotation_ok


@dataclass
class StageTiming:
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    p95_ms: float
    iters: int
    replicas: int

    def as_dict(self) -> dict:
        return {
            "mean": self.mean_ms,
            "median": self.median_ms,
            "std": self.std_ms,
            "min": self.min_ms,
            "p95": self.p95_ms,
        }


def bench_stage(
    fn: Callable[[Replica], None],
    replicas: list[Replica],
    warmup: int,
    iters: int,
) -> StageTiming:
    R = len(replicas)
    # Warmup also triggers cuBLAS heuristic selection, which is per-shape —
    # must rerun for every (stage, M) pair.
    for i in range(warmup):
        fn(replicas[i % R])
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(replicas[i % R])
        ends[i].record()
    torch.cuda.synchronize()

    lats = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return StageTiming(
        mean_ms=statistics.fmean(lats),
        median_ms=statistics.median(lats),
        std_ms=statistics.stdev(lats) if len(lats) > 1 else 0.0,
        min_ms=lats[0],
        p95_ms=lats[min(len(lats) - 1, round(0.95 * (len(lats) - 1)))],
        iters=iters,
        replicas=R,
    )
