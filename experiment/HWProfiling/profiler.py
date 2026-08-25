"""Hardware/measurement layer for the MoE expert FFN micro-benchmark.

Sections: GPU/instance detection -> expert FFN ops -> expert rotation ->
timing -> FLOPs/bytes formulas -> result output.
"""

import csv
import json
import logging
import statistics
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger("hwprofiling")


# ---------------------------------------------------------------- GPU / instance

IMDS_BASE = "http://169.254.169.254/latest"

# Dense (non-sparsity) FP16/BF16 tensor-core TFLOPS and memory bandwidth GB/s,
# from NVIDIA datasheets. T4 has no BF16 tensor cores — its number is FP16 TC
# and the profiler falls back to fp16 there (same as vLLM's <SM80 behavior).
PEAK_TABLE = {
    "g4dn": {"gpu": "Tesla T4", "tensor_tflops_dense": 65.0, "hbm_gbps": 320.0,
             "source": "NVIDIA T4 datasheet (FP16 TC 65 dense, GDDR6 320 GB/s)"},
    "g5": {"gpu": "NVIDIA A10G", "tensor_tflops_dense": 70.0, "hbm_gbps": 600.0,
           "source": "NVIDIA A10G datasheet (FP16 TC 70 dense / 140 sparse, GDDR6 600 GB/s)"},
    "g6": {"gpu": "NVIDIA L4", "tensor_tflops_dense": 121.0, "hbm_gbps": 300.0,
           "source": "NVIDIA L4 datasheet (FP16 TC 121 dense / 242 sparse, GDDR6 300 GB/s)"},
    "g6e": {"gpu": "NVIDIA L40S", "tensor_tflops_dense": 362.0, "hbm_gbps": 864.0,
            "source": "NVIDIA L40S datasheet (FP16 TC 362 dense / 733 sparse, GDDR6 864 GB/s)"},
}

# Ordered: longer substrings first so "L4" does not shadow "L40S".
_GPU_NAME_TO_PREFIX = [("L40S", "g6e"), ("A10G", "g5"), ("T4", "g4dn"), ("L4", "g6")]


def _imds_instance_type(timeout: float = 1.0) -> str:
    """Query the EC2 IMDSv2 metadata endpoint for the instance type."""
    req = urllib.request.Request(
        f"{IMDS_BASE}/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        token = r.read().decode()
    req = urllib.request.Request(
        f"{IMDS_BASE}/meta-data/instance-type",
        headers={"X-aws-ec2-metadata-token": token},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def detect_instance_type(cli_override: str | None) -> tuple[str, str]:
    """Returns (label, source) with source in {"cli", "imds", "gpu-name"}."""
    if cli_override:
        return cli_override, "cli"
    try:
        return _imds_instance_type(), "imds"
    except Exception as e:
        logger.info("IMDSv2 unavailable (%s); falling back to GPU-name label", e)
    name = torch.cuda.get_device_name(0)
    for sub, prefix in _GPU_NAME_TO_PREFIX:
        if sub in name:
            return f"gpu-{sub.lower()}", "gpu-name"
    return "gpu-" + name.lower().replace(" ", "-"), "gpu-name"


def peaks_for(instance_label: str, gpu_name: str) -> dict | None:
    """Look up datasheet peaks by instance prefix, then by GPU name substring."""
    prefix = instance_label.split(".")[0]
    if prefix in PEAK_TABLE:
        return PEAK_TABLE[prefix]
    for sub, pfx in _GPU_NAME_TO_PREFIX:
        if sub in gpu_name:
            return PEAK_TABLE[pfx]
    return None


def select_dtype(requested: str) -> tuple[torch.dtype, bool]:
    """Returns (dtype, fell_back). auto/bf16 -> fp16 on <SM80 (no BF16 TC, e.g. T4)."""
    cap = torch.cuda.get_device_capability(0)
    if requested == "fp16":
        return torch.float16, False
    if cap >= (8, 0):
        return torch.bfloat16, False
    logger.warning(
        "GPU compute capability %s < (8,0): no BF16 tensor cores — using fp16 "
        "(mirrors vLLM's dtype policy on this hardware)", cap,
    )
    return torch.float16, requested != "auto"


def gpu_info() -> dict:
    """Collect GPU/toolchain facts recorded in the results metadata."""
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "l2_cache_bytes": props.L2_cache_size,
        "total_mem_bytes": props.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def log_gpu_clocks(tag: str) -> None:
    """Best-effort thermal/clock snapshot via nvidia-smi (T4 throttles easily)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        logger.debug("[%s] sm_clock/temp/power: %s", tag, out)
    except Exception:
        pass


# ---------------------------------------------------------------- expert FFN ops

SiluAndMul = Callable[[torch.Tensor, torch.Tensor], None]  # (out[M,I], x[M,2I])


def _silu_and_mul_eager(out: torch.Tensor, x: torch.Tensor) -> None:
    """out = silu(gate) * up — same math as vLLM SiluAndMul (gate = first half)."""
    d = x.shape[-1] // 2
    out.copy_(F.silu(x[..., :d]) * x[..., d:])


def _silu_sanity_ok(fn: SiluAndMul) -> bool:
    """Check a silu_and_mul candidate against the eager reference on a tiny input."""
    x = torch.randn(4, 32, device="cuda", dtype=torch.float16)
    out = torch.empty(4, 16, device="cuda", dtype=torch.float16)
    ref = torch.empty_like(out)
    try:
        fn(out, x)
    except Exception as e:
        logger.debug("silu_and_mul candidate failed sanity check: %s", e)
        return False
    _silu_and_mul_eager(ref, x)
    return torch.allclose(out, ref, rtol=1e-2, atol=1e-3)


def resolve_silu_and_mul(disable_vllm: bool = False) -> tuple[SiluAndMul, str]:
    """Resolution order: vLLM CUDA op -> torch.compile-fused -> eager.

    The eager path launches multiple kernels (~5I/3I extra traffic), inflating
    the activation-stage bandwidth number — the chosen impl is recorded.
    """
    if not disable_vllm:
        for mod in ("vllm._C", "vllm._custom_ops"):
            try:
                __import__(mod)
                break
            except Exception:
                continue
        op = getattr(getattr(torch.ops, "_C", None), "silu_and_mul", None)
        if op is not None:
            def vllm_fn(out: torch.Tensor, x: torch.Tensor) -> None:
                op(out, x)

            if _silu_sanity_ok(vllm_fn):
                return vllm_fn, "vllm_op"
            logger.warning("torch.ops._C.silu_and_mul failed sanity check; skipping")

    try:
        # dynamic=True: one compile covers all M. Inductor fuses the pointwise
        # chain + copy_ into a single kernel — same HBM traffic as the vLLM
        # kernel. Never use cudagraph modes (breaks event timing + rotation).
        compiled = torch.compile(_silu_and_mul_eager, dynamic=True)
        if _silu_sanity_ok(compiled):
            return compiled, "torch_compile"
    except Exception as e:
        logger.warning("torch.compile fallback unavailable: %s", e)

    logger.warning(
        "Using eager silu_and_mul — activation-stage GB/s will be inflated; "
        "flagged as silu_impl=eager in results."
    )
    return _silu_and_mul_eager, "eager"


def make_stages(silu_fn: SiluAndMul) -> dict[str, Callable]:
    """Stage callables over an Expert, mimicking vLLM's per-expert dataflow:
    w13 GEMM -> HBM -> silu_and_mul -> HBM -> w2 GEMM (unfused, matching the
    default bf16 triton fused_moe path)."""

    def w13_gemm(r):
        torch.matmul(r.x, r.w13.t(), out=r.h)  # [M,K]@[K,2I] -> [M,2I]

    def silu_and_mul(r):
        silu_fn(r.a, r.h)  # [M,2I] -> [M,I]

    def w2_gemm(r):
        torch.matmul(r.a, r.w2.t(), out=r.y)  # [M,I]@[I,K] -> [M,K]

    def full_chain(r):
        w13_gemm(r)
        silu_and_mul(r)
        w2_gemm(r)

    return {
        "w13_gemm": w13_gemm,
        "silu_and_mul": silu_and_mul,
        "w2_gemm": w2_gemm,
        "full_chain": full_chain,
    }


# ---------------------------------------------------------------- expert rotation

@dataclass
class Expert:
    """One layer's worth of expert weights + activations.

    Rotating across experts each iteration mimics AFD-style serving where
    every layer streams a different expert's weights through the GPU — this
    defeats L2 (and TLB) residency that a single reused buffer would have.
    """

    w13: torch.Tensor  # [2I, K] gate first, up second (MergedColumnParallelLinear)
    w2: torch.Tensor  # [K, I]
    x: torch.Tensor  # [M, K]
    h: torch.Tensor  # [M, 2I]
    a: torch.Tensor  # [M, I]
    y: torch.Tensor  # [M, K]


def weight_nbytes(K: int, I: int, elem: int) -> int:
    """Bytes of one expert's weights (w13 + w2)."""
    return elem * (2 * I * K + K * I)


def expert_nbytes(M: int, K: int, I: int, elem: int) -> int:
    """Bytes of one full expert: weights + activation buffers x/h/a/y."""
    return weight_nbytes(K, I, elem) + elem * (M * K + M * 2 * I + M * I + M * K)


def _new_expert(M: int, K: int, I: int, dtype, device) -> Expert:
    """Allocate one expert with random weights/inputs.

    randn * 0.02: values don't affect GEMM timing, but NaN/inf/subnormals
    from uninitialized memory can perturb pointwise kernels.
    """
    def rand(*shape):
        return torch.randn(*shape, dtype=dtype, device=device) * 0.02

    return Expert(
        w13=rand(2 * I, K),
        w2=rand(K, I),
        x=rand(M, K),
        h=rand(M, 2 * I),
        a=rand(M, I),
        y=torch.zeros(M, K, dtype=dtype, device=device),
    )


def make_experts(
    M: int,
    K: int,
    I: int,
    dtype: torch.dtype,
    num_moe_layers: int,
    mem_budget_bytes: int,
    l2_bytes: int,
    device: str = "cuda",
) -> tuple[list[Expert], bool]:
    """Allocate up to num_moe_layers distinct weight+input sets.

    Returns (experts, rotation_ok). rotation_ok=False means the rotating
    weight footprint does not exceed 2x L2 and bandwidth numbers may reflect
    cache, not HBM.
    """
    elem = torch.finfo(dtype).bits // 8
    per = expert_nbytes(M, K, I, elem)
    target = max(1, min(num_moe_layers, mem_budget_bytes // per))
    if target < num_moe_layers:
        logger.warning(
            "M=%d: clamping rotated layers %d -> %d (%.2f GB each, budget %.2f GB)",
            M, num_moe_layers, target, per / 1e9, mem_budget_bytes / 1e9,
        )

    experts: list[Expert] = []
    try:
        for _ in range(target):
            experts.append(_new_expert(M, K, I, dtype, device))
    except torch.cuda.OutOfMemoryError:
        if experts:
            experts.pop()
        logger.warning(
            "M=%d: OOM while allocating experts; proceeding with %d",
            M, len(experts),
        )
        torch.cuda.empty_cache()
    if not experts:
        raise torch.cuda.OutOfMemoryError(
            f"M={M}: cannot fit even one expert ({per / 1e9:.2f} GB)"
        )

    rotation_ok = len(experts) * weight_nbytes(K, I, elem) >= 2 * l2_bytes
    if not rotation_ok:
        logger.warning(
            "M=%d: rotating weight footprint %.1f MB < 2x L2 (%.1f MB) — "
            "bandwidth may be L2-inflated (rotation_ok=false)",
            M,
            len(experts) * weight_nbytes(K, I, elem) / 1e6,
            2 * l2_bytes / 1e6,
        )
    return experts, rotation_ok


# ---------------------------------------------------------------- timing

def bench_stage(
    fn: Callable[[Expert], None],
    experts: list[Expert],
    warmup: int,
    iters: int,
) -> dict:
    """Returns latency stats in ms: {mean, median, std, min, p95}."""
    R = len(experts)
    # Warmup also triggers cuBLAS heuristic selection, which is per-shape —
    # must rerun for every (stage, M) pair.
    for i in range(warmup):
        fn(experts[i % R])
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(experts[i % R])
        ends[i].record()
    torch.cuda.synchronize()

    lats = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return {
        "mean": statistics.fmean(lats),
        "median": statistics.median(lats),
        "std": statistics.stdev(lats) if len(lats) > 1 else 0.0,
        "min": lats[0],
        "p95": lats[min(len(lats) - 1, round(0.95 * (len(lats) - 1)))],
    }


# ---------------------------------------------------------------- FLOPs / bytes

def stage_flops_bytes(stage: str, M: int, K: int, I: int, elem: int) -> tuple[int, int]:
    """Nominal FLOPs and HBM bytes per stage (unfused dataflow, matching vLLM).

    silu_and_mul FLOPs (~5/elem) are nominal — bandwidth is the meaningful
    metric for that stage.
    """
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


def derived_metrics(flops: int, nbytes: int, median_ms: float, peaks: dict | None) -> dict:
    """Turn (FLOPs, bytes, latency) into achieved TFLOPS / GB/s / peak utilization."""
    sec = median_ms / 1e3
    tflops = flops / sec / 1e12 if sec > 0 else None
    gbps = nbytes / sec / 1e9 if sec > 0 else None
    out = {"achieved_tflops": tflops, "achieved_gbps": gbps}
    if peaks and tflops is not None:
        out["compute_util"] = tflops / peaks["tensor_tflops_dense"]
        out["mem_bw_util"] = gbps / peaks["hbm_gbps"]
    else:
        out["compute_util"] = None
        out["mem_bw_util"] = None
    return out


# ---------------------------------------------------------------- output

CSV_FIELDS = [
    "expert_kind", "M", "stage",
    "median_ms", "mean_ms", "min_ms", "p95_ms",
    "achieved_tflops", "achieved_gbps", "compute_util", "mem_bw_util",
    "experts_used", "rotation_ok",
]


def setup_console_logging(verbose: bool) -> None:
    """Attach the stdout handler to the shared "hwprofiling" logger (idempotent)."""
    root = logging.getLogger("hwprofiling")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(h)


def setup_run_dir(results_base: Path, model_key: str, instance_label: str) -> Path:
    """Create (if needed) and return results/<model>/<instance_type>/."""
    run_dir = results_base / model_key / instance_label
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def attach_file_log(run_dir: Path, timestamp: str) -> logging.Handler:
    """Start mirroring the run's log into run_<timestamp>.log."""
    handler = logging.FileHandler(run_dir / f"run_{timestamp}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger("hwprofiling").addHandler(handler)
    return handler


def detach_file_log(handler: logging.Handler) -> None:
    """Stop and close the per-run file log handler."""
    logging.getLogger("hwprofiling").removeHandler(handler)
    handler.close()


def write_results(run_dir: Path, timestamp: str, payload: dict) -> None:
    """config_<ts>.json holds run config/metadata; results_<ts>.csv the measurements."""
    measurements = payload.pop("measurements")

    json_path = run_dir / f"config_{timestamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", json_path)

    csv_path = run_dir / f"results_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for m in measurements:
            row = {k: m.get(k) for k in CSV_FIELDS if k in m}
            lat = m["latency_ms"]
            row.update(
                median_ms=round(lat["median"], 6),
                mean_ms=round(lat["mean"], 6),
                min_ms=round(lat["min"], 6),
                p95_ms=round(lat["p95"], 6),
            )
            for k in ("achieved_tflops", "achieved_gbps", "compute_util", "mem_bw_util"):
                if m.get(k) is not None:
                    row[k] = round(m[k], 4)
            writer.writerow(row)
    logger.info("wrote %s", csv_path)
