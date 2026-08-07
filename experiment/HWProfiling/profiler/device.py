import logging
import subprocess
import urllib.request

import torch

logger = logging.getLogger("hwprofiling")

IMDS_BASE = "http://169.254.169.254/latest"

# Dense (non-sparsity) FP16/BF16 tensor-core TFLOPS and memory bandwidth GB/s.
# T4 has no BF16 tensor cores — its number is FP16 TC and the profiler falls
# back to fp16 there (same as vLLM's <SM80 behavior).
PEAK_TABLE = {
    # instance prefix: NVIDIA datasheet values
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
    prefix = instance_label.split(".")[0]
    if prefix in PEAK_TABLE:
        return PEAK_TABLE[prefix]
    for sub, pfx in _GPU_NAME_TO_PREFIX:
        if sub in gpu_name:
            return PEAK_TABLE[pfx]
    return None


def select_dtype(requested: str) -> tuple[torch.dtype, bool]:
    """Returns (dtype, fell_back). auto/bf16 → fp16 on <SM80 (no BF16 TC, e.g. T4)."""
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
