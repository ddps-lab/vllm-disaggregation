import logging
from typing import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger("hwprofiling")

SiluAndMul = Callable[[torch.Tensor, torch.Tensor], None]  # (out[M,I], x[M,2I])


def _silu_and_mul_eager(out: torch.Tensor, x: torch.Tensor) -> None:
    # Same math as vLLM SiluAndMul: gate is the first half, up the second.
    d = x.shape[-1] // 2
    out.copy_(F.silu(x[..., :d]) * x[..., d:])


def _sanity_check(fn: SiluAndMul, device: str = "cuda") -> bool:
    x = torch.randn(4, 32, device=device, dtype=torch.float16)
    out = torch.empty(4, 16, device=device, dtype=torch.float16)
    ref = torch.empty_like(out)
    try:
        fn(out, x)
    except Exception as e:
        logger.debug("silu_and_mul candidate failed sanity check: %s", e)
        return False
    _silu_and_mul_eager(ref, x)
    return torch.allclose(out, ref, rtol=1e-2, atol=1e-3)


def resolve_silu_and_mul(disable_vllm: bool = False) -> tuple[SiluAndMul, str]:
    """Resolution order: vLLM CUDA op → torch.compile-fused → eager.

    The eager path launches multiple kernels (~5I·M traffic vs the fused
    kernel's 3I·M), inflating the activation-stage bandwidth number — callers
    must record which impl was used.
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

            if _sanity_check(vllm_fn):
                return vllm_fn, "vllm_op"
            logger.warning("torch.ops._C.silu_and_mul failed sanity check; skipping")

    try:
        # dynamic=True: one compile covers all M. Inductor fuses the pointwise
        # chain + copy_ into a single kernel (read 2I·M, write I·M) — same HBM
        # traffic as vLLM's kernel. Never use cudagraph modes here: they break
        # per-iteration event timing and buffer rotation.
        compiled = torch.compile(_silu_and_mul_eager, dynamic=True)
        if _sanity_check(compiled):
            return compiled, "torch_compile"
    except Exception as e:
        logger.warning("torch.compile fallback unavailable: %s", e)

    logger.warning(
        "Using eager silu_and_mul — activation-stage GB/s will be inflated "
        "(extra intermediate traffic); flagged as silu_impl=eager in results."
    )
    return _silu_and_mul_eager, "eager"


def make_stages(silu_fn: SiluAndMul) -> dict[str, Callable]:
    """Stage callables over a Replica, mimicking vLLM's per-expert dataflow:
    w13 GEMM → HBM → silu_and_mul → HBM → w2 GEMM (no activation fusion,
    matching the default bf16 triton fused_moe path)."""

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
