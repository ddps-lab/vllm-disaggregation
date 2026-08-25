"""MoE expert FFN hardware micro-benchmark.

Mimics vLLM's per-expert FFN dataflow (w13 gate_up GEMM -> silu_and_mul ->
w2 down GEMM, unfused — identical HBM traffic to the default bf16 triton
fused_moe path) and measures effective FLOPS / memory bandwidth per batch
size on the local GPU. Run on EC2 GPU instances, never locally.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

import profiler
from models import MODEL_REGISTRY

logger = logging.getLogger("hwprofiling")



def parse_args() -> argparse.Namespace:
    """Define and parse the CLI: model selection, sweep range, timing knobs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="all", choices=[*MODEL_REGISTRY.keys(), "all"])
    p.add_argument(
        "--batch-sizes",
        type=lambda s: s if s == "auto" else [int(v) for v in s.split(",")],
        default="auto",
        help="comma-separated token counts M, or 'auto' (default): powers of 2 "
        "from 1 until even one expert no longer fits in VRAM",
    )
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--instance-type", default=None, help="override IMDSv2 detection")
    p.add_argument(
        "--mem-budget-gb",
        type=float,
        default=None,
        help="expert memory cap (default: 0.85 x free VRAM)",
    )
    p.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    p.add_argument("--cooldown", type=float, default=0.0, help="sleep between Ms (s)")
    p.add_argument("--no-shared", action="store_true", help="skip shared-expert FFN")
    p.add_argument("--no-vllm-op", action="store_true", help="skip vLLM silu kernel")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def iter_powers_of_two():
    """Yield 1, 2, 4, ... — the auto sweep stops via the VRAM check in the loop."""
    M = 1
    while True:
        yield M
        M *= 2


def bench_model(model_key: str, args, env: dict, silu_fn, silu_impl: str) -> None:
    """Run the full sweep for one model: load its HF config, then for every
    (expert_kind, M) allocate rotating experts, time the 4 stages, and write
    the metadata JSON + measurement CSV."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = profiler.setup_run_dir(args.results_dir, model_key, env["instance_type"])
    log_handler = profiler.attach_file_log(run_dir, timestamp)
    try:
        spec = MODEL_REGISTRY[model_key]()
        logger.info(
            "%s: K=%d I=%d experts=%d top_k=%d moe_layers=%d shared=%d "
            "rotation_target=%d (%s)",
            spec.model_key, spec.hidden_size, spec.intermediate_size,
            spec.num_experts, spec.top_k, spec.num_moe_layers,
            spec.n_shared_experts, spec.rotation_target, spec.config_source,
        )

        dtype, peaks = env["_dtype"], env["_peaks"]
        elem = torch.finfo(dtype).bits // 8
        free_mem, _ = torch.cuda.mem_get_info()
        budget = (
            int(args.mem_budget_gb * 1e9) if args.mem_budget_gb else int(0.85 * free_mem)
        )
        stages = profiler.make_stages(silu_fn)

        kinds = [("routed", spec.intermediate_size)]
        if spec.n_shared_experts > 0 and not args.no_shared:
            kinds.append(("shared", spec.intermediate_size * spec.n_shared_experts))

        auto = args.batch_sizes == "auto"
        measurements = []
        K = spec.hidden_size
        for kind, I in kinds:
            batch_sizes = iter_powers_of_two() if auto else args.batch_sizes
            for M in batch_sizes:
                profiler.log_gpu_clocks(f"{kind} M={M}")
                try:
                    experts, rotation_ok = profiler.make_experts(
                        M, K, I, dtype, spec.rotation_target, budget,
                        env["l2_cache_bytes"],
                    )
                except torch.cuda.OutOfMemoryError as e:
                    if auto:
                        logger.info(
                            "%s: VRAM limit at M=%d — max M is %d", kind, M, M // 2
                        )
                        break
                    logger.warning("skipping %s M=%d: %s", kind, M, e)
                    continue

                for stage_name, fn in stages.items():
                    timing = profiler.bench_stage(fn, experts, args.warmup, args.iters)
                    flops, nbytes = profiler.stage_flops_bytes(stage_name, M, K, I, elem)
                    derived = profiler.derived_metrics(
                        flops, nbytes, timing["median"], peaks
                    )
                    measurements.append({
                        "expert_kind": kind,
                        "M": M,
                        "stage": stage_name,
                        "latency_ms": timing,
                        **derived,
                        "experts_used": len(experts),
                        "rotation_ok": rotation_ok,
                    })
                    logger.info(
                        "%s M=%-6d %-12s median=%.4fms tflops=%s gbps=%s",
                        kind, M, stage_name, timing["median"],
                        f"{derived['achieved_tflops']:.2f}"
                        if derived["achieved_tflops"] else "-",
                        f"{derived['achieved_gbps']:.1f}"
                        if derived["achieved_gbps"] else "-",
                    )

                del experts
                torch.cuda.empty_cache()
                if args.cooldown > 0:
                    time.sleep(args.cooldown)

        profiler.write_results(run_dir, timestamp, {
            "schema_version": 1,
            "model": {
                "key": spec.model_key,
                "hf_repo": spec.hf_repo,
                "hidden_size": spec.hidden_size,
                "intermediate_size": spec.intermediate_size,
                "num_experts": spec.num_experts,
                "top_k": spec.top_k,
                "num_moe_layers": spec.num_moe_layers,
                "n_shared_experts": spec.n_shared_experts,
                "expert_cap": spec.expert_cap,
                "rotation_target": spec.rotation_target,
                "config_source": spec.config_source,
            },
            "environment": {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **{k: v for k, v in env.items() if not k.startswith("_")},
                "dtype": str(dtype).removeprefix("torch."),
                "silu_impl": silu_impl,
                "peaks": peaks,
            },
            "run_config": {
                "warmup": args.warmup,
                "iters": args.iters,
                "batch_sizes": "auto (powers of 2 up to VRAM limit)"
                if auto else args.batch_sizes,
                "mem_budget_bytes": budget,
                "cooldown_s": args.cooldown,
            },
            "measurements": measurements,
        })
    finally:
        profiler.detach_file_log(log_handler)


def main() -> int:
    """Detect the environment once (instance, GPU, dtype, silu kernel), then
    bench each requested model, continuing past per-model failures."""
    args = parse_args()
    profiler.setup_console_logging(args.verbose)

    if not torch.cuda.is_available():
        logger.error("CUDA GPU required — run this on a GPU instance, not locally")
        return 1

    instance_type, source = profiler.detect_instance_type(args.instance_type)
    info = profiler.gpu_info()
    dtype, dtype_fallback = profiler.select_dtype(args.dtype)
    peaks = profiler.peaks_for(instance_type, info["gpu_name"])
    silu_fn, silu_impl = profiler.resolve_silu_and_mul(disable_vllm=args.no_vllm_op)
    logger.info(
        "instance=%s(%s) gpu=%s dtype=%s silu_impl=%s peaks=%s",
        instance_type, source, info["gpu_name"], dtype, silu_impl,
        peaks["source"] if peaks else "unknown",
    )

    env = {
        "instance_type": instance_type,
        "instance_type_source": source,
        **info,
        "dtype_fallback": dtype_fallback,
        "_dtype": dtype,
        "_peaks": peaks,
    }

    model_keys = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    failed = []
    for key in model_keys:
        try:
            bench_model(key, args, env, silu_fn, silu_impl)
        except Exception:
            logger.exception("model %s failed; continuing", key)
            failed.append(key)
    if failed:
        logger.error("failed models: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
