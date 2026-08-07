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

from models import MODEL_REGISTRY
from profiler import bench, device, metrics, ops, report

logger = logging.getLogger("hwprofiling")

DEFAULT_BATCH_SIZES = [2**i for i in range(13)]  # 1 .. 4096


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="all",
        choices=[*MODEL_REGISTRY.keys(), "all"],
    )
    p.add_argument(
        "--batch-sizes",
        type=lambda s: [int(v) for v in s.split(",")],
        default=DEFAULT_BATCH_SIZES,
        help="comma-separated token counts M (default: 1,2,...,4096)",
    )
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--instance-type", default=None, help="override IMDSv2 detection")
    p.add_argument(
        "--mem-budget-gb",
        type=float,
        default=None,
        help="replica memory cap (default: 0.85 x free VRAM)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    p.add_argument("--cooldown", type=float, default=0.0, help="sleep between Ms (s)")
    p.add_argument("--no-shared", action="store_true", help="skip shared-expert FFN")
    p.add_argument("--no-vllm-op", action="store_true", help="skip vLLM silu kernel")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def bench_model(model_key: str, args, env: dict, silu_fn, silu_impl: str) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = report.setup_run_dir(args.results_dir, model_key, env["instance_type"])
    log_handler = report.attach_file_log(run_dir, timestamp)
    try:
        spec = MODEL_REGISTRY[model_key]()
        logger.info(
            "%s: K=%d I=%d experts=%d top_k=%d moe_layers=%d shared=%d "
            "rotation_target=%d (%s)",
            spec.model_key, spec.hidden_size, spec.intermediate_size,
            spec.num_experts, spec.top_k, spec.num_moe_layers,
            spec.n_shared_experts, spec.rotation_target, spec.config_source,
        )

        dtype = env["_dtype"]
        elem = torch.finfo(dtype).bits // 8
        peaks = env["_peaks"]
        free_mem, _ = torch.cuda.mem_get_info()
        budget = (
            int(args.mem_budget_gb * 1e9)
            if args.mem_budget_gb
            else int(0.85 * free_mem)
        )
        l2 = env["l2_cache_bytes"]
        stages = ops.make_stages(silu_fn)

        kinds = [("routed", spec.intermediate_size)]
        if spec.n_shared_experts > 0 and not args.no_shared:
            kinds.append(
                ("shared", spec.intermediate_size * spec.n_shared_experts)
            )

        measurements = []
        for kind, I in kinds:
            K = spec.hidden_size
            for M in args.batch_sizes:
                device.log_gpu_clocks(f"{kind} M={M}")
                try:
                    replicas, rotation_ok = bench.make_replicas(
                        M, K, I, dtype, spec.rotation_target, budget, l2
                    )
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning("skipping %s M=%d: %s", kind, M, e)
                    continue

                for stage_name, fn in stages.items():
                    timing = bench.bench_stage(fn, replicas, args.warmup, args.iters)
                    flops, nbytes = metrics.stage_flops_bytes(stage_name, M, K, I, elem)
                    derived = metrics.derived_metrics(
                        flops, nbytes, timing.median_ms, peaks
                    )
                    if (
                        peaks
                        and derived["achieved_gbps"] is not None
                        and derived["achieved_gbps"] > 1.15 * peaks["hbm_gbps"]
                    ):
                        logger.warning(
                            "%s %s M=%d: %.0f GB/s exceeds HBM peak %.0f — "
                            "likely L2-cached (insufficient rotation)",
                            kind, stage_name, M,
                            derived["achieved_gbps"], peaks["hbm_gbps"],
                        )
                    measurements.append({
                        "expert_kind": kind,
                        "M": M,
                        "stage": stage_name,
                        "latency_ms": timing.as_dict(),
                        "flops": flops,
                        "bytes": nbytes,
                        **derived,
                        "replicas_used": timing.replicas,
                        "rotation_ok": rotation_ok,
                    })
                    logger.info(
                        "%s M=%-5d %-12s median=%.4fms tflops=%s gbps=%s",
                        kind, M, stage_name, timing.median_ms,
                        f"{derived['achieved_tflops']:.2f}"
                        if derived["achieved_tflops"] else "-",
                        f"{derived['achieved_gbps']:.1f}"
                        if derived["achieved_gbps"] else "-",
                    )

                del replicas
                torch.cuda.empty_cache()
                if args.cooldown > 0:
                    time.sleep(args.cooldown)

        payload = {
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
                "replica_cap": spec.replica_cap,
                "rotation_target": spec.rotation_target,
                "config_source": spec.config_source,
            },
            "environment": {
                "timestamp_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                **{k: v for k, v in env.items() if not k.startswith("_")},
                "dtype": str(dtype).removeprefix("torch."),
                "silu_impl": silu_impl,
                "peaks": peaks,
            },
            "run_config": {
                "warmup": args.warmup,
                "iters": args.iters,
                "batch_sizes": args.batch_sizes,
                "mem_budget_bytes": budget,
                "cooldown_s": args.cooldown,
            },
            "measurements": measurements,
        }
        report.write_results(run_dir, timestamp, payload)
    finally:
        report.detach_file_log(log_handler)


def main() -> int:
    args = parse_args()
    report.setup_console_logging(args.verbose)

    if not torch.cuda.is_available():
        logger.error("CUDA GPU required — run this on a GPU instance, not locally")
        return 1

    instance_type, source = device.detect_instance_type(args.instance_type)
    info = device.gpu_info()
    dtype, dtype_fallback = device.select_dtype(args.dtype)
    peaks = device.peaks_for(instance_type, info["gpu_name"])
    silu_fn, silu_impl = ops.resolve_silu_and_mul(disable_vllm=args.no_vllm_op)
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
