def stage_flops_bytes(stage: str, M: int, K: int, I: int, elem: int) -> tuple[int, int]:
    """Nominal FLOPs and HBM bytes per stage (unfused dataflow, matching vLLM).

    silu_and_mul FLOPs (~5/elem: sigmoid + 2 mul) are nominal — bandwidth is
    the meaningful metric for that stage.
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


def derived_metrics(
    flops: int, nbytes: int, median_ms: float, peaks: dict | None
) -> dict:
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
