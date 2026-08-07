from .base import ExpertSpec, fetch_config, spec_from_fallback

MODEL_KEY = "deepseek-v2-lite"
HF_REPO = "deepseek-ai/DeepSeek-V2-Lite"

FALLBACK = dict(
    hidden_size=2048,
    intermediate_size=1408,
    num_experts=64,
    top_k=6,
    num_moe_layers=26,  # 27 layers, first_k_dense_replace=1
    n_shared_experts=2,
)


def count_deepseek_moe_layers(cfg) -> int:
    # Mirrors DeepseekV2DecoderLayer MoE placement in vLLM's deepseek_v2.py.
    first_dense = getattr(cfg, "first_k_dense_replace", 0)
    freq = getattr(cfg, "moe_layer_freq", 1)
    return sum(
        1
        for i in range(cfg.num_hidden_layers)
        if i >= first_dense and i % freq == 0
    )


def load_spec() -> ExpertSpec:
    try:
        cfg = fetch_config(HF_REPO, trust_remote_code=True)
    except Exception as e:
        return spec_from_fallback(MODEL_KEY, HF_REPO, FALLBACK, e)
    return ExpertSpec(
        model_key=MODEL_KEY,
        hf_repo=HF_REPO,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.moe_intermediate_size,
        num_experts=cfg.n_routed_experts,
        top_k=cfg.num_experts_per_tok,
        num_moe_layers=count_deepseek_moe_layers(cfg),
        n_shared_experts=getattr(cfg, "n_shared_experts", 0) or 0,
    )
