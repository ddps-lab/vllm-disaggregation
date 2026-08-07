from .base import ExpertSpec, fetch_config, spec_from_fallback
from .deepseek_v2_lite import count_deepseek_moe_layers

MODEL_KEY = "glm-5.2"
HF_REPO = "zai-org/GLM-5.2"

FALLBACK = dict(
    hidden_size=6144,
    intermediate_size=2048,
    num_experts=256,
    top_k=8,
    num_moe_layers=75,  # 78 layers, first_k_dense_replace=3
    n_shared_experts=1,
)


def load_spec() -> ExpertSpec:
    """Fetch the HF config (glm_moe_dsa = DeepSeek-style keys) into an ExpertSpec."""
    try:
        cfg = fetch_config(HF_REPO, trust_remote_code="auto")
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
