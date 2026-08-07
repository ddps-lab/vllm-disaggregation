from .base import ExpertSpec, fetch_config, spec_from_fallback

MODEL_KEY = "qwen3-30b-a3b"
HF_REPO = "Qwen/Qwen3-30B-A3B"

FALLBACK = dict(
    hidden_size=2048,
    intermediate_size=768,
    num_experts=128,
    top_k=8,
    num_moe_layers=48,
    n_shared_experts=0,
)


def _count_moe_layers(cfg) -> int:
    # Mirrors Qwen3MoeSparseMoeBlock placement logic in vLLM's qwen3_moe.py.
    mlp_only = set(getattr(cfg, "mlp_only_layers", None) or [])
    step = getattr(cfg, "decoder_sparse_step", 1)
    return sum(
        1
        for i in range(cfg.num_hidden_layers)
        if i not in mlp_only and cfg.num_experts > 0 and (i + 1) % step == 0
    )


def load_spec() -> ExpertSpec:
    """Fetch the HF config and map its keys to an ExpertSpec (fallback on failure)."""
    try:
        cfg = fetch_config(HF_REPO)
    except Exception as e:
        return spec_from_fallback(MODEL_KEY, HF_REPO, FALLBACK, e)
    return ExpertSpec(
        model_key=MODEL_KEY,
        hf_repo=HF_REPO,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.moe_intermediate_size,
        num_experts=cfg.num_experts,
        top_k=cfg.num_experts_per_tok,
        num_moe_layers=_count_moe_layers(cfg),
        n_shared_experts=0,
    )
