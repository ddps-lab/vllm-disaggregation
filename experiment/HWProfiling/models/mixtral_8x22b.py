from .base import ExpertSpec, fetch_config, spec_from_fallback

MODEL_KEY = "mixtral-8x22b"
HF_REPO = "mistralai/Mixtral-8x22B-Instruct-v0.1"

# One expert is 604 MB (3*16384*6144 bf16) — rotating all 56 layers would need
# 34+ GB of weights alone, so cap rotation at 8 replicas (4.8 GB, still >> 2xL2).
REPLICA_CAP = 8

FALLBACK = dict(
    hidden_size=6144,
    intermediate_size=16384,
    num_experts=8,
    top_k=2,
    num_moe_layers=56,
    n_shared_experts=0,
    replica_cap=REPLICA_CAP,
)


def load_spec() -> ExpertSpec:
    try:
        cfg = fetch_config(HF_REPO)
    except Exception as e:
        return spec_from_fallback(MODEL_KEY, HF_REPO, FALLBACK, e)
    return ExpertSpec(
        model_key=MODEL_KEY,
        hf_repo=HF_REPO,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_experts=cfg.num_local_experts,
        top_k=cfg.num_experts_per_tok,
        num_moe_layers=cfg.num_hidden_layers,  # every Mixtral layer is MoE
        n_shared_experts=0,
        replica_cap=REPLICA_CAP,
    )
