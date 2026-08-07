import logging
from dataclasses import dataclass

logger = logging.getLogger("hwprofiling")


@dataclass(frozen=True)
class ExpertSpec:
    model_key: str
    hf_repo: str
    hidden_size: int  # K
    intermediate_size: int  # I (routed expert FFN width)
    num_experts: int
    top_k: int
    num_moe_layers: int  # replica rotation count (AFD-style layer streaming)
    n_shared_experts: int = 0  # shared expert FFN width = I * n_shared_experts
    replica_cap: int | None = None  # cap rotation below num_moe_layers (huge experts)
    config_source: str = "hf_autoconfig"

    @property
    def rotation_target(self) -> int:
        if self.replica_cap is not None:
            return min(self.num_moe_layers, self.replica_cap)
        return self.num_moe_layers


def fetch_config(hf_repo: str, trust_remote_code: bool | str = False):
    """Fetch model config from HuggingFace (config only, no weights).

    trust_remote_code="auto" tries without first, then retries with it.
    """
    from transformers import AutoConfig

    if trust_remote_code == "auto":
        try:
            return AutoConfig.from_pretrained(hf_repo, trust_remote_code=False)
        except Exception as e:
            logger.info(
                "%s: AutoConfig without trust_remote_code failed (%s); "
                "retrying with trust_remote_code=True",
                hf_repo,
                e,
            )
            return AutoConfig.from_pretrained(hf_repo, trust_remote_code=True)
    return AutoConfig.from_pretrained(hf_repo, trust_remote_code=trust_remote_code)


def spec_from_fallback(model_key: str, hf_repo: str, fallback: dict, err) -> ExpertSpec:
    logger.warning(
        "%s: HF config fetch failed (%s) — using hardcoded fallback dims. "
        "Results JSON will record config_source=fallback_dims.",
        model_key,
        err,
    )
    return ExpertSpec(
        model_key=model_key, hf_repo=hf_repo, config_source="fallback_dims", **fallback
    )
