from . import deepseek_v2_lite, glm_5_2, mixtral_8x22b, qwen3_30b_a3b
from .base import ExpertSpec

MODEL_REGISTRY = {
    m.MODEL_KEY: m.load_spec
    for m in (mixtral_8x22b, qwen3_30b_a3b, deepseek_v2_lite, glm_5_2)
}

__all__ = ["MODEL_REGISTRY", "ExpertSpec"]
