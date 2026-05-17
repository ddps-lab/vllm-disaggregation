# KV-transfer timing wrapper around LMCacheConnectorV1.
# Used only on the prefill (producer) side of configs C and D.
# Writes one JSON line to $EXP_LOG_DIR/kv_transfer.jsonl per completed save batch.
#
# Register via kv-transfer-config (kv_connector_module_path is REQUIRED):
#   {
#     "kv_connector": "InstrumentedLMCacheConnector",
#     "kv_connector_module_path": "instrumented_connector",
#     "kv_role": "kv_producer",
#     "kv_connector_extra_config": {...}
#   }
# PYTHONPATH must include the disagg-exp/ directory (launch_configs.sh sets this).
#
# Design notes:
#   - Only the rank-0 worker writes to the log to avoid concurrent-write corruption
#     across TP workers (each TP rank is a separate OS process with the same log path).
#   - save_kv_layer is called once per transformer layer per forward batch; _save_start
#     captures the first call and wait_for_save closes the interval.
#   - Decode-only batches never call save_kv_layer, so wait_for_save skips logging.

import json
import os
import time
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

_LOG_PATH = os.path.join(
    os.environ.get("EXP_LOG_DIR", "./results"),
    "kv_transfer.jsonl",
)
_logged_engine_attrs = False


def _write(record: dict) -> None:
    with open(_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

# 공식 LMcacheConnectorV1을 상속 받아서 쓰는거
class InstrumentedLMCacheConnector(LMCacheConnectorV1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        # rank within this vLLM instance (0..TP-1). With TP=2 there are two
        # worker processes; only rank 0 writes to the log to avoid concurrent
        # appends from separate OS processes corrupting JSONL lines.
        self._local_rank: int = vllm_config.parallel_config.rank
        self._save_start: float | None = None
        self._layers_seen: int = 0

        logger.info(
            "InstrumentedLMCacheConnector ready (rank=%d, logging=%s, path=%s)",
            self._local_rank,
            "yes" if self._local_rank == 0 else "no (rank>0)",
            _LOG_PATH,
        )

    # ── worker-side: KV save path ─────────────────────────────────────────────

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if self._save_start is None:
            self._save_start = time.monotonic()
            self._layers_seen = 0
        self._layers_seen += 1
        super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        # Record start before blocking; if _save_start is None this was a
        # decode-only batch — skip logging to avoid polluting the log.
        had_saves = self._save_start is not None
        t0 = self._save_start if had_saves else time.monotonic()
        self._save_start = None
        layers = self._layers_seen
        self._layers_seen = 0

        super().wait_for_save()

        if had_saves and self._local_rank == 0:
            _write({
                "ts_utc": time.time(),
                "event": "wait_for_save_done",
                "duration_ms": round((time.monotonic() - t0) * 1000.0, 2),
                "layers_saved": layers,
            })

    # ── worker-side: completion notification ──────────────────────────────────

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        global _logged_engine_attrs
        result = super().get_finished(finished_req_ids)

        if self._local_rank == 0:
            # Log the lmcache engine's public interface once so we can verify
            # attribute names at runtime without needing to read lmcache source.
            if not _logged_engine_attrs:
                try:
                    attrs = [a for a in dir(self._lmcache_engine) if not a.startswith("_")]
                    logger.info(
                        "connector lmcache_engine type=%s public_attrs=%s",
                        type(self._lmcache_engine).__name__,
                        attrs,
                    )
                    _logged_engine_attrs = True
                except Exception:
                    pass

            if result[0]:
                for req_id in result[0]:
                    _write({
                        "ts_utc": time.time(),
                        "event": "send_finished",
                        "req_id": req_id,
                    })

        return result

    # ── scheduler-side: metadata shape logging ────────────────────────────────

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> Any:
        meta = super().build_connector_meta(scheduler_output)
        # build_connector_meta runs in the scheduler (main) process, not in
        # worker subprocesses. Log its type once; rank guard is irrelevant here
        # since there is only one scheduler process.
        if not _logged_engine_attrs:
            try:
                logger.info(
                    "connector metadata shape: type=%s attrs=%s",
                    type(meta).__name__,
                    [a for a in dir(meta) if not a.startswith("__")],
                )
            except Exception:
                pass
        return meta
