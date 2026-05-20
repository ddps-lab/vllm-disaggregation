#!/bin/bash
# Launch vllm serve for each config (A/B/C/D) and role.
#
# Usage:
#   bash launch_configs.sh configA1           # monolithic 4×T4, TP=2 PP=2 (default A)
#   bash launch_configs.sh configA2           # monolithic 4×T4, TP=4 PP=1
#   bash launch_configs.sh configA3           # monolithic 4×T4, TP=1 PP=4
#   bash launch_configs.sh configA            # alias for configA1
#   bash launch_configs.sh configB            # monolithic 1×L40S
#   bash launch_configs.sh configC1 prefill   # same-node PD TP=2 PP=1, prefill side (default C)
#   bash launch_configs.sh configC1 decode    # same-node PD TP=2 PP=1, decode side
#   bash launch_configs.sh configC1 proxy     # launch disagg proxy (after both)
#   bash launch_configs.sh configC2 prefill   # same-node PD TP=1 PP=2, prefill side
#   bash launch_configs.sh configC2 decode    # same-node PD TP=1 PP=2, decode side
#   bash launch_configs.sh configC2 proxy     # launch disagg proxy
#   bash launch_configs.sh configC ...        # alias for configC1
#   bash launch_configs.sh configD prefill    # cross-node PD, prefill side
#   bash launch_configs.sh configD decode     # cross-node PD, decode side
#   bash launch_configs.sh configD proxy      # launch disagg proxy
#
# Environment overrides (before calling this script):
#   MODEL                 default: meta-llama/Llama-3.1-8B-Instruct
#   DECODER_HOST          for D decode role — the private IP of the decode node
#   MAX_MODEL_LEN         default: 4096
#   GPU_MEM_UTIL          default: 0.85
#   MAX_NUM_SEQS          default: 512   (vLLM stock default is 128; raised for this experiment)
#   EXP_LOG_DIR           default: ./results
#   PYTHONHASHSEED        default: 123  (must match on prefill+decode)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROXY_SCRIPT="$REPO_ROOT/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_proxy_server.py"
CONN_SCRIPT="$REPO_ROOT/disagg-exp/instrumented_connector.py"

# ${변수명:-기본값}: 쉘 스크립트의 매우 유용한 문법입니다. 만약 밖에서 MODEL이라는 환경변수가 들어왔으면 그걸 쓰고, 비어있으면(-) 뒤에 적힌 기본값을 쓰라는 뜻입니다.
# 파이썬의 os.environ.get('MODEL', '기본값')과 완전히 같습니다.
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
LOG_DIR="${EXP_LOG_DIR:-./results}"
# PYTHONHASHSEED가 설정 안되어 있으면 기본값으로 123을 설정. 이 시드는 동일한 시드로 실행되어야 재현가능성이 보장됨. 
export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"

mkdir -p "$LOG_DIR"

CONFIG="${1:-}"
ROLE="${2:-}"

# -z "$CONFIG": CONFIG 변수의 문자열 길이가 0인지(Zero) 체크합니다. 만약 아무 인자도 안 주고 스크립트만 실행했다면, 사용법(Usage)을 친절하게 출력하고 프로그램 상태코드 1(에러)로 종료합니다.
if [[ -z "$CONFIG" ]]; then
    echo "Usage: $0 <configA|configB|configC|configD> [prefill|decode|proxy]"
    exit 1
fi

# ── dtype: explicit float16 for cross-config comparison ──────────────────────
# All configs use float16 (half) for fair comparison.
# T4 (CC 7.5) native, L40S/L4 (CC 8.9+) downgrade from bfloat16.
# Mixing dtype across configs would confound dtype effect with throughput/latency.
DTYPE="half"
echo "[launch] using --dtype half (float16) for all configs"

# ── common flags ──────────────────────────────────────────────────────────────
# Bash의 배열(Array) 문법입니다. 파이썬의 list와 같습니다.
#--no-enable-prefix-caching: KV 캐시 재사용 기능을 끕니다. 이 실험은 'Prefill에서 Decode로 KV 캐시를 넘기는 것'이 메인인데, 로컬 캐시가 켜져 있으면 데이터가 오염될 수 있기 때문에
COMMON_FLAGS=(
    --model "$MODEL"
    --served-model-name llama-3.1-8b
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --no-enable-prefix-caching
    --dtype "$DTYPE"
    --max-num-seqs "${MAX_NUM_SEQS:-512}"
)
# PD connectors need the instrumented connector on PYTHONPATH
# : LMCache의 고성능 네트워크 백엔드인 NIXL 기능을 켜기 위한 필수 환경변수입니다.
export PYTHONPATH="$REPO_ROOT/disagg-exp:${PYTHONPATH:-}"
export LMCACHE_USE_EXPERIMENTAL=True
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ── Config A — monolithic 4×T4. Three TP/PP variants on the same hardware ────
configA1() {
    echo "[launch] configA1: monolithic 4×T4, TP=2 PP=2, port 8000"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    vllm serve "${COMMON_FLAGS[@]}" \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 2 \
        --port 8000 \
        2>&1 | tee "$LOG_DIR/vllm_configA1_$(hostname).log"
}

configA2() {
    echo "[launch] configA2: monolithic 4×T4, TP=4 PP=1, port 8000"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    vllm serve "${COMMON_FLAGS[@]}" \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 1 \
        --port 8000 \
        2>&1 | tee "$LOG_DIR/vllm_configA2_$(hostname).log"
}

configA3() {
    echo "[launch] configA3: monolithic 4×T4, TP=1 PP=4, port 8000"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    vllm serve "${COMMON_FLAGS[@]}" \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 4 \
        --port 8000 \
        2>&1 | tee "$LOG_DIR/vllm_configA3_$(hostname).log"
}

# Back-compat alias: configA → configA1 (TP=2 PP=2 default).
configA() { configA1; }

# ── Config B — monolithic 1×L40S ─────────────────────────────────────────────
configB() {
    echo "[launch] configB: monolithic 1×L40S, port 8000"
    CUDA_VISIBLE_DEVICES=0 \
    vllm serve "${COMMON_FLAGS[@]}" \
        --tensor-parallel-size 1 \
        --port 8000 \
        2>&1 | tee "$LOG_DIR/vllm_configB_$(hostname).log"
}

# ── Config C — same-node PD, GPU 0-1 prefill, GPU 2-3 decode ────────────────
# UCX_TLS: cuda_ipc doesn't work when CUDA_VISIBLE_DEVICES splits the bus;
# use shm (host pinned shared mem) + cuda_copy as the fast path.

# LM cache 라이브러리가 설정파일 yaml을 자동으로 읽어서 실행하는데, 그떄 yaml파일을 여기서 만들어주고 환경변수 -> LMCACHE_CONFIG_FILE="$cfg" 로 넘겨줌 만드는것들은 아래와같음
# vllm-disagg/disagg-exp/launch_configs.sh
# local_cpu: False → CPU 사용 안함
# max_local_cpu_size: 0
# max_local_disk_size: 0
# remote_serde: NULL → 직렬화 사용 안함
# enable_nixl: True → NIXL 사용
# nixl_role: "sender" 또는 "receiver" → sender는 prefill, receiver는 decode
# nixl_peer_host: "127.0.0.1" → 같은 노드이므로 localhost
# nixl_peer_port: 55555 → 통신 포트
# nixl_buffer_size: 1073741824 → 버퍼 크기 1GB
# nixl_buffer_device: "cuda" → CUDA 버퍼 사용
# nixl_enable_gc: True → 가비지 컬렉션 활성화

# [여기부터 핵심 요약]
# 1. 통신 최적화 (NIXL + UCX_TLS=cuda_copy,shm,tcp)
#    - AWS 환경에서는 보안상 GPU 간 직통 통신(cuda_ipc)이 차단됨.
#    - 따라서 NIXL(고속 통신망)을 켜고, cuda_copy(GPU DMA 하드웨어 복사기)를 사용하여
#    - 메인보드의 공유 메모리(shm)를 거쳐 통신하는 것이 AWS에서 구현 가능한 가장 빠른 방법임.
#    - remote_serde: NULL → 직렬화 사용은 라이브러리 안쓰고 직접 보내면 써야하는데, 사실 요즘 라이브러리들은 통신엔진이 알아서 하기 떄문에 직렬화를 파이썬에게 맡겨서 느려지게 할필요가 없음

# 2. 캐시 오프로딩 차단 (max_local_cpu/disk_size: 0)
#    - LMCache의 본업인 'CPU/디스크로의 프리픽스 캐시 저장'을 원천 차단함.어자피 프리픽스 캐싱 끌거라 같이 끈거
#    - 캐싱을 꺼서 디스크 I/O 병목을 막고, 오직 '네트워크 통신(파이프) 속도' 측정에만 집중하기 위함.
# 3. 청크드 프리필 끄기 (--no-enable-chunked-prefill)
#    - 한 번에 연산하여 GPU 연산량 한계(Compute Bound)를 뽑아내기 위해 껐음. (추후 켜고 비교 측정 필요)
# C1: TP=2 PP=1 (prefill GPU 0-1, decode GPU 2-3) — 기존 default
configC1_prefill() {
    echo "[launch] configC1 prefill: TP=2 PP=1 on GPU 0,1, port 8100"
    local cfg="$LOG_DIR/lmcache_prefill_C1.yaml"
    cat > "$cfg" <<YAML
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0
remote_serde: NULL
enable_nixl: True
nixl_role: "sender"
nixl_peer_host: "127.0.0.1"
nixl_peer_port: 55555
nixl_buffer_size: 1073741824
nixl_buffer_device: "cuda"
nixl_enable_gc: True
YAML
    CUDA_VISIBLE_DEVICES=0,1 \
    UCX_TLS=cuda_copy,shm,tcp \
    LMCACHE_CONFIG_FILE="$cfg" \
    vllm serve "${COMMON_FLAGS[@]}" \
        --no-enable-chunked-prefill \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 1 \
        --kv-transfer-config '{"kv_connector":"InstrumentedLMCacheConnector","kv_connector_module_path":"instrumented_connector","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"producerC"}}' \
        --port 8100 \
        2>&1 | tee "$LOG_DIR/vllm_configC1_prefill_$(hostname).log"
}

configC1_decode() {
    echo "[launch] configC1 decode: TP=2 PP=1 on GPU 2,3, port 8200"
    local cfg="$LOG_DIR/lmcache_decode_C1.yaml"
    cat > "$cfg" <<YAML
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0
remote_serde: NULL
enable_nixl: True
nixl_role: "receiver"
nixl_peer_host: "127.0.0.1"
nixl_peer_port: 55555
nixl_buffer_size: 1073741824
nixl_buffer_device: "cuda"
nixl_enable_gc: True
YAML
    CUDA_VISIBLE_DEVICES=2,3 \
    UCX_TLS=cuda_copy,shm,tcp \
    LMCACHE_CONFIG_FILE="$cfg" \
    vllm serve "${COMMON_FLAGS[@]}" \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 1 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"consumerC"}}' \
        --port 8200 \
        2>&1 | tee "$LOG_DIR/vllm_configC1_decode_$(hostname).log"
}

# C2: TP=1 PP=2 — 같은 4× T4 하드웨어, 같은 NIXL/shm 경로. TP/PP만 다름.
configC2_prefill() {
    echo "[launch] configC2 prefill: TP=1 PP=2 on GPU 0,1, port 8100"
    local cfg="$LOG_DIR/lmcache_prefill_C2.yaml"
    cat > "$cfg" <<YAML
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0
remote_serde: NULL
enable_nixl: True
nixl_role: "sender"
nixl_peer_host: "127.0.0.1"
nixl_peer_port: 55555
nixl_buffer_size: 1073741824
nixl_buffer_device: "cuda"
nixl_enable_gc: True
YAML
    CUDA_VISIBLE_DEVICES=0,1 \
    UCX_TLS=cuda_copy,shm,tcp \
    LMCACHE_CONFIG_FILE="$cfg" \
    vllm serve "${COMMON_FLAGS[@]}" \
        --no-enable-chunked-prefill \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 2 \
        --kv-transfer-config '{"kv_connector":"InstrumentedLMCacheConnector","kv_connector_module_path":"instrumented_connector","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"producerC"}}' \
        --port 8100 \
        2>&1 | tee "$LOG_DIR/vllm_configC2_prefill_$(hostname).log"
}

configC2_decode() {
    echo "[launch] configC2 decode: TP=1 PP=2 on GPU 2,3, port 8200"
    local cfg="$LOG_DIR/lmcache_decode_C2.yaml"
    cat > "$cfg" <<YAML
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0
remote_serde: NULL
enable_nixl: True
nixl_role: "receiver"
nixl_peer_host: "127.0.0.1"
nixl_peer_port: 55555
nixl_buffer_size: 1073741824
nixl_buffer_device: "cuda"
nixl_enable_gc: True
YAML
    CUDA_VISIBLE_DEVICES=2,3 \
    UCX_TLS=cuda_copy,shm,tcp \
    LMCACHE_CONFIG_FILE="$cfg" \
    vllm serve "${COMMON_FLAGS[@]}" \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 2 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"consumerC"}}' \
        --port 8200 \
        2>&1 | tee "$LOG_DIR/vllm_configC2_decode_$(hostname).log"
}

# Back-compat aliases: configC → configC1
configC_prefill() { configC1_prefill; }
configC_decode()  { configC1_decode; }

# ── Config D — cross-node PD, TCP ─────────────────────────────────────────────
# Run prefill side on the prefill node, decode side on the decode node.
# Set DECODER_HOST to the decode node's private IP before calling this script.
configD_prefill() {
    local decoder_host="${DECODER_HOST:-DECODER_HOST_NOT_SET}"
    if [[ "$decoder_host" == "DECODER_HOST_NOT_SET" ]]; then
        echo "ERROR: Set DECODER_HOST=<decode-node-private-ip> before running configD prefill"
        exit 1
    fi
    echo "[launch] configD prefill: TP=1, peer=$decoder_host:55555, port 8100"
    local cfg="$LOG_DIR/lmcache_prefill_D.yaml"
    cat > "$cfg" <<YAML
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0
remote_serde: NULL
enable_nixl: True
nixl_role: "sender"
nixl_peer_host: "$decoder_host"
nixl_peer_port: 55555
nixl_buffer_size: 1073741824
nixl_buffer_device: "cuda"
nixl_enable_gc: True
YAML
    CUDA_VISIBLE_DEVICES=0 \
    UCX_TLS=tcp \
    LMCACHE_CONFIG_FILE="$cfg" \
    vllm serve "${COMMON_FLAGS[@]}" \
        --no-enable-chunked-prefill \
        --tensor-parallel-size 1 \
        --kv-transfer-config '{"kv_connector":"InstrumentedLMCacheConnector","kv_connector_module_path":"instrumented_connector","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"producerD"}}' \
        --port 8100 \
        2>&1 | tee "$LOG_DIR/vllm_configD_prefill_$(hostname).log"
}

configD_decode() {
    echo "[launch] configD decode: TP=1, port 8200"
    local cfg="$LOG_DIR/lmcache_decode_D.yaml"
    # The decode node writes its own config; peer_host is not used by receiver.
    cat > "$cfg" <<YAML
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0
remote_serde: NULL
enable_nixl: True
nixl_role: "receiver"
nixl_peer_host: "127.0.0.1"
nixl_peer_port: 55555
nixl_buffer_size: 1073741824
nixl_buffer_device: "cuda"
nixl_enable_gc: True
YAML
    CUDA_VISIBLE_DEVICES=0 \
    UCX_TLS=tcp \
    LMCACHE_CONFIG_FILE="$cfg" \
    vllm serve "${COMMON_FLAGS[@]}" \
        --tensor-parallel-size 1 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"consumerD"}}' \
        --port 8200 \
        2>&1 | tee "$LOG_DIR/vllm_configD_decode_$(hostname).log"
}

# ── proxy launcher (shared by C and D) ───────────────────────────────────────
launch_proxy() {
    local prefill_host="${1:-127.0.0.1}"
    local decode_host="${2:-127.0.0.1}"
    echo "[launch] disagg proxy: prefill=$prefill_host:8100 decode=$decode_host:8200 proxy=8000"
    python "$PROXY_SCRIPT" \
        --host 0.0.0.0 \
        --port 8000 \
        --prefiller-host "$prefill_host" \
        --prefiller-port 8100 \
        --decoder-host "$decode_host" \
        --decoder-port 8200 \
        2>&1 | tee "$LOG_DIR/pd_proxy_$(hostname).log"
}

# ── dispatch ──────────────────────────────────────────────────────────────────
case "$CONFIG" in
    configA)  configA  ;;   # alias for configA1
    configA1) configA1 ;;
    configA2) configA2 ;;
    configA3) configA3 ;;
    configB) configB ;;
    configC)   # alias for configC1
        case "$ROLE" in
            prefill) configC1_prefill ;;
            decode)  configC1_decode ;;
            proxy)   launch_proxy "127.0.0.1" "127.0.0.1" ;;
            *) echo "configC needs role: prefill | decode | proxy"; exit 1 ;;
        esac ;;
    configC1)
        case "$ROLE" in
            prefill) configC1_prefill ;;
            decode)  configC1_decode ;;
            proxy)   launch_proxy "127.0.0.1" "127.0.0.1" ;;
            *) echo "configC1 needs role: prefill | decode | proxy"; exit 1 ;;
        esac ;;
    configC2)
        case "$ROLE" in
            prefill) configC2_prefill ;;
            decode)  configC2_decode ;;
            proxy)   launch_proxy "127.0.0.1" "127.0.0.1" ;;
            *) echo "configC2 needs role: prefill | decode | proxy"; exit 1 ;;
        esac ;;
    configD)
        case "$ROLE" in
            prefill) configD_prefill ;;
            decode)  configD_decode ;;
            proxy)
                PREFILL_HOST="${PREFILL_HOST:-127.0.0.1}"
                DECODE_HOST="${DECODE_HOST:-127.0.0.1}"
                launch_proxy "$PREFILL_HOST" "$DECODE_HOST"
                ;;
            *) echo "configD needs role: prefill | decode | proxy"; exit 1 ;;
        esac ;;
    *)
        echo "Unknown config: $CONFIG. Valid: configA configA1 configA2 configA3 configB configC configC1 configC2 configD"
        exit 1 ;;
esac
