#!/bin/bash
# Launch vllm serve for each config (A/B/C/D) and role.
# *** Branch: experiment/tier1-vllm-benchmark — uses the official P2pNcclConnector ***
# *** instead of the custom InstrumentedLMCacheConnector. Goal: minimize bespoke   ***
# *** code surface area by following vLLM's upstream disaggregated_prefill example. ***
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
#   bash launch_configs.sh configC2 ...       # NOT SUPPORTED with P2pNcclConnector (PP not implemented)
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
#   ENFORCE_EAGER         default: 0     (1 → add --enforce-eager; turn on if CUDA Graph + P2pNccl misbehaves)
#   EXP_LOG_DIR           default: ./results
#   VLLM_HOST_IP          default: 127.0.0.1 (set to the node's private IP for cross-node)
#   PROXY_PORT            default: 30001 (ZMQ control channel for P2pNccl)
#   PYTHONHASHSEED        default: 123  (must match on prefill+decode)
#
# AWS / NCCL notes:
#   - g4dn / g6 / g6e have no NVLink and no InfiniBand. NCCL falls back to PCIe/SHM
#     (same-node) or TCP sockets (cross-node) automatically.
#   - To avoid NCCL probing IB for several seconds before giving up, set:
#       export NCCL_IB_DISABLE=1
#       export NCCL_SOCKET_IFNAME=<your-nic>   # e.g. ens5 on most Nitro instances
#   - First-time verification: export NCCL_DEBUG=INFO once and check the chosen transport.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROXY_SCRIPT="$REPO_ROOT/benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
LOG_DIR="${EXP_LOG_DIR:-./results}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
# (Fix: Config D cross-node proxy 통신을 위해 PROXY_IP 추가 연동)
export PROXY_IP="${PROXY_IP:-$VLLM_HOST_IP}"
PROXY_PORT="${PROXY_PORT:-30001}"

# Default NCCL settings — safe defaults for AWS. Override before calling if needed.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp39s0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

mkdir -p "$LOG_DIR"

CONFIG="${1:-}"
ROLE="${2:-}"

if [[ -z "$CONFIG" ]]; then
    echo "Usage: $0 <configA|configA1|configA2|configA3|configB|configC1|configD> [prefill|decode|proxy]"
    exit 1
fi

# ── common flags ──────────────────────────────────────────────────────────────
# Per the experiment design's noise-control list.
COMMON_FLAGS=(
    --model "$MODEL"
    --served-model-name llama-3.1-8b
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --no-enable-prefix-caching
    # (Fix: T4 GPU 호환성을 위해 bfloat16 대신 half 강제 적용)
    --dtype half
    --max-num-seqs "${MAX_NUM_SEQS:-512}"
)

# Optional: enforce-eager fallback. CUDA Graph capture can clash with the
# P2pNcclConnector send/recv path; the xpyd upstream example uses --enforce-eager
# as a safety net. We follow the experiment spec (CUDA Graph ON by default) and
# expose an env knob so the operator can flip it without editing the script.
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
    COMMON_FLAGS+=( --enforce-eager )
fi

# ── Config A — monolithic 4×T4. Three TP/PP variants on the same hardware ────
# Monolithic: no KV connector. P2pNccl is irrelevant here.
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

# ── Config C1 — same-node PD on 4× T4. P2pNcclConnector via PCIe/SHM ─────────
#
# Why P2pNcclConnector (vs the LMCache+NIXL combo on the other branch):
#   - Upstream vLLM example. Zero custom Python in this path.
#   - Pure NCCL send/recv. NCCL auto-picks the transport (PCIe P2P / SHM for
#     same-node; TCP sockets for cross-node). Works on AWS without IB/NVLink.
#
# kv_buffer_size:
#   - Producer "1e1" (placeholder; the prefill side does not buffer).
#   - Consumer "2e9" (~2GB host pinned). Smaller than xpyd's 8e9 because T4 is
#     tight; can be raised if KV-transfer back-pressure is observed.
# mem_pool_size_gb: pinned host RAM allocated by TensorMemoryPool. Default is 32
#   GB which is fine on g4dn.12xlarge (192GB RAM) but blows up on g6.xlarge
#   (16GB RAM). We pin it explicitly per config.
#
# Ports plan (mind that each TP rank consumes one kv_port → kv_port + rank):
#   Prefill HTTP   : 8100              Decode HTTP   : 8200
#   Prefill kv_port: 14600, 14601      Decode kv_port: 14700, 14701
#   Proxy frontend : 8000              Proxy ZMQ     : 30001

configC1_prefill() {
    echo "[launch] configC1 prefill: TP=2 PP=1 on GPU 0,1, port 8100"
    CUDA_VISIBLE_DEVICES=0,1 \
    vllm serve "${COMMON_FLAGS[@]}" \
        --no-enable-chunked-prefill \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 1 \
        --port 8100 \
        --kv-transfer-config "$(cat <<JSON
{"kv_connector":"P2pNcclConnector",
 "kv_role":"kv_producer",
 "kv_rank":0,
 "kv_parallel_size":2,
 "kv_buffer_size":"1e1",
 "kv_port":"14600",
 "kv_connector_extra_config":{
   "proxy_ip":"$VLLM_HOST_IP",
   "proxy_port":"$PROXY_PORT",
   "http_ip":"$VLLM_HOST_IP",
   "http_port":"8100",
   "send_type":"PUT_ASYNC",
   "nccl_num_channels":"8",
   "mem_pool_size_gb":"8"
 }}
JSON
)" 2>&1 | tee "$LOG_DIR/vllm_configC1_prefill_$(hostname).log"
}

configC1_decode() {
    echo "[launch] configC1 decode: TP=2 PP=1 on GPU 2,3, port 8200"
    CUDA_VISIBLE_DEVICES=2,3 \
    vllm serve "${COMMON_FLAGS[@]}" \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 1 \
        --port 8200 \
        --kv-transfer-config "$(cat <<JSON
{"kv_connector":"P2pNcclConnector",
 "kv_role":"kv_consumer",
 "kv_rank":1,
 "kv_parallel_size":2,
 "kv_buffer_size":"2e9",
 "kv_port":"14700",
 "kv_connector_extra_config":{
   "proxy_ip":"$VLLM_HOST_IP",
   "proxy_port":"$PROXY_PORT",
   "http_ip":"$VLLM_HOST_IP",
   "http_port":"8200",
   "send_type":"PUT_ASYNC",
   "nccl_num_channels":"8",
   "mem_pool_size_gb":"8"
 }}
JSON
)" 2>&1 | tee "$LOG_DIR/vllm_configC1_decode_$(hostname).log"
}

# ── Config C2 — NOT SUPPORTED on this branch ──────────────────────────────────
# P2pNcclConnector explicitly rejects pipeline-parallel (see
# vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:528-530:
#   "Currently, only symmetric TP is supported. Asymmetric TP, PP, and
#    others will be supported in future PRs.").
# Keep the function stub for parity with the other branch but exit loudly.
configC2_prefill() {
    echo "[launch] configC2 is NOT supported with P2pNcclConnector (PP not implemented)."
    echo "         Use experiment/tier1 branch (LMCache+NIXL) for the TP=1 PP=2 variant."
    exit 2
}
configC2_decode() { configC2_prefill; }

configC_prefill() { configC1_prefill; }
configC_decode()  { configC1_decode; }

# ── Config D — cross-node PD, TCP sockets ────────────────────────────────────
# Run prefill on one g6.xlarge, decode on another. They reach each other via the
# VPC network. NCCL falls back to TCP since neither IB nor NVLink exists.
#
# Pre-reqs on each node:
#   export VLLM_HOST_IP=<this-node-private-ip>
# And on the prefill node:
#   export DECODER_HOST=<decode-node-private-ip>
#
# mem_pool_size_gb=4 — g6.xlarge has only 16GB host RAM; default 32GB OOMs.
configD_prefill() {
    local decoder_host="${DECODER_HOST:-DECODER_HOST_NOT_SET}"
    if [[ "$decoder_host" == "DECODER_HOST_NOT_SET" ]]; then
        echo "ERROR: Set DECODER_HOST=<decode-node-private-ip> before running configD prefill"
        exit 1
    fi
    echo "[launch] configD prefill: TP=1, my_ip=$VLLM_HOST_IP, decoder=$decoder_host"
    CUDA_VISIBLE_DEVICES=0 \
    vllm serve "${COMMON_FLAGS[@]}" \
        --no-enable-chunked-prefill \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 1 \
        --host 0.0.0.0 \
        --port 8100 \
        --kv-transfer-config "$(cat <<JSON
{"kv_connector":"P2pNcclConnector",
 "kv_role":"kv_producer",
 "kv_rank":0,
 "kv_parallel_size":1,
 "kv_buffer_size":"1e1",
 "kv_port":"14600",
 "kv_connector_extra_config":{
   "proxy_ip":"$PROXY_IP",
   "proxy_port":"$PROXY_PORT",
   "http_ip":"$VLLM_HOST_IP",
   "http_port":"8100",
   "send_type":"PUT_ASYNC",
   "nccl_num_channels":"8",
   "mem_pool_size_gb":"4"
 }}
JSON
)" 2>&1 | tee "$LOG_DIR/vllm_configD_prefill_$(hostname).log"
}

# 여기 Prefill ip넣어야함
configD_decode() {
    echo "[launch] configD decode: TP=1, my_ip=$VLLM_HOST_IP"
    CUDA_VISIBLE_DEVICES=0 \
    vllm serve "${COMMON_FLAGS[@]}" \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 1 \
        --host 0.0.0.0 \
        --port 8200 \
        --kv-transfer-config "$(cat <<JSON
{"kv_connector":"P2pNcclConnector",
 "kv_role":"kv_consumer",
 "kv_rank":1,
 "kv_parallel_size":1,
 "kv_buffer_size":"2e9",
 "kv_port":"14700",
 "kv_connector_extra_config":{
   "proxy_ip":"$PROXY_IP",
   "proxy_port":"$PROXY_PORT",
   "http_ip":"$VLLM_HOST_IP",
   "http_port":"8200",
   "send_type":"PUT_ASYNC",
   "nccl_num_channels":"8",
   "mem_pool_size_gb":"4"
 }}
JSON
)" 2>&1 | tee "$LOG_DIR/vllm_configD_decode_$(hostname).log"
}

# ── proxy launcher (shared by C1 and D) ──────────────────────────────────────
# This is vLLM's *official* disagg_prefill_proxy_server.py — quart-based.
# It listens on $PROXY_FRONTEND_PORT (default 8000), forwards prefill→8100 then
# decode→8200, and embeds the KV socket addresses in the X-KV-Target header so
# the workers can NCCL-send directly.
launch_proxy() {
    local prefill_host="${1:-127.0.0.1}"
    local decode_host="${2:-127.0.0.1}"
    local frontend_port="${PROXY_FRONTEND_PORT:-8000}"
    local prefill_kv_port="${PREFILL_KV_PORT:-14600}"
    local decode_kv_port="${DECODE_KV_PORT:-14700}"
    echo "[launch] official disagg proxy: prefill=$prefill_host:8100 decode=$decode_host:8200 frontend=$frontend_port"
    # quart binds to localhost by default in the upstream script. We invoke it
    # via the CLI args it exposes; nothing else is patched.
    python "$PROXY_SCRIPT" \
        --port "$frontend_port" \
        --prefill-url "http://$prefill_host:8100" \
        --decode-url  "http://$decode_host:8200" \
        --prefill-kv-host "$prefill_host" \
        --decode-kv-host  "$decode_host" \
        --prefill-kv-port "$prefill_kv_port" \
        --decode-kv-port  "$decode_kv_port" \
        2>&1 | tee "$LOG_DIR/pd_proxy_$(hostname).log"
}

# ── dispatch ──────────────────────────────────────────────────────────────────
case "$CONFIG" in
    configA)  configA  ;;
    configA1) configA1 ;;
    configA2) configA2 ;;
    configA3) configA3 ;;
    configB)  configB  ;;
    configC|configC1)
        case "$ROLE" in
            prefill) configC1_prefill ;;
            decode)  configC1_decode  ;;
            proxy)   launch_proxy "$VLLM_HOST_IP" "$VLLM_HOST_IP" ;;
            *) echo "configC1 needs role: prefill | decode | proxy"; exit 1 ;;
        esac ;;
    configC2)
        case "$ROLE" in
            prefill|decode|proxy) configC2_prefill ;;
            *) echo "configC2 needs role: prefill | decode | proxy"; exit 1 ;;
        esac ;;
    configD)
        case "$ROLE" in
            prefill) configD_prefill ;;
            decode)  configD_decode  ;;
            proxy)
                # On the prefill node, point at itself for prefill and at
                # DECODER_HOST for decode. PREFILL_HOST/DECODE_HOST overrides
                # let you run the proxy from a third box if desired.
                local_prefill="${PREFILL_HOST:-$VLLM_HOST_IP}"
                local_decode="${DECODE_HOST:-${DECODER_HOST:-127.0.0.1}}"
                launch_proxy "$local_prefill" "$local_decode"
                ;;
            *) echo "configD needs role: prefill | decode | proxy"; exit 1 ;;
        esac ;;
    *)
        echo "Unknown config: $CONFIG. Valid: configA configA1 configA2 configA3 configB configC configC1 configD"
        echo "Note: configC2 (TP=1 PP=2 PD) is NOT supported on this branch."
        exit 1 ;;
esac
