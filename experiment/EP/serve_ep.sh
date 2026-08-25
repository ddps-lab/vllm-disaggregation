#!/usr/bin/env bash
# Launch EP online serving on the head node. Ray places one engine per node.
#   ./serve_ep.sh
#   DP=4 MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 ./serve_ep.sh
set -euo pipefail

MODEL=${MODEL:-deepseek-ai/DeepSeek-V2-Lite-Chat}
DP=${DP:-2}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-8192}

vllm serve "$MODEL" \
  --data-parallel-size "$DP" \
  --data-parallel-size-local 1 \
  --data-parallel-backend ray \
  --enable-expert-parallel \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT"
