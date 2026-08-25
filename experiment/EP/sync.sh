#!/usr/bin/env bash
# Whitelist sync: ONLY the paths in SYNC_PATHS are sent to each node
# (into ~/vllm-disaggregation/). Nothing else is touched or deleted.
#   ./sync.sh <node_ip> [<node_ip> ...]
set -euo pipefail

# 동기화할 폴더 목록 — 필요해지면 여기에 추가 (예: vllm csrc benchmarks)
SYNC_PATHS=(experiment)

[ $# -ge 1 ] || { echo "usage: sync.sh <node_ip> [<node_ip> ...]" >&2; exit 1; }

REMOTE_DIR=${REMOTE_DIR:-vllm-disaggregation}
SSH_USER=${SSH_USER:-ubuntu}
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

for NODE_IP in "$@"; do
  echo "=== $SSH_USER@$NODE_IP:$REMOTE_DIR/ <- ${SYNC_PATHS[*]} ==="
  rsync -avz \
    --exclude '__pycache__' \
    "${SYNC_PATHS[@]/#/$REPO_ROOT/}" \
    "$SSH_USER@$NODE_IP:$REMOTE_DIR/"
done
