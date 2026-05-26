#!/bin/bash
# Bootstrap script for disagg-exp nodes. Idempotent — safe to re-run.
# Run as: bash setup.sh
# Assumes: Ubuntu 22.04 DLAMI, Python 3.12 available, nvidia-smi present.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv"
LOG_DIR="${EXP_LOG_DIR:-./results}"
mkdir -p "$LOG_DIR"

# ── 1. uv ────────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.cargo/bin:$PATH"
fi

# ── 2. venv ───────────────────────────────────────────────────────────────────
if [[ ! -f "$VENV/bin/python" ]]; then
    uv venv "$VENV" --python 3.12
fi
source "$VENV/bin/activate"

# ── 3. vLLM fork (editable) ──────────────────────────────────────────────────
if ! python -c "import vllm" &>/dev/null; then
    # --torch-backend=auto requires uv ≥ 0.5; fall back to plain install if unsupported.
    if uv pip install --help 2>&1 | grep -q "torch-backend"; then
        VLLM_USE_PRECOMPILED=1 uv pip install -e "$REPO_ROOT" --torch-backend=auto
    else
        echo "[setup] uv does not support --torch-backend; falling back to plain install"
        VLLM_USE_PRECOMPILED=1 uv pip install -e "$REPO_ROOT"
    fi
fi

# ── 4. LMCache ───────────────────────────────────────────────────────────────
# Disaggregation에서 KV캐시를 전송하는 라이브러리 중 LM캐시가 정배 
LMCACHE_VER=$(python -c "import lmcache; print(lmcache.__version__)" 2>/dev/null || echo "")
NEED_LMCACHE=0
if [[ -z "$LMCACHE_VER" ]]; then
    NEED_LMCACHE=1
else
    # require >= 0.3.9
    python -c "
from packaging.version import Version
import sys
v = '$LMCACHE_VER'
if Version(v) < Version('0.3.9'):
    sys.exit(1)
" 2>/dev/null || NEED_LMCACHE=1
fi
if [[ $NEED_LMCACHE -eq 1 ]]; then
    uv pip install "lmcache>=0.3.9"
fi

# ── 5. misc python deps ───────────────────────────────────────────────────────
# 뭐가 필수고 아닌지 확인

uv pip install httpx fastapi uvicorn numpy packaging aiohttp 2>/dev/null || true
# quart: required by benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py (P2pNccl proxy)
uv pip install quart msgpack 2>/dev/null || true

# ── 5b. system tools ─────────────────────────────────────────────────────────
# 네트워크 대역폭 측정기 (ifstat) 현재 1초당 네트워크 데이터가 몇 메가바이트나 흐르고 있는지를 기록해 주는 아주 가벼운 툴
if ! command -v ifstat &>/dev/null; then
    echo "[setup] installing ifstat ..."
    sudo apt-get install -y -q ifstat 2>/dev/null || echo "[setup] WARN: apt install ifstat failed (non-root?)"
fi

# s5cmd: fast S3 sync for log upload during experiment
# 초고속 S3 백업 툴, aws s3 sync 명령어도 있지만 속도가 느려서 사용

if ! command -v s5cmd &>/dev/null; then
    echo "[setup] installing s5cmd ..."
    S5CMD_VER="2.2.2"
    wget -q "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VER}/s5cmd_${S5CMD_VER}_Linux-64bit.tar.gz" \
        -O /tmp/s5cmd.tar.gz \
    && tar xzf /tmp/s5cmd.tar.gz -C /tmp \
    && sudo mv /tmp/s5cmd /usr/local/bin/ \
    && rm /tmp/s5cmd.tar.gz \
    && echo "[setup] s5cmd $(s5cmd version) installed" \
    || echo "[setup] WARN: s5cmd install failed — S3 sync will not work"
fi

# nvtop: GPU 메모리 실시간 모니터링 (htop for GPU)
# KV 캐시 전송 중 GPU 메모리 사용량을 실시간으로 보기 위함
if ! command -v nvtop &>/dev/null; then
    echo "[setup] installing nvtop ..."
    sudo apt-get install -y -q nvtop 2>/dev/null || echo "[setup] WARN: apt install nvtop failed (non-root?)"
fi

# huggingface_hub CLI (for model download)
# 허깅페이스 모델 다운로드 CLI 나중에 sweep시 허깅페이스 모델 다운로드할 때 사용
python -c "import huggingface_hub" &>/dev/null \
    || uv pip install huggingface_hub 2>/dev/null || true

# ── 6. chrony 검증만 ─────────────────────────────────────────────────────────
# chrony baseline 파일은 launch_configs.sh 가 실험 단위로 RUN_DIR/system_logs/ 안에
# 생성. 여기서는 chronyc 존재 여부만 확인.
if ! command -v chronyc &>/dev/null; then
    echo "[setup] WARN: chronyc not found. Install with: sudo apt install chrony"
fi

# ── 7. DCGM exporter (best-effort) ──────────────────────────────────────────
# DCGM exporter: GPU 메트릭 수집 exporter(GPU 안의 수천 개 연산 코어(SM)가 진짜로 일을 하고 있는지, 아니면 메모리(DRAM) 대역폭이 꽉 차서 놀고 있는지)
# Assumes dcgm-exporter already installed on DLAMI.
# Check if already running; if not, try to start.
if ! curl -sf "http://localhost:9400/metrics" | grep -q DCGM_FI 2>/dev/null; then
    if command -v dcgm-exporter &>/dev/null; then
        nohup dcgm-exporter -f /etc/dcgm-exporter/default-counters.csv \
            -a ":9400" >> "$LOG_DIR/dcgm_exporter.log" 2>&1 &
        echo "[setup] started dcgm-exporter (pid $!)"
    else
        echo "[setup] WARN: dcgm-exporter not found. DCGM metrics will be absent."
    fi
fi

# ── 8. background metric collectors — REMOVED ───────────────────────────────
# 시스템 metric collector (nvidia-smi dmon, ifstat, DCGM scrape loop) 는 더이상
# setup.sh 가 직접 실행하지 않음. 이유: 실험 단위로 분리된 폴더
# ($EXP_LOG_DIR/{CONFIG}-{MODEL}/system_logs/) 에 출력해야 깔끔하게 S3 sync 되는데,
# setup.sh 시점엔 어떤 config 가 실행될지 모르기 때문.
# → launch_configs.sh 가 vllm serve 시작과 동시에 자기 RUN_DIR 에 맞춰 collector 시작/종료.

# ── 9. validation ─────────────────────────────────────────────────────────────
python -c "
import vllm, lmcache
from packaging.version import Version
print(f'vllm={vllm.__version__}  lmcache={lmcache.__version__}')
assert Version(lmcache.__version__) >= Version('0.3.9'), 'lmcache too old'
print('OK: version check passed')
"

echo ""
echo "=== setup.sh done ==="
echo "  VENV:    $VENV"
echo "  LOG_DIR: $LOG_DIR"
echo ""
echo "Next steps:"
echo "  source $VENV/bin/activate"
echo "  # 실험 식별자 (양쪽 노드에서 같은 값 권장)"
echo "  export RUN_TAG=\$(date +%Y%m%d-%H%M)-baseline"
echo "  bash disagg-exp/launch_configs.sh configA          # monolithic"
echo "  bash disagg-exp/launch_configs.sh configD prefill  # cross-node PD"
