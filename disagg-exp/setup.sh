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

# ── 6. chrony ─────────────────────────────────────────────────────────────────
# chrony: 시간 동기화 도구인데 다른 인스턴스간 시간 오차가 발생할 수 있기 때문에 로그에서 이를 보정 하기 위해 사용
if command -v chronyc &>/dev/null; then
    chronyc tracking > "$LOG_DIR/clock_baseline_$(hostname).txt" 2>&1 || true
    echo "[setup] chrony baseline saved → $LOG_DIR/clock_baseline_$(hostname).txt"
else
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

# ── 8. background metric collectors ─────────────────────────────────────────
# 실험이 진행되는 동안, 1초 단위로 컴퓨터의 GPU, 네트워크를 계속해서 기록하는 로직
PIDFILE_DMON="$LOG_DIR/.pid_nvidia_dmon"
PIDFILE_IFSTAT="$LOG_DIR/.pid_ifstat"
PIDFILE_DCGM="$LOG_DIR/.pid_dcgm_loop"

# 여러 sweep 동안 백그라운드 프로세스가 실행 중일 수 있으므로, 일단 초기화
_kill_pid_file() {
    local pf="$1"
    if [[ -f "$pf" ]]; then
        local pid; pid=$(cat "$pf")
        kill "$pid" 2>/dev/null || true
        rm -f "$pf"
    fi
}

_kill_pid_file "$PIDFILE_DMON"
_kill_pid_file "$PIDFILE_IFSTAT"
_kill_pid_file "$PIDFILE_DCGM"
# belt-and-suspenders: also kill by command pattern
pkill -f "nvidia-smi dmon" 2>/dev/null || true
pkill -f "ifstat -t" 2>/dev/null || true

# nvidia-smi dmon: 1Hz, Power | Utilization | SM Clk | Memory
# DCGM 보단 라이트 하고 혹시 몰라서 DCGM과 함꼐 이중으로 수집하는것 
nohup nvidia-smi dmon -s pucvmet -d 1 -o DT \
    > "$LOG_DIR/nvidia_smi.csv" 2>&1 &
echo $! > "$PIDFILE_DMON"
echo "[setup] nvidia-smi dmon pid=$(cat "$PIDFILE_DMON")"

# ifstat: NIC throughput 1Hz with timestamps
#  Prefill ➔ Decode 노드로 KV 캐시를 초당 몇 MB/s로 밀어넣고 있는지 실시간으로 기록
if command -v ifstat &>/dev/null; then
    IFACE=$(ip route get 1 2>/dev/null | awk '/dev/{print $5;exit}')
    nohup ifstat -t -i "${IFACE:-eth0}" 1 \
        > "$LOG_DIR/ifstat.csv" 2>&1 &
    echo $! > "$PIDFILE_IFSTAT"
    echo "[setup] ifstat pid=$(cat "$PIDFILE_IFSTAT")"
else
    echo "[setup] WARN: ifstat not found — NIC metrics absent."
fi

# DCGM scrape loop — use PID file so we can reliably kill it on re-run
# 자세한 GPU 메트릭을 수집 
(while true; do
    curl -sf "http://localhost:9400/metrics" \
        | grep -E "DCGM_FI_DEV_(FB_USED|GPU_UTIL|SM_OCCUPANCY|POWER_USAGE|DRAM_ACTIVE|MEM_COPY_UTILIZATION)" \
        >> "$LOG_DIR/dcgm.log" 2>/dev/null
    echo "---" >> "$LOG_DIR/dcgm.log"
    sleep 2
done) &
echo $! > "$PIDFILE_DCGM"
echo "[setup] dcgm scrape loop pid=$(cat "$PIDFILE_DCGM")"

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
echo "  bash disagg-exp/launch_configs.sh configA"
