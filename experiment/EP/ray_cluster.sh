#!/usr/bin/env bash
# Start Ray on a node. Env vars exported here are inherited by Ray workers,
# which is the only way remote vLLM engines receive them.
#   ./ray_cluster.sh head
#   ./ray_cluster.sh worker <head_ip>

# -e: 명령 실패 시 즉시 종료 / -u: 미정의 변수 사용 시 에러 /
# -o pipefail: 파이프 중간 명령이 실패해도 실패로 처리
set -euo pipefail

# $1이 없으면 usage를 찍고 종료 (head 또는 worker 역할 필수)
ROLE=${1:?usage: ray_cluster.sh head|worker [head_ip]}

# NCCL/gloo가 쓸 NIC. 인터페이스 이름이 인스턴스마다 다르므로(ens5,
# enX0, eth0 ...) 하드코딩하지 않고 기본 라우트가 타는 인터페이스를
# 노드별로 자동 감지. gloo는 prefix/제외 문법 없이 정확한 이름만
# 받아들이기 때문에 감지 방식이 필수
IFACE=${IFACE:-$(ip -o -4 route show to default | awk '{print $5}' | head -1)}

# Ray object store 크기(바이트). 기본값은 호스트 RAM의 30%(g6.xlarge에선
# ~4.8GB)인데, 16GiB RAM 노드에서 모델 로딩과 경합해 OOM을 유발하므로
# 2GB로 캡. vLLM DP는 object store를 거의 안 쓰므로 작아도 무방
OBJECT_STORE_BYTES=${OBJECT_STORE_BYTES:-2000000000}

# NCCL이 통신에 쓸 NIC을 고정. 기본 동작은 lo/docker만 제외하고 아무
# 인터페이스나 잡기 때문에, 가상 인터페이스가 있으면 노드 간 all-to-all이
# 소리 없이 hang됨. ray start 전에 export해야 Ray가 띄우는 원격 엔진
# 프로세스까지 상속됨 (serve 셸에서 export하면 head에만 적용되는 함정)
export NCCL_SOCKET_IFNAME=$IFACE

# torch.distributed의 제어 채널(gloo 백엔드)도 같은 NIC으로 고정.
# NCCL은 데이터 전송용, gloo는 초기 핸드셰이크/제어용으로 둘 다 쓰임
export GLOO_SOCKET_IFNAME=$IFACE

# TCP socket 전송을 이름 필터로 강제. NCCL 2.28.9(torch 2.11 번들)의
# per-comm net 플러그인 init 실패 경로에는 double free 버그가 있음:
# 실패한 플러그인이 comm->netContext를 free한 뒤 NULL로 리셋하지 않아,
# 다음 플러그인(IB)의 실패 경로가 같은 포인터를 다시 free함
# (ncclIbFinalize에서 "double free or corruption" SIGABRT).
# 이전에 쓰던 NCCL_IB_DISABLE=1은 IB init을 "실패"시키는 방식이라
# aws-ofi-nccl(DLAMI 기본 탑재)이 EFA 없는 g6.xlarge에서 먼저 실패하면
# 오히려 크래시를 유발. NCCL_NET=Socket은 IB/외부 플러그인의 init 자체를
# 건너뛰므로 버그 경로가 실행되지 않음
export NCCL_NET=Socket

# 외부 net 플러그인(libnccl-net.so, aws-ofi-nccl) 로드를 원천 차단.
# EFA가 없어 어차피 실패하며, 실패 과정에서 위 dangling ctx를 남기는
# 첫 번째 트리거이므로 로드하지 않는 것이 안전
export NCCL_NET_PLUGIN=none

# Ray의 메모리 감시자를 끔. 켜져 있으면 RAM 사용률이 임계치(95%)를
# 넘는 순간 worker 프로세스를 죽이는데, 16GiB 노드에선 모델 로딩
# 스파이크만으로 vLLM 엔진이 살해당할 수 있음
export RAY_memory_monitor_refresh_ms=0

case "$ROLE" in
head)
  # --head: 이 노드를 클러스터의 GCS(제어부)로 만듦
  # --port=6379: GCS 포트를 고정 — worker의 --address와 SG 규칙이
  #   이 포트를 참조하므로 랜덤이면 안 됨
  # --dashboard-host=0.0.0.0: 대시보드(8265)를 외부에서 접속 가능하게 바인딩
  #   (기본값 127.0.0.1이면 로컬에서만 보임)
  ray start --head --port=6379 --dashboard-host=0.0.0.0 \
    --object-store-memory="$OBJECT_STORE_BYTES"
  ;;
worker)
  # worker는 head의 IP가 추가로 필요
  HEAD_IP=${2:?usage: ray_cluster.sh worker <head_ip>}
  # --address: head의 GCS(6379)에 접속해 클러스터에 합류.
  #   이 노드의 GPU가 Ray 자원 풀에 등록되어 vLLM 엔진 actor 배치 대상이 됨
  ray start --address="$HEAD_IP:6379" \
    --object-store-memory="$OBJECT_STORE_BYTES"
  ;;
*)
  echo "unknown role: $ROLE" >&2
  exit 1
  ;;
esac

# 클러스터 상태 출력 — 노드 수와 GPU 자원이 기대대로 잡혔는지 즉시 확인
ray status
