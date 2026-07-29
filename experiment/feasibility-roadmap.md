# 구현 가능성 조사: DP 배치 제어 · 노드별 DP degree · Elastic EP · K8s (2026-07-28)

> 대상: vLLM 0.26.0 소스트리 + upstream 동향. Fleet: 2-4x g6.xlarge (노드당 L4 1개,
> TCP 10Gbps, RDMA 없음), Ray 백엔드 기본.
> 판정 등급: works-today(지금 됨) / config-only(설정만) / custom-code-small(소규모
> 코드) / custom-code-large(대규모) / not-feasible(불가)

## 판정 요약

| # | 목표 | 판정 | 예상 규모 |
|---|---|---|---|
| 1 | 노드 IP 집합 제한 배치 | **config-only** | env var 1개 |
| 1' | rank i → IP x 정확 고정 | **custom-code-small** | ~10줄 패치 or monkeypatch |
| 2 | 노드별 상이한 DP 개수 (mp) | **works-today** | 0 (노드별 플래그) |
| 2' | 노드별 상이한 DP 개수 (ray, 명시 지정) | **custom-code-small** | 50-100줄 |
| 3 | Elastic EP 2→4→2 데모 | **config-only** | 1-2일 |
| 4 | K8s (k3s+LWS) 적용 | **config-only** | 2-3일 |
| 4' | MPI 런처 | **not-feasible** | 오해 — vLLM에 MPI 없음 |

---

## 1. IP로 노드를 지정해 DP 배치

**이미 절반은 있다.** `VLLM_RAY_DP_PLACEMENT_NODE_IPS` (envs.py:173)가 배치
가능한 노드를 IP 목록으로 **제한**한다. 단, 값이 set으로 파싱되어 **순서가
버려지므로** "rank→IP 매핑"은 아니다 (utils.py:551-555). 보장되는 것:
master IP가 rank 0 (master-first 정렬, utils.py:541-548), 나머지 rank는
Ray의 노드 등록 순서(비결정적).

- 대칭 DP+EP 워크로드면 "노드 집합 제한"만으로 충분 → **env var로 끝**
- **정확한 rank i → IP x가 필요하면** 두 가지 소규모 경로:
  1. `create_dp_placement_groups`의 노드 순회(utils.py:640) 전에 env 목록
     순서로 재정렬 — **~10줄 패치** (rank index == 노드 순회 순서이므로)
  2. 무패치: `CoreEngineActorManager.create_dp_placement_groups`는
     `@staticmethod`라 monkeypatch 가능. **vLLM 자체 테스트가 이 패턴을 사용**
     (tests/v1/engine/test_core_engine_actor_manager.py:170-181) — 계약:
     반환 리스트의 인덱스 = DP rank. PG 형식:
     `bundles=[{"GPU":1.0, "node:<ip>":0.001}]*world_size + control(CPU)`,
     `strategy="STRICT_PACK"`. `node:<ip>`는 Ray 내장 노드 리소스라 이게 곧 고정 장치다.
- `CoreEngineActorManager.__init__(placement_groups=, local_dp_ranks=)` 주입
  시맨틱도 존재(utils.py:379-387, 443-460)하나 production 호출자
  (`launch_core_engines`)가 kwargs를 안 넘김 — CLI 배관이 없을 뿐 구조는 열려 있음.
- **mp+headless는 원래 결정적**: 노드에서 직접 `--data-parallel-start-rank R`로
  실행하므로 rank→IP가 구성상 확정. 코드 0줄, 대신 노드별 명령 오케스트레이션 필요.

## 2. 노드별로 다른 DP degree

**"uniform만 된다"는 부분적으로만 사실.**

| 방식 | 노드별 개수 | 비고 |
|---|---|---|
| ray `strict` (기본) | **균등 강제** — 모든 노드가 정확히 dp_size_local개, 못 맞추면 skip | utils.py:663-673 |
| ray `fill` | **비균등 자동** — master는 dp_size_local, 나머지는 `n_devices//world_size`씩 용량대로 | 사용자 지정은 불가. DeepEP와 비호환(무관) |
| mp+headless | **완전 자유** — 노드마다 `--data-parallel-size-local X` 독립 지정 | works-today. [0,dp_size) 타일링은 사용자 책임 |
| ray + 커스텀 PG 주입 | **완전 자유** | 1'과 동일 monkeypatch, 50-100줄 |

핵심 발견: **DP 내부(DPCoordinator, stateless gloo 그룹, EP 그룹)는 전부 flat
world라 노드별 개수 가정이 전혀 없다** (coordinator.py:90의 local 비교는 ZMQ
IPC/TCP 선택용일 뿐). 즉 배치만 해결하면 내부는 그대로 동작한다.
단, 현 fleet(노드당 GPU 1)에서는 노드별 개수가 0 또는 1뿐이라 이 질문 자체가
당장은 무의미 — multi-GPU 노드를 섞을 때 유효해진다.

## 3. Elastic EP (EEP)

**가장 좋은 소식: 0.26에 완전히 구현돼 있고, CI 테스트 스택이 정확히 우리
구성이다** — DeepSeek-V2-Lite-Chat + TP=1 + ray DP + allgather_reducescatter
(tests/distributed/test_elastic_ep.py, examples/ray_serving/elastic_ep/).

동작 방식: `/scale_elastic_ep` → 신규 요청 503 차단 → (기본은 drain 없이
TCPStore 2단 배리어로 동기화) → standby stateless NCCL 그룹 생성 → 신규 Ray
actor는 dummy-load 후 **비-expert weight를 NCCL로 전송받음** (HF 다운로드
불필요!) → 그룹 스왑 → EPLB가 expert 재배치.

제약 (전부 config 수준):
- `--enable-eplb` 필수, PP=1, TP=1(테스트 범위), ray 백엔드 전용,
  api_server_count=1, external/hybrid LB 불가
- **boot DP가 하한** — DP=2로 시작해야 2→4→2 가능 (expert 슬롯이 boot 시 고정)
- RDMA/DeepEP 불필요, sm_89 블로커 없음. NIXL은 async EPLB에만 필요
- 재구성 동안 **서빙 전면 중단**: upstream 실측 H100/NVLink에서 35-50초 —
  TCP 10Gbps에서는 수 분 예상. multi-node는 upstream 미벤치마크 영역
  (= 연구 기여 지점)

데모 레시피 (1-2일):
```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code \
  --data-parallel-size 2 --data-parallel-size-local 1 \
  --data-parallel-backend ray --enable-expert-parallel \
  --enable-eplb --enable-elastic-ep \
  --all2all-backend allgather_reducescatter \
  --eplb-config.use_async false --enforce-eager
# 노드 3,4를 ray join 시킨 뒤:
curl -X POST :8000/scale_elastic_ep -d '{"new_data_parallel_size": 4}'
```
scale-up 시 placement group을 **그 시점의 live 노드 목록**으로 계산하므로
서빙 시작 후 합류한 노드로의 확장이 동작한다.

Upstream 상태: RFC #20323(2025-07) → PR #34861 구현 → 공식 블로그(2026-05-14).
명시적 early-stage — async 재구성(PR #47288, 다운타임 50s→7s), mp 백엔드,
K8s 연동은 진행 중.

## 4. K8s 적용 (학습 겸)

**config-only, 2-3일.** 추천 경로: **k3s + 드라이버 프리인스톨 AMI + NVIDIA
device plugin(standalone, gpu-operator 불필요) + LeaderWorkerSet v0.9.0**.

- Day 1: k3s 설치 (`--default-runtime nvidia`), agent join, device plugin,
  `nvidia.com/gpu` 리소스 확인
- Day 2: LWS 설치(매니페스트 1개), DP/EP LWS YAML 작성 —
  leader: 현재 serve_ep.sh 커맨드 그대로 / worker:
  `--headless --data-parallel-start-rank $(LWS_WORKER_INDEX)` (LWS 자동 주입 변수)
- 스팟 대응: **DP/EP는 rank 하나만 죽어도 전체 사망**(EngineDeadError) —
  orchestrator와 무관한 본질. LWS `restartPolicy: RecreateGroupOnPodRestart`가
  그룹 전체 재생성으로 대응하는 자연스러운 짝. HF 캐시는 hostPath로.
- KubeRay 대안: 동작하지만 K8s 학습량이 적고 부품이 더 많음 → phase 2로.
- **MPI는 오해로 확정**: `distributed_executor_backend`는
  `Literal["ray","mp","uni","external_launcher"]` (parallel.py:35), 트리 전체에
  mpirun/mpi4py 0건. external_launcher(torchrun)는 DP에서 **오프라인 전용**
  (API 서버 없음). 비-Ray 온라인 경로는 mp+headless이고, LWS가 그걸 자동화한다.

### 4-1. LWS-mp 상세 — LeaderWorkerSet이 정확히 무엇을 하는가

**LWS는 "pod 그룹을 하나의 원자적 단위로 다루는" K8s API**다
(kubernetes-sigs/lws). 일반 Deployment는 pod들이 서로 무관하다고 가정하지만,
멀티노드 추론은 "leader 1 + worker N-1이 한 몸"이라는 전제가 필요해서 만들어졌다.

핵심 API 필드:

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
spec:
  replicas: 1                # 그룹(=모델 replica) 수. 2로 올리면 DP그룹 2벌
  leaderWorkerTemplate:
    size: 2                  # 그룹당 pod 수 (leader 1 + worker size-1)
    restartPolicy: RecreateGroupOnPodRestart   # pod 하나 죽으면 그룹 전체 재생성
    leaderTemplate: {...}    # leader pod 템플릿 (역할별 템플릿이 2개인 게 핵심)
    workerTemplate: {...}    # worker pod 템플릿
```

LWS 컨트롤러가 각 pod에 **자동 주입하는 env var** 3개가 vLLM 플래그와 연결된다:

| env var | 값 | vLLM에서의 쓰임 |
|---|---|---|
| `LWS_GROUP_SIZE` | 그룹 pod 수 | `--data-parallel-size` |
| `LWS_WORKER_INDEX` | leader=0, worker=1..N-1 | `--data-parallel-start-rank` |
| `LWS_LEADER_ADDRESS` | leader의 DNS 주소 | `--data-parallel-address` |

**"mp"인 이유**: 이 구성엔 Ray가 아예 없다. 각 pod가 자기 rank를 알고 직접
`vllm serve`를 실행하며, cross-node 배선은 vLLM 자체의 ZMQ/TCP handshake가
한다. 즉 **mp+headless에서 "사람이 노드마다 명령 치던 일"을 LWS 컨트롤러가
대신하는 것**이 전부다. 우리가 이미 이해한 메커니즘 그대로다.

우리 fleet(1 GPU/노드, DP=N)용으로 변환한 command — 공식 LWS 예제
(docs/deployment/frameworks/lws.md — TP/PP용이라 `--nnodes` 사용)를 DP 문서
(docs/serving/data_parallel_deployment.md:37-46) 레시피로 바꾼 것:

```yaml
# leaderTemplate command
vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code \
  --data-parallel-size $(LWS_GROUP_SIZE) --data-parallel-size-local 1 \
  --data-parallel-address $(LWS_LEADER_ADDRESS) --data-parallel-rpc-port 13345 \
  --enable-expert-parallel --port 8000
# workerTemplate command (차이: --headless + start-rank, --port 없음)
vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code \
  --headless --data-parallel-size $(LWS_GROUP_SIZE) --data-parallel-size-local 1 \
  --data-parallel-start-rank $(LWS_WORKER_INDEX) \
  --data-parallel-address $(LWS_LEADER_ADDRESS) --data-parallel-rpc-port 13345 \
  --enable-expert-parallel
```

부속물: `nvidia.com/gpu: 1` 리소스, `/dev/shm` emptyDir(Memory), leader만 고르는
ClusterIP Service(`role: leader` 셀렉터), leader readinessProbe(8000 tcp).
NCCL env(`NCCL_SOCKET_IFNAME` 등)는 pod env로 명시 — ray_cluster.sh가 하던 일이
매니페스트로 이동한다.

현재 구성과의 대응표:

| 지금 (Terraform+Ray) | LWS-mp 후 |
|---|---|
| ray_cluster.sh head/worker 수동 실행 | 없음 — LWS가 그룹 생성 |
| serve_ep.sh (head에서 1회) | leaderTemplate command |
| (Ray가 원격 엔진 actor 생성) | workerTemplate command (--headless) |
| NCCL env를 ray start 시점에 export | pod spec의 env 블록 |
| 스팟 중단 → 수동 복구 | RecreateGroupOnPodRestart 자동 재생성 |

DP 문서의 부수 발견: `--data-parallel-size-local 0`으로 "첫 노드는 API 서버만,
엔진은 전부 다른 노드" 토폴로지가 **공식 문서에 명시**돼 있다
(data_parallel_deployment.md:49-58). CPU 전용 head 논의와 연결 — 단 GPU 전무
노드에서의 플랫폼 감지 문제는 별도 검증 필요(기존 결론 유지).

### 4-2. EEP의 K8s(mp) 포팅 설계

**경계 분석** — Ray에 묶인 부분은 `CoreEngineActorManager.scale_up_elastic_ep`
(utils.py:826-943) 하나이며, 하는 일의 전부는:
① placement 계산 → ② rank별 VllmConfig 사본 준비 → ③ **actor 생성 + 인자 전달**
(`vllm_config, executor_class, log_stats, local_client, addresses(ZMQ),
dp_rank, local_dp_rank` + env `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH=1`) → ④ init 대기.

어려운 90%는 전부 백엔드 무관 계층에 있다: 엔진 내 `ElasticEPScalingState`
상태기계, stateless NCCL 재그룹, TCPStore 2단 배리어, 비-expert weight NCCL
전송(신규 rank는 dummy-load, HF 다운로드 불필요), EPLB 재배치, ZMQ 소켓
handover — 엔진은 자기를 누가 띄웠는지 모른다.

**포팅 컴포넌트**:

| 컴포넌트 | 역할 | 규모 |
|---|---|---|
| `K8sEngineManager` | scale_up/scale_down 두 메서드 — Ray actor 대신 K8s API로 pod 생성/삭제 | ~200줄 |
| join-mode 엔진 entrypoint | pod 진입점: 직렬화된 vllm_config+addresses+dp_rank를 env/ConfigMap으로 받아 엔진 코어 인스턴스화 (Ray actor 생성자의 CLI판) | ~100-200줄 |
| readiness/failure 감시 | `wait_for_init` 대응(기존 ZMQ ready 신호 활용) + pod 상태 watch | ~100줄 |

scale-down은 더 단순: 제거 대상 rank의 pod 삭제 + 잔존 엔진 재구성 RPC는
기존 클라이언트 흐름이 이미 처리.

**리스크/전제**: pod 네트워크 NCCL은 hostNetwork 권장(CNI 오버레이 오버헤드),
재구성 중 pod 실패 처리 설계 필요, boot DP가 하한이라는 제약은 그대로.
예상 규모: 코드 수백 줄 + **분산 bring-up 디버깅이 본체** — 프로토타입 1~2주.
선행조건: 현 Ray 구성에서 EEP 데모를 먼저 돌려 재구성 파이프라인 감각 확보.

**Upstream 정렬**: RFC #20323 로드맵에 "K8s 연동/Ray 디커플링"이 미해결 항목
— 이 포팅은 연구 산출물이자 기여 후보. 착수 시 AGENTS.md 절차대로 중복
PR/RFC 확인 필수.

---

## 종합 로드맵 제안

1. **지금**: EP 기본 실험 마무리 (현 serve_ep.sh)
2. **다음**: EEP 데모 — 설정만으로 되고, 우리 fleet이 "multi-node elastic EP
   over TCP"라는 upstream 미측정 영역이라 측정 자체가 기여가 됨
3. **병행**: k3s+LWS 전환 (2-3일, 학습 목적 달성 + 스팟 자동복구 확보 — §4-1)
4. **연구 트랙**: EEP의 K8s(mp) 포팅 (§4-2) — Ray 결합부가 얇아 수백 줄 규모,
   upstream 미해결 항목이라 기여 후보. LWS 전환과 EEP 데모 경험이 선행조건
5. **필요 시**: rank→IP 고정/노드별 개수 지정은 monkeypatch 래퍼로 —
   vLLM 테스트가 쓰는 공식적 이음새라 버전업에도 비교적 안전
6. 이기종 실험 단계에서는 (기존 결론대로) 풀별 독립 인스턴스 + 자체 라우터

## 근거 (주요 코드 위치)

- `vllm/envs.py:173` VLLM_RAY_DP_PLACEMENT_NODE_IPS /
  `vllm/v1/engine/utils.py:541-715` placement 로직 전체
- `tests/v1/engine/test_core_engine_actor_manager.py:170-181` monkeypatch 계약
- `vllm/v1/engine/coordinator.py` flat-world 확인
- elastic EP: `vllm/entrypoints/serve/elastic_ep/`, `vllm/v1/engine/core_client.py:1551-1750`,
  `vllm/distributed/elastic_ep/`, `tests/distributed/test_elastic_ep.py`,
  `examples/ray_serving/elastic_ep/serve_deepseek_v2.sh`
- `vllm/config/parallel.py:35` 백엔드 Literal (MPI 부재)
- upstream: RFC #20323, PR #34861/#47288, vLLM 블로그 "Elastic Expert
  Parallelism in vLLM" (2026-05-14), docs LWS/data_parallel_deployment/kuberay
