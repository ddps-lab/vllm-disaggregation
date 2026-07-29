# NCCL `double free or corruption` 크래시 — 원인 분석 및 수정 보고서

**대상 실험**: `experiment/EP` (vLLM 0.26 Expert Parallel, Ray DP backend)
**하드웨어**: 2× g6.xlarge spot (L4 24GB, EFA/InfiniBand 없음), AWS Deep Learning AMI
**스택**: vLLM 0.26 · torch 2.11.0 · **NCCL 2.28.9** (`nvidia-nccl-cu13==2.28.9`, torch 2.11.0가 핀)
**작성일**: 2026-07-28

---

## 1. TL;DR

`vllm serve`로 EP를 띄우면 `ncclCommInitRank` 도중 **`double free or corruption (fasttop)` → SIGABRT**로 워커가 죽었다.

- **직접 원인**: NCCL 2.28.9의 net 플러그인 초기화 루프가 여러 플러그인이 **공유하는 `comm->netContext` 포인터를 finalize 후 NULL로 리셋하지 않는** use-after-free 버그.
- **방아쇠(irony)**: 우리가 "IB 없으니 끄자"고 넣었던 **`NCCL_IB_DISABLE=1`이 바로 크래시를 유발**했다. 이 플래그가 IB init을 *ctx를 덮어쓰기 전에* 실패시켜, 앞선 플러그인이 남긴 dangling 포인터가 살아남아 두 번 free된다.
- **수정**: `ray_cluster.sh`에서 `NCCL_IB_DISABLE=1`을 제거하고 **`NCCL_NET=Socket` + `NCCL_NET_PLUGIN=none`**으로 교체. 버그 경로 자체가 실행되지 않는다.
- **검증**: NCCL v2.28.9-1 소스를 5개 에이전트로 독립·교차 검증(적대적 반증 포함). 핵심 메커니즘 high-confidence.

---

## 2. 증상

`serve_ep.sh` 실행 시 원격 Ray 워커에서:

```
(RayWorkerProc pid=2158, ip=192.168.10.254) double free or corruption (fasttop)
(RayWorkerProc pid=2158, ip=192.168.10.254) *** SIGABRT received at time=... ***
...
(RayWorkerProc ...)     @   ...  cfree
(RayWorkerProc ...)     @   ...  ncclIbFinalize()
(RayWorkerProc ...)     @   ...  commAlloc()
(RayWorkerProc ...)     @ ... and at least 1 more frames
Fatal Python error: Aborted

Stack (most recent call first):
  File ".../pynccl_wrapper.py", line 423 in ncclCommInitRank
  File ".../pynccl.py", line 137 in __init__
  File ".../cuda_communicator.py", line 84 in __init__
  File ".../parallel_state.py", line 487 in __init__ (init_model_parallel_group)
  ...
  File ".../gpu_worker.py", line 1444 in init_worker_distributed_environment
```

핵심 단서 두 가지:

1. glibc 힙 진단 문자열 **`double free or corruption (fasttop)`** — 이미 free된 non-NULL 청크를 다시 `free()`했다는 뜻. 단순 refcount underflow나 `free(NULL)`은 이 메시지를 내지 않는다.
2. C 스택 **`cfree ← ncclIbFinalize() ← commAlloc()`** — 크래시하는 `free()`는 NCCL **내부 IB 플러그인**의 `ncclIbFinalize`이고, comm 최초 초기화(`commAlloc → ncclNetInit`) 도중 발생.

---

## 3. 근본 원인

### 3.1 NCCL 2.28의 per-comm 플러그인 초기화 루프

NCCL 2.28부터 net 플러그인 초기화가 **comm마다** 일어난다. `ncclNetInit`은 등록된 플러그인들을 **하나의 루프**로 순회하며 각각 init을 시도한다:

- 등록 순서 (`initPluginLibsOnceFunc`, [`plugin/net.cc:316-337`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net.cc#L316-L337)):
  `[0]` 외부 플러그인 `libnccl-net.so` → `[1]` 내부 **IB** (`&ncclNetIb`) → `[2]` 내부 **Socket** (`&ncclNetSocket`).
- 루프 ([`plugin/net.cc:355-377`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net.cc#L355-L377))는 각 플러그인의 init에 **동일한 `comm->netContext` 필드의 주소**를 넘긴다 ([`plugin/net.cc:178`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net.cc#L178)):

  ```c
  pluginLib->ncclNet->init(&comm->netContext, comm->commHash, &commConfig, ...);
  ```

  즉 **모든 플러그인 시도가 같은 포인터 슬롯을 공유**한다.

### 3.2 버그: finalize 후 `comm->netContext`를 NULL로 리셋하지 않음

init이 실패하면 fail 경로가 현재 플러그인의 finalize를 부르는데, **포인터를 리셋하지 않는다** ([`plugin/net.cc:224-232`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net.cc#L224-L232)):

```c
fail:
  ...
  pluginLib->ncclNet->finalize(comm->netContext);   // ← free 하지만
  ...
  goto exit;                                         // ← comm->netContext = NULL 이 없음
```

에러는 삼켜지고(루프는 다음 플러그인으로 계속됨), `comm->netContext`에는 **이미 free된 포인터가 그대로 남는다(dangling)**. 소스 전체에서 `comm->netContext`는 init(:178)과 parent 상속(:385)에서만 대입되고, **finalize 이후 NULL로 되돌리는 코드는 어디에도 없다.**

> 같은 dangling을 만드는 두 번째 경로: init은 성공했지만 device-version 체크에서 거부되는 경우([`net.cc:236`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net.cc#L236) → `ncclNetPluginFinalize` [`net.cc:340-341`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net.cc#L340-L341))도 동일하게 free 후 리셋하지 않는다.

### 3.3 "첫 번째 free" — 외부 aws-ofi 플러그인 (v10 API shim)

AWS DLAMI는 **aws-ofi-nccl**를 `/opt/amazon/ofi-nccl/lib`(LD_LIBRARY_PATH 등록)에 탑재하고, NCCL의 기본 외부 플러그인 이름이 `libnccl-net.so`이므로([`net.cc:282`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net.cc#L282)) 자동 로드된다.

이 플러그인이 **v10 net API**로 로드되면, NCCL의 v10 호환 shim이 **실제 init을 부르기 전에** ctx를 할당해 슬롯에 넣는다 ([`plugin/net/net_v10.cc:73-81`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net/net_v10.cc#L73-L81)):

```c
NCCLCHECK(ncclCalloc(&config_v10, 1));
config_v10->trafficClass = config->trafficClass;
*ctx = config_v10;                       // ← comm->netContext = 힙 포인터 (실제 init 전에!)
if (refCount[NET_INDEX]++) return ncclSuccess;
NCCLCHECK(ncclNet_v10->init(logfn, proffn));   // ← g6.xlarge엔 EFA 없어 여기서 실패
```

g6.xlarge에는 EFA가 없어 aws-ofi 실제 init이 실패한다. 그러면 `net.cc:226`의 fail 경로가 shim의 finalize를 부르고, 이 shim은 **ctx를 free한다** ([`net_v10.cc:61-65`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net/net_v10.cc#L61-L65)):

```c
static ncclResult_t ncclNet_finalize(void* ctx) {
  refCount[NET_INDEX]--;
  free(ctx);            // ← 첫 번째 free
  return ncclSuccess;
}
```

결과: `config_v10`는 free됐지만 `comm->netContext`는 여전히 그 주소를 가리킨다 → **dangling**.

### 3.4 "두 번째 free" — 왜 `NCCL_IB_DISABLE=1`이 방아쇠인가 (핵심)

루프는 다음으로 내부 IB 플러그인을 시도한다. `ncclIbInit`은 이렇게 생겼다 ([`transport/net_ib.cc:849-857`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/transport/net_ib.cc#L849-L857)):

```c
ncclResult_t ncclIbInit(void** ctx, ...) {
  NCCLCHECK(ncclIbInitDevices(...));         // (A)
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));  // (B)
  netCommConfig->trafficClass = ...;
  *ctx = (void *)netCommConfig;              // (C) ← comm->netContext 덮어쓰기
  ...
}
```

`NCCL_IB_DISABLE=1`이면 `ncclIbInitDevices`가 **(C)에 도달하기 전에** 조기 실패한다 ([`net_ib.cc:677`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/transport/net_ib.cc#L677)):

```c
if (ncclParamIbDisable()) return ncclInternalError;   // ← 여기서 리턴
```

즉 IB 시도는 **`comm->netContext`를 덮어쓰지 못하고**, 3.3에서 남긴 dangling 포인터가 그대로 살아 있다. 이어서 IB의 fail 경로가 `ncclIbFinalize(comm->netContext)`를 호출하는데, 이 함수는 **무조건** free한다 ([`net_ib.cc:2659-2662`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/transport/net_ib.cc#L2659-L2662)):

```c
ncclResult_t ncclIbFinalize(void* ctx) {
  free(ctx);                        // ← 두 번째 free (같은 포인터!) → double free
  return ncclIbFinalizeDevices();
}
```

**같은 `config_v10` 포인터가 두 번 free** → `double free or corruption (fasttop)` → SIGABRT. 스택의 `cfree ← ncclIbFinalize() ← commAlloc()`과 정확히 일치한다.

#### 왜 IB_DISABLE이 **없으면** 안 죽나

`NCCL_IB_DISABLE`이 없으면, IB 하드웨어가 없어도 `ncclIbInitDevices`는 "No device found"를 로그하고 **성공을 리턴**한다 ([`net_ib.cc:825-844`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/transport/net_ib.cc#L825-L844)). 그러면 `ncclIbInit`이 (B)(C)까지 진행해 **새 `netCommConfig`로 `comm->netContext`를 덮어쓴다.** dangling 포인터가 안전하게 사라지고, 이후 free는 방금 할당한 fresh 청크를 대상으로 하므로 double free가 아니다.

> **정리**: "IB를 끈다"는 설정이 이 버전에서는 오히려 크래시를 만드는 조합이었다. dangling 포인터를 덮어써 무해화하던 코드 경로를 IB_DISABLE이 건너뛰게 만들기 때문.

### 3.5 필요조건 요약

이 double free가 성립하려면 **세 가지가 동시에** 필요하다:

1. `comm->netContext`에 heap 포인터를 넣고 free하는 **선행 플러그인**이 존재 → 실무상 **aws-ofi(`libnccl-net.so`) v10** 로드 + init 실패 (DLAMI × non-EFA 인스턴스의 전형).
2. **`NCCL_IB_DISABLE=1`** → IB init이 `*ctx` 덮어쓰기 전에 실패해 dangling 포인터 생존.
3. NCCL의 근본 버그: 공유 `comm->netContext`를 finalize 후 NULL로 리셋하지 않음 + `ncclIbFinalize`의 무조건 `free()`.

내부 Socket 플러그인은 ctx를 건드리지 않으므로([`net_socket.cc`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net/net_socket.cc)), 외부 플러그인 없는 순수 IB↔Socket 구성은 `free(NULL)`만 일어나 안전하다. 즉 **aws-ofi는 방아쇠, `NCCL_IB_DISABLE=1`은 필요조건, NCCL 자체 버그가 진짜 뿌리다.**

---

## 4. 수정

`experiment/EP/ray_cluster.sh`에서 `NCCL_IB_DISABLE=1`을 제거하고 다음으로 교체:

```bash
export NCCL_NET=Socket        # Socket 플러그인만 init → IB/외부 플러그인 버그 경로 미실행
export NCCL_NET_PLUGIN=none   # libnccl-net.so(aws-ofi) 로드 자체 차단 → 첫 번째 free 트리거 제거
```

두 변수가 **독립적으로** 버그 체인을 끊으며, 함께 쓰면 belt-and-suspenders다:

- **`NCCL_NET=Socket`** — `NCCL_NET` env는 `comm->config.netName`이 되고([`init.cc:1706-1714`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/init.cc#L1706-L1714)), 루프의 netName 필터([`net.cc:359-360`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net.cc#L359-L360))가 이름이 `"Socket"`인 플러그인만 init한다. IB(`"IB"`)와 외부 플러그인의 **init/finalize 경로가 아예 실행되지 않는다.**
  - 뉘앙스: 이 설정도 `libnccl-net.so`를 dlopen하기는 한다(로드는 netName과 무관). 하지만 init/finalize를 안 하므로 dangling ctx를 남길 수 없다.
- **`NCCL_NET_PLUGIN=none`** — `"none"`은 빈 리스트로 처리되어([`net.cc:292-293`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net.cc#L292-L293)) 외부 플러그인이 **로드조차 되지 않는다.** dangling ctx를 만드는 최초 트리거(aws-ofi)를 원천 제거.

g6.xlarge에는 어차피 EFA/IB가 없어 실제 전송은 TCP socket이므로, Socket 강제는 성능상 손해도 없다.

### 재배포 방법

env는 `ray start` **전에** export돼야 Ray 워커 액터까지 상속되므로(로컬 serve 셸에서 export하면 head에만 적용되는 함정), **양쪽 노드 모두 Ray를 재시작**해야 한다:

```bash
# 로컬에서 최신 스크립트 동기화
./experiment/EP/sync.sh 100.23.153.138 32.186.232.33

# head (private 192.168.10.144):
ray stop && ./experiment/EP/ray_cluster.sh head

# worker (private 192.168.10.254):
ray stop && ./experiment/EP/ray_cluster.sh worker 192.168.10.144

# head에서 serve:
./experiment/EP/serve_ep.sh
```

만전을 기하려면 serve 셸에서도 명시적으로:
`NCCL_NET=Socket NCCL_NET_PLUGIN=none ./serve_ep.sh`

---

## 5. 검증 방법

이 보고서의 인과 사슬은 **NCCL v2.28.9-1 원본 소스**를 직접 읽어(로컬에서 vLLM/NCCL을 실행하지 않음 — 프로젝트 규칙 준수) 확인했다. 5개 에이전트가 각 주장을 독립적으로 검증하고, 그중 하나는 전체 재구성을 **반증(refute)**하도록 지시했다:

- 공유 `comm->netContext` + finalize 후 non-null → **confirmed** (high).
- `NCCL_IB_DISABLE=1`이 `*ctx` 쓰기 전 조기 실패시켜 dangling 생존 → **confirmed** (high).
- `NCCL_NET=Socket` / `NCCL_NET_PLUGIN=none`이 경로를 끊음 → **confirmed** (high).
- 적대적 검증: refcount underflow / GIN finalize 등 대안 원인은 **모두 기각**(glibc `fasttop`은 non-NULL 재free를 뜻하므로 `ncclIbFinalize`의 무조건 free가 유일한 정합 설명).

초기 재구성에서 교정된 부분:
- "v10/v11 shim" → **v10만** ctx를 선할당한다. v11 shim은 순수 passthrough([`net_v11.cc:15-22`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net/net_v11.cc#L15-L22))라, 외부 플러그인이 v11이면 alloc/free-without-null은 플러그인 자체 코드 몫이다.
- 첫 free는 v10 shim의 `ncclNet_finalize`, 크래시하는 두 번째 free는 내부 IB의 `ncclIbFinalize` — **서로 다른 함수가 같은 공유 포인터를 free**하는 것이 double free의 본질.

---

## 6. 상위 시사점 (upstream)

이 버그의 진짜 뿌리는 aws-ofi가 아니라 **NCCL 2.28.9 자체**다:

1. `ncclNetInit` 루프가 공유 `comm->netContext`를 finalize한 뒤 NULL로 리셋하지 않음 (use-after-free/double-free의 구조적 원인).
2. `ncclIbFinalize`는 무조건 `free(ctx)` — 반면 GIN 쪽 `ncclGinIbFinalize`는 `if (ctx) free(ctx)`로 가드한다([`net_ib.cc:2707-2708`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/transport/net_ib.cc#L2707-L2708)). IB finalize에도 같은 가드/리셋이 있었다면 이 크래시는 없었다.

즉 env로 우회(위 수정)는 확실히 동작하지만, **잠재 버그는 남아 있어** 다른 외부 플러그인(v10 API로 ctx를 선할당하는 종류)과 IB 실패 조합에서 재발할 수 있다. 향후 NCCL 버전 업 시 이 경로가 고쳐졌는지 확인할 가치가 있다.

---

## 7. 참고 — 소스 위치 (NCCL v2.28.9-1)

| 위치 | 내용 |
|---|---|
| [`plugin/net.cc:178`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net.cc#L178) | 모든 플러그인이 `&comm->netContext` 공유 |
| [`plugin/net.cc:224-232`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net.cc#L224-L232) | fail 경로: finalize 후 NULL 리셋 없음 (핵심 버그) |
| [`plugin/net.cc:316-337`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net.cc#L316-L337) | 플러그인 등록 순서 + `NCCL_NET_PLUGIN=none` 처리 |
| [`plugin/net.cc:355-377`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net.cc#L355-L377) | 초기화 루프 + netName 필터 |
| [`plugin/net/net_v10.cc:73-81`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net/net_v10.cc#L73-L81) | v10 shim: 실제 init 전 ctx 선할당 |
| [`plugin/net/net_v10.cc:61-65`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/plugin/net/net_v10.cc#L61-L65) | v10 shim finalize: `free(ctx)` (첫 free) |
| [`transport/net_ib.cc:677`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/transport/net_ib.cc#L677) | `NCCL_IB_DISABLE` 조기 실패 지점 |
| [`transport/net_ib.cc:849-857`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/transport/net_ib.cc#L849-L857) | `ncclIbInit`: `*ctx`는 InitDevices 성공 후에만 기록 |
| [`transport/net_ib.cc:2659-2662`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/transport/net_ib.cc#L2659-L2662) | `ncclIbFinalize`: 무조건 `free(ctx)` (둘째 free, 크래시 프레임) |
| [`init.cc:421`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/init.cc#L421) | `commAlloc → ncclNetInit` (스택의 `commAlloc` 프레임) |
| [`init.cc:1706-1714`](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/init.cc#L1706-L1714) | `NCCL_NET` → `comm->config.netName` |

---

## 8. Upstream 상태 — 최신 버전 · 수정 여부 · 리포트 여부 (조사 2026-07-28)

### 8.1 현재 최신 NCCL

**v2.30.7-1** (2026-06-04). 이후 릴리스는 부속 패키지(nccl4py, nccl-ep)뿐이고 코어 NCCL은 2.30.7이 최신.

### 8.2 고쳐졌나? → **예. v2.29.7-1 (2026-02-27)에서 수정.**

`src/plugin/net.cc`의 `ncclNetPluginInit`에 **`bool initCompleted` 가드**가 추가되어, **플러그인 init이 실제로 완료됐을 때만** finalize를 호출한다:

```c
bool initCompleted = false;
...
if (pluginLib->ncclNet->init(&comm->netContext, ...) != ncclSuccess) goto fail;
initCompleted = true;              // init 성공해야 true
...
fail:
  if (initCompleted) pluginLib->ncclNet->finalize(comm->netContext);   // ← 가드
```

우리 크래시는 IB init이 실패한 뒤(`initCompleted == false`) `ncclIbFinalize`가 dangling 포인터를 재free하는 것이었으므로, 이 가드가 정확히 차단한다. 소스 이분 탐색으로 확정한 취약 구간:

| 버전 | 상태 |
|---|---|
| v2.28.7 / **v2.28.9 (우리 버전)** / v2.29.2 / v2.29.3 | **버그 있음** (무조건 finalize) |
| **v2.29.7-1** (2026-02-27) 이상 | **수정됨** (`initCompleted` 가드) |
| v2.30.x | 수정 유지 |

> **수정 위치 주의**: `ncclIbFinalize` 자체는 최신 v2.30.7에서도 여전히 무조건 `free(ctx)`다 ([net_ib/init.cc:588](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/transport/net_ib/init.cc#L588); 2.30.x에서 `net_ib.cc`가 `net_ib/` 디렉터리로 분리됨). 즉 근본 패턴(공유 `netContext` 미리셋 + 무조건 free)은 남아 있고, 수정은 **호출부(`plugin/net.cc`)에서 실패한 플러그인을 finalize하지 않도록** 막는 방식이다.

### 8.3 리포트된 적 있나? → **공개 리포트는 없음. NVIDIA가 내부적으로 발견·수정한 것으로 보임.**

4갈래 검색(NVIDIA/nccl 이슈·PR·커밋 / vLLM 이슈 / PyTorch 이슈 / 전역 GitHub + 웹·포럼)에서 **우리 시그니처와 일치하는 공개 리포트는 하나도 없었다.** 특징적 토큰 `ncclIbFinalize`는 GitHub 이슈 검색 결과가 0건.

단, **NVIDIA 공식 릴리스 노트에는 수정이 명시**돼 있다 — NCCL 2.29.7 "Fixed Issues" (RN-08645-000_v2.29.7, p.12):

> *"Fixed a crash when calling ncclNet.finalize() after a failed ncclNet_v10->init()."*

이 문구는 우리가 소스에서 재구성한 시나리오(**v10 shim = aws-ofi 플러그인의 init 실패 후 finalize 호출**)와 정확히 일치한다. 주목할 점:

- 같은 목록의 다른 항목들은 GitHub 이슈 번호를 달고 있으나(#1960, #1950, #2019, #1962), **이 항목만 이슈 링크가 없다** → NVIDIA가 공개 리포트가 아니라 **내부에서 발견**했다는 강한 정황.
- **GitHub 릴리스 본문(markdown)의 "Bug fixes" 목록에는 이 줄이 빠져 있다.** 공식 PDF/HTML 릴리스 노트에만 있음. 즉 "완전 무공개(silent)"는 아니고 **문서화는 됐으나 눈에 잘 안 띄고, 추적 이슈가 없는** 상태.

가장 근접했던 무관 항목들(참고용, 모두 우리 버그 아님): NVIDIA/nccl [#1913](https://github.com/NVIDIA/nccl/issues/1913)(2.28.9 GIN×외부 플러그인, EFA 있는 호스트), [#2200](https://github.com/NVIDIA/nccl/pull/2200)(반대 방향 — context *leak* 수정), [#2000](https://github.com/NVIDIA/nccl/issues/2000)(inspector 플러그인 UAF), vLLM [#27628](https://github.com/vllm-project/vllm/issues/27628)(torch 업그레이드 후 net 플러그인 선택).

### 8.4 우리에게 주는 실무적 결론

torch 2.11.0은 **`nvidia-nccl-cu13==2.28.9`를 핀**하므로 우리 스택은 **버그가 있는 쪽**이다. 선택지:

- **(권장) env 워크어라운드 유지** — `NCCL_NET=Socket` + `NCCL_NET_PLUGIN=none` (§4). 2.28.9에서 즉시 동작하고, EFA 없는 g6.xlarge에선 성능 손해도 없음.
- **(대안) NCCL ≥ 2.29.7로 교체** — 근본 수정이 들어간 버전. 다만 torch 2.11이 번들한 2.28.9를 덮어써야 하므로(휠 교체 또는 `LD_PRELOAD`) ABI/호환성 리스크가 있어, 지금 실험 단계에선 env 워크어라운드가 더 안전하다.

---

**수정 파일**: [ray_cluster.sh](ray_cluster.sh) · **관련**: [serve_ep.sh](serve_ep.sh), [README.md](README.md)
