# Disagg-Exp Tier-1 — vLLM 공식 도구 기반 PD 분리 실험

> Branch: **`experiment/tier1-vllm-benchmark`** — vLLM upstream의 공식 도구만 사용.
> 커스텀 connector / 커스텀 클라이언트 / 커스텀 분석 모두 제거. 위험 surface 최소화.

vLLM fork (`releases/v0.21.0`) 기반의 prefill/decode disaggregation 비용대비가치 평가.
모델 `meta-llama/Llama-3.1-8B-Instruct` (BF16, ~16GB), 리전 us-west-2.

> **핵심 질문**: 같은 예산이면 큰 GPU 한 장이 나은가, 작은 GPU 여러 장 + PD 분리가 나은가?
>
> 같은 워크로드를 6가지 토폴로지에 던지고 **TTFT / TPOT / $/M-tokens**로 비교한다.

---

## 1. 이 브랜치가 다른 점

| 영역 | `experiment/tier1` (커스텀) | **이 브랜치 (공식)** |
|---|---|---|
| KV connector | `InstrumentedLMCacheConnector` (LMCache 0.3.9 wrapping + NIXL/UCX) | **`P2pNcclConnector`** (vLLM upstream, NCCL send/recv) |
| 전송 백엔드 | NIXL (cuda_copy/shm/tcp) | **NCCL** (auto: PCIe/SHM same-node, TCP cross-node) |
| Proxy | LMCache용 `disagg_proxy_server.py` (httpx) | **공식 `disagg_prefill_proxy_server.py`** (quart) |
| 클라이언트 | 커스텀 `sweep.py` (Poisson, 2-phase warmup/measured) | **`vllm bench serve`** 공식 래퍼 (`sweep_official.py`) |
| 분석 | 커스텀 `analyze.py` (JSONL → 표) | **`analyze_official.py`** (`vllm bench serve` 결과 JSON → 표) |
| 커스텀 Python | 약 3 파일, 300+ 줄 | **0 파일.** Wrapper bash + Python만 (실험 grid 자동화 용도) |

→ 동일한 측정 대상(TTFT/TPOT/$/M-tokens)을 vLLM이 공식적으로 보장하는 코드 경로로 측정.

---

## 2. 개념 정리 (왜 이렇게 짰는가)

### 2.1 Prefill vs Decode

LLM 추론은 두 단계로 나뉜다:

- **Prefill**: 입력 프롬프트 전체를 한 번에 통과시켜 KV cache를 만든다. 계산 집약적 (compute-bound).
- **Decode**: 토큰을 하나씩 생성한다. 메모리 대역폭 집약적 (memory-bound).

**Monolithic**: 두 단계를 같은 인스턴스에서 처리.
**PD Disaggregation**: prefill 노드와 decode 노드를 분리 → 각 단계를 최적 GPU에 배치 가능.

### 2.2 Tensor Parallel (TP) vs Pipeline Parallel (PP)

- **TP=N**: 한 레이어를 N개 GPU가 분담 (행렬 곱을 행/열로 자름). 통신 빈도 ↑, latency ↓.
- **PP=N**: 레이어를 N개 GPU에 순차 배치. 통신 빈도 ↓, throughput ↑.
- 8B 모델은 T4 16GB 단독에 안 올라가서 **TP ≥ 2 또는 PP ≥ 2** 필수.

### 2.3 P2pNcclConnector — 어떻게 작동하나

[vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py](../vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py) 참고.

1. **Control plane** (ZMQ): prefill ↔ decode가 `kv_port`로 ZMQ DEALER/ROUTER 연결. NCCL communicator의 `unique_id` 교환.
2. **Data plane** (NCCL): `ncclSend` / `ncclRecv`로 KV tensor 직접 전송.
3. **버퍼**: `TensorMemoryPool`이 pinned **host RAM** 위에 풀 잡음 (기본 32GB). cross-node는 GPU↔host↔NIC로 흐름.
4. **NCCL transport 자동 선택**: NVLink → PCIe P2P → SHM → IB → TCP sockets. AWS는 보통 **PCIe/SHM** (same-node) 또는 **TCP** (cross-node)로 떨어짐.

### 2.4 변인 통제 및 결과 오염 방지 (Anti-Contamination)

1. **캐시 오염 방지 (`--no-enable-prefix-caching`)** — 같은 프롬프트 재사용 방지. 모든 요청이 정직하게 전체 prefill + KV 전송.
2. **`PYTHONHASHSEED=123`** — cross-node에서 직렬화 순서 동기화.
3. **`--no-enable-chunked-prefill` (prefill role만)** — 한 번에 prefill해서 compute bound 측정. PD에서는 chunked prefill이 KV transfer와 충돌할 수 있어 끄는 게 안전.
4. **CUDA Graph ON** — `--enforce-eager` 안 씀. 성능 왜곡 방지. **단** P2pNccl과 충돌 시 `ENFORCE_EAGER=1` env로 fallback.
5. **`--max-num-seqs 512`** — 배치 제한을 풀어서 throughput 최대.

> ⚠️ **WARNING — `MAX_NUM_SEQS=512` & P2pNccl consumer buffer**
>
> 공식 xpyd 예제는 `--max-num-seqs 256` 사용. 우리는 spec대로 512 유지.
> 그러나 consumer 측 `kv_buffer_size` 풀이 작으면 KV transfer back-pressure가 걸려 측정 왜곡 가능.
> **첫 노드 검증 단계에서 `MAX_NUM_SEQS=256`으로도 한번 돌려서 결과 비교 권장.**
> 차이가 크면 buffer 풀이 병목 → 512에서 측정한 값은 부정확.

---

## 3. 6가지 Config

| Config | Instance | GPU 구성 | Topology | $/hr OD |
|---|---|---|---|---:|
| **A1** | g4dn.12xlarge | 4× T4 16GB | Monolithic **TP=2 PP=2** | $3.91 |
| **A2** | g4dn.12xlarge | 4× T4 16GB | Monolithic **TP=4 PP=1** | $3.91 |
| **A3** | g4dn.12xlarge | 4× T4 16GB | Monolithic **TP=1 PP=4** | $3.91 |
| **B**  | g6e.xlarge | 1× L40S 48GB | Monolithic TP=1 | $1.86 |
| **C1** | g4dn.12xlarge | 4× T4 16GB | Same-node PD **TP=2 PP=1**: prefill GPU 0-1 + decode GPU 2-3 | $3.91 |
| ~~**C2**~~ | ~~g4dn.12xlarge~~ | ~~4× T4~~ | ~~Same-node PD **TP=1 PP=2**~~ | ~~$3.91~~ |
| **D**  | 2× g6.xlarge | 2× L4 24GB | Cross-node PD: prefill TP=1 + decode TP=1 | $1.61 |

### ⚠️ configC2는 이 브랜치에서 NOT SUPPORTED

[p2p_nccl_connector.py:528-530](../vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py#L528-L530)에 명시:
> "Currently, only symmetric TP is supported. Asymmetric TP, **PP**, and others will be supported in future PRs."

PP > 1 PD는 P2pNcclConnector가 거부. `launch_configs.sh`의 `configC2_*` 함수는 호출 시 exit 2로 종료.
TP=1 PP=2 변형이 필요하면 **`experiment/tier1` 브랜치 (LMCache+NIXL)** 사용.

### 비교 의미

- **A1↔A2↔A3**: 동일 4× T4. Monolithic의 최적 TP/PP 탐색.
- **A↔C1**: 동일 4× T4. Monolithic vs same-node PD → 순수 소프트웨어 비교.
- **B↔A**: 큰 GPU 1장 vs 작은 GPU 4장 → 멀티-GPU 오버헤드.
- **C1↔D**: 둘 다 PD. T4 PCIe/SHM vs L4 TCP → $/M-tokens 정규화.
- **B↔D**: 비슷 가격대. 큰 GPU 1장 vs 작은 GPU 2장 + PD.

---

## 4. 파일 구성

```
disagg-exp/
├── setup.sh             # 노드 부트스트랩 (uv, vLLM editable, lmcache, quart, ifstat, DCGM)
├── launch_configs.sh    # vllm serve 디스패처 (P2pNcclConnector 사용)
├── sweep_official.py    # `vllm bench serve` 그리드 자동화 래퍼 + S3 sync
├── analyze_official.py  # 결과 JSON → 표/플롯 + 커스텀 브랜치와 cross-check
└── README.md            # 이 문서
```

> 제거된 파일 (이 브랜치):  
> `instrumented_connector.py`, `sweep.py`, `analyze.py`, `README_official.md`  
> → 모두 커스텀 코드 경로용. 공식 도구로 충분.

### 4.1 `setup.sh` — 노드 부트스트랩 (idempotent)

| 단계 | 내용 |
|---|---|
| 1. `uv` 설치 | Astral 패키지 매니저 |
| 2. `.venv` 생성 | Python 3.12 |
| 3. vLLM editable install | `VLLM_USE_PRECOMPILED=1 uv pip install -e .` |
| 4. LMCache ≥ 0.3.9 | 이 브랜치는 안 쓰지만 vLLM dependency check 통과용 |
| 5. Python 의존성 | httpx, fastapi, aiohttp, numpy, **quart** (proxy) |
| 5b. 시스템 도구 | `ifstat`, `s5cmd` |
| 6. chrony | `chronyc tracking` 스냅샷 |
| 7. DCGM exporter | Prometheus `:9400` |
| 8. Metric collectors | `nvidia-smi dmon` (1Hz), `ifstat` (1Hz), DCGM scrape (2s) — PID 파일 관리 |
| 9. 검증 | `import vllm, lmcache` |

PID 파일: `$EXP_LOG_DIR/.pid_{nvidia_dmon,ifstat,dcgm_loop}` → 재실행 시 안전하게 kill 후 재시작.

### 4.2 `launch_configs.sh` — `vllm serve` 디스패처

각 config + role을 명령행 한 줄로 실행. P2pNccl `kv-transfer-config` JSON을 자동 조립.

| 함수 | 역할 |
|---|---|
| `configA1/A2/A3()` | TP/PP 변형 monolithic, port 8000 |
| `configB()` | TP=1 monolithic, port 8000 |
| `configC1_prefill()` | TP=2 PP=1 (GPU 0-1), `kv_role=kv_producer`, port 8100, kv_port=14600 |
| `configC1_decode()` | TP=2 PP=1 (GPU 2-3), `kv_role=kv_consumer`, port 8200, kv_port=14700 |
| `configC2_*()` | **exit 2** with explanation (P2pNccl PP 미지원) |
| `configD_prefill()` | TP=1, peer=`$DECODER_HOST`, port 8100 |
| `configD_decode()` | TP=1, port 8200 |
| `launch_proxy()` | `benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py` (quart) |

공통 flag (`COMMON_FLAGS`):
```
--no-enable-prefix-caching --dtype bfloat16 --served-model-name llama-3.1-8b
--max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL
--max-num-seqs ${MAX_NUM_SEQS:-512}
```
CUDA Graph **ON** (enforce-eager 안 씀, `ENFORCE_EAGER=1` env로 override).

**Port plan**:
- Prefill HTTP: 8100, kv_port: 14600 (+rank)
- Decode HTTP : 8200, kv_port: 14700 (+rank)
- Proxy HTTP : 8000, ZMQ control: 30001

### 4.3 `sweep_official.py` — 워크로드 그리드 자동화

각 (prefill_len, decode_len, rate) 포인트마다 다음을 실행:

```
vllm bench serve \
  --backend openai --base-url <url> \
  --endpoint /v1/completions --model llama-3.1-8b \
  --dataset-name random --random-input-len $PL --random-output-len $DL \
  --random-range-ratio 0.0 \
  --num-prompts 350 --request-rate $RATE --burstiness 1.0 \
  --ignore-eos --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --save-detailed \
  --result-dir $EXP_LOG_DIR/<config>/ \
  --result-filename p{prefill}_d{decode}_r{rate}.json \
  --extra-body '{"min_tokens": <decode_len>}' \
  --disable-tqdm
```

Resume: `.done_<point>` / `.failed_<point>` 마커. S3 sync 백그라운드 (`raw/official/{date}/{host}/{config}/`).

### 4.4 `analyze_official.py`

`{EXP_LOG_DIR}/{config}/p*_d*_r*.json`을 읽어 표 + matplotlib plot 출력.

- **`--skip-warmup 50`** (기본): `--save-detailed`의 per-request `ttfts[]` 배열에서 앞 50개 버리고 percentile 재계산.
- **`--skip-warmup 0`**: 벤치마크의 자체 집계 그대로 사용.
- **`--compare-custom <dir>`**: `experiment/tier1` 브랜치의 `sweep.py` JSONL 결과와 side-by-side 비교 (교차 검증용).

---

## 5. 워크로드 그리드

```
PREFILL_LENS = [512, 2048, 8192]
DECODE_LENS  = [128, 512, 1024, 4096]
RATES        = [1.0, 2.0, 4.0]

Cross 1 (prefill × rate, decode=512 고정):  3 × 3 = 9 points
Cross 2 (decode × rate, prefill=2048 고정): 4 × 3 = 12 points
                                            (중복 제거)
Total: ~18 points/config
num_prompts = 350 (warmup 50 + measured 300, analyze에서 앞 50 skip)
```

env override: `SWEEP_PREFILL_LENS=512,2048 SWEEP_RATES=1.0,2.0 ...`

---

## 6. AWS 환경 주의사항

### NCCL on AWS (No IB / NVLink)

g4dn / g6 / g6e 모두 InfiniBand 없고 NVLink 없음. NCCL은 자동 fallback:
1. NVLink — 없음
2. PCIe P2P — same-node에서 가능
3. SHM — same-node host shared memory
4. IB Verbs — 없음
5. **TCP sockets** ← cross-node에서 무조건 여기로

NCCL이 IB 찾으러 헛수고하다가 hang 막으려면:
```bash
export NCCL_IB_DISABLE=1                # launch_configs.sh에서 기본 적용
export NCCL_SOCKET_IFNAME=ens5          # 인스턴스 NIC 이름 (ens5/eth0/…)
export NCCL_DEBUG=INFO                  # 첫 노드 검증 시 한 번 켜서 transport 확인
```

NIC 확인: `ip route get 1` 또는 `ip addr | grep -B1 inet`.

### Mem pool sizing

P2pNcclConnector의 `TensorMemoryPool`은 **pinned host RAM**에 풀 잡음. 기본 32GB.

| Instance | Host RAM | 권장 `mem_pool_size_gb` |
|---|---:|---:|
| g4dn.12xlarge | 192GB | 8 (launch_configs 기본) |
| g6.xlarge | 16GB | **4** (default 32GB는 즉시 OOM) |
| g6e.xlarge | 32GB | 8 |

`launch_configs.sh`에서 config별로 명시함.

### CUDA Graph + P2pNccl

공식 xpyd 예제는 `--enforce-eager`를 명시함. 기본 `disaggregated_prefill.sh`는 안 씀.
spec 준수로 **기본 ON 유지**. 검증 단계에서 KV transfer 실패하거나 hang 발생 시:
```bash
ENFORCE_EAGER=1 bash launch_configs.sh configC1 prefill
```

---

## 7. 실행 가이드

### 7.1 노드 부트스트랩 (모든 config 공통)

```bash
git clone <your-fork> && cd vllm-disaggregation
git checkout experiment/tier1-vllm-benchmark

export EXP_LOG_DIR=/home/ubuntu/exp-logs
bash disagg-exp/setup.sh
source .venv/bin/activate

# 검증
python -c "import vllm, lmcache; print(vllm.__version__, lmcache.__version__)"
# 기대: 0.21.x  0.3.9+
curl -s localhost:9400/metrics | grep -m1 DCGM_FI_DEV_FB_USED
chronyc tracking | grep "Last offset"

# NCCL 환경변수 (cross-node 또는 안전책)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=$(ip route get 1 | awk '/dev/{print $5; exit}')
```

### 7.2 Config A1/A2/A3/B — Monolithic

```bash
bash disagg-exp/launch_configs.sh configA1
# (다른 터미널)
python disagg-exp/sweep_official.py --config A1 --base-url http://localhost:8000
```

A2, A3, B 동일 패턴.

### 7.3 Config C1 — Same-node PD (4× T4)

**3개 터미널 필요. decode를 먼저 띄워야 ZMQ가 안정적으로 연결됨.**

```bash
# Terminal 1 — decode (먼저)
bash disagg-exp/launch_configs.sh configC1 decode

# Terminal 2 — prefill (decode가 "Application startup complete" 출력 후)
bash disagg-exp/launch_configs.sh configC1 prefill

# Terminal 3 — proxy (둘 다 ready 후)
bash disagg-exp/launch_configs.sh configC1 proxy

# Terminal 4 — sweep
python disagg-exp/sweep_official.py --config C1 --base-url http://localhost:8000
```

### 7.4 Config D — Cross-node PD (2× g6.xlarge)

**Decode node:**
```bash
export VLLM_HOST_IP=10.0.x.y       # 이 노드의 private IP
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens5     # 실제 NIC 이름으로
bash disagg-exp/launch_configs.sh configD decode
```

**Prefill node:**
```bash
export VLLM_HOST_IP=10.0.x.z       # 이 노드의 private IP
export DECODER_HOST=10.0.x.y       # decode 노드의 private IP
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens5
bash disagg-exp/launch_configs.sh configD prefill
```

**Prefill node (또 다른 터미널, 둘 다 ready 후):**
```bash
bash disagg-exp/launch_configs.sh configD proxy
python disagg-exp/sweep_official.py --config D --base-url http://localhost:8000
```

### 7.5 첫 검증 단계 권장 사항

| 체크 | 명령 / 기대값 |
|---|---|
| vLLM/LMCache 버전 | `python -c "import vllm, lmcache; print(vllm.__version__, lmcache.__version__)"` → 0.21.x / 0.3.9+ |
| DCGM | `curl -s localhost:9400/metrics \| grep DCGM_FI_DEV_FB_USED` → 라인 나옴 |
| Chrony | `chronyc tracking` → offset 수십 ms 이내 |
| NCCL transport | `NCCL_DEBUG=INFO` 한 번 켜서 첫 launch 로그 확인 → "Channel ... via SHM/IPC/SOCKET" 출력 |
| Small sweep | `SWEEP_PREFILL_LENS=512 SWEEP_DECODE_LENS=128 SWEEP_RATES=1.0 SWEEP_NUM_PROMPTS=20 python sweep_official.py --config C1 ...` → JSON에 `completed > 0` |
| **MAX_NUM_SEQS 민감도** | 위 small sweep을 `MAX_NUM_SEQS=256`으로도 한 번 → output_throughput 차이 < 10%면 512 안전. 큰 차이면 buffer pool 병목 의심 |

### 7.6 Config C2 (PP=2 PD) 시도 시

```bash
bash disagg-exp/launch_configs.sh configC2 prefill
# [launch] configC2 is NOT supported with P2pNcclConnector (PP not implemented).
#          Use experiment/tier1 branch (LMCache+NIXL) for the TP=1 PP=2 variant.
# exit 2
```

함수는 코드 형태로 남아있지만 호출 시 즉시 종료. PP=2 변형이 필요하면 `experiment/tier1` 브랜치에서 LMCache+NIXL 경로로 측정.

---

## 8. 결과 회수 + 분석

```bash
# S3에서 로컬로 다운로드
aws s3 sync s3://hdjung-disaggregation-result/raw/official/$(date +%Y%m%d)/ ./data/

# 표 출력
python disagg-exp/analyze_official.py --log-dir ./data --configs A1 A2 A3 B C1 D --plot

# 커스텀 브랜치 결과와 cross-check
python disagg-exp/analyze_official.py --log-dir ./data --configs C1 D \
    --compare-custom /path/to/custom_branch_results/
```

`--plot`은 `(prefill, decode)` 조합별로 TTFT/TPOT/$/M-tokens 차트를 `plots_official/` 하위에 PNG로 저장.

---

## 9. 로그 구조

```
$EXP_LOG_DIR/
├── <config>/
│   ├── p512_d512_r1.0.json       # vllm bench serve 결과 (rich)
│   ├── p512_d512_r1.0.log        # subprocess stdout/stderr
│   ├── .done_p512_d512_r1.0      # resume 마커
│   └── .failed_p..._r4.0         # 실패 마커 (지우면 retry)
├── nvidia_smi.csv                # 1Hz
├── ifstat.csv                    # 1Hz
├── dcgm.log                      # 2s
├── clock_baseline_<host>.txt     # chrony snapshot
├── vllm_configC1_{prefill,decode}_<host>.log
├── pd_proxy_<host>.log
└── .pid_{nvidia_dmon,ifstat,dcgm_loop}
```

S3 sync 경로: `s3://hdjung-disaggregation-result/raw/official/{YYYYMMDD}/{hostname}/{config}/`

(커스텀 브랜치는 `raw/custom/...`, 이 브랜치는 `raw/official/...` — 안 섞임.)
