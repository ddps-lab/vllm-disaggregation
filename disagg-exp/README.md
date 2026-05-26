# Disagg-Exp Tier-1 — vLLM 공식 도구 기반 PD 분리 실험

> Branch: **`experiment/tier1-vllm-benchmark`** — vLLM upstream의 공식 도구만 사용.
> 커스텀 connector / 커스텀 클라이언트 / 커스텀 분석 모두 제거. 위험 surface 최소화.

vLLM fork (`releases/v0.21.0`) 기반의 prefill/decode disaggregation 비용대비가치 평가.
모델 `Qwen/Qwen2.5-3B-Instruct` (FP16 / `--dtype half`, weights ~6GB, 양자화 미사용), 리전 us-west-2.

> **모델 변경 이력 (2026-05)**: 기존 `meta-llama/Llama-3.1-8B-Instruct-AWQ-INT4` → `Qwen/Qwen2.5-3B-Instruct`.
> 양자화 dequant 오버헤드/노이즈 제거, T4(16GB)/L4(24GB) 모두 native FP16 동작,
> Llama-3.2-3B 와 함께 3B 급 표준 베이스라인이라 분석 비교 용이.

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
- 3B FP16 (~6GB) 은 T4 16GB / L4 24GB 단독 모두 가능. TP/PP 비교는 모델 size
  제약이 아니라 "여러 GPU 활용 vs 단일 GPU" 자체를 측정하는 게 목적.

### 2.3 P2pNcclConnector — 어떻게 작동하나

[vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py](../vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py) 참고.

1. **Control plane** (ZMQ): prefill ↔ decode가 `kv_port`로 ZMQ DEALER/ROUTER 연결. NCCL communicator의 `unique_id` 교환.
2. **Data plane** (NCCL): `ncclSend` / `ncclRecv`로 KV tensor 직접 전송.
3. **버퍼**: `TensorMemoryPool`이 pinned **host RAM** 위에 풀 잡음 (기본 32GB). cross-node는 GPU↔host↔NIC로 흐름.
4. **NCCL transport 자동 선택**: NVLink → PCIe P2P → SHM → IB → TCP sockets. AWS는 보통 **PCIe/SHM** (same-node) 또는 **TCP** (cross-node)로 떨어짐.

> ⚠️ **알아둬야 할 vLLM v1 버그 — `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` 필수**
>
> P2pNcclConnector 는 KV tensor 를 `request_id + "#" + layer_name` 키로 식별합니다.
> 그런데 vLLM v1 의 [input_processor.py:assign_request_id](../vllm/v1/engine/input_processor.py)
> 가 외부 request_id 에 **8자리 random hex** 를 붙입니다. 이게 Prefill / Decode
> 각 인스턴스에서 **독립적으로** 호출되어서 같은 외부 요청에 서로 다른 hash 가 생성됨
> → tensor_id 불일치 → Decode 가 `recv_tensor()` 에서 영원히 hang.
>
> Proxy 가 이미 uuid 로 unique request_id 를 만들어주니까 randomization 불필요.
> `launch_configs.sh` 가 양쪽 노드에 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` 자동 export.

### 2.4 변인 통제 및 결과 오염 방지 (Anti-Contamination)

1. **캐시 오염 방지 (`--no-enable-prefix-caching`)** — 같은 프롬프트 재사용 방지. 모든 요청이 정직하게 전체 prefill + KV 전송.
2. **`PYTHONHASHSEED=123`** — cross-node에서 직렬화 순서 동기화.
3. **`--no-enable-chunked-prefill` (Prefill + Decode 양쪽)** — Prefill 은 한 번에 prefill 해서 compute bound 측정. Decode 는 disagg 에서 실제로 prefill 안 하기 때문에 chunked prefill 옵션이 무의미 → 설정 비대칭 제거 목적으로 양쪽 모두 OFF.
4. **CUDA Graph OFF (기본 `--enforce-eager`)** — P2pNcclConnector send/recv 와 graph capture 의 상호작용 가능성 제거. xpyd 업스트림 예제도 enforce-eager 사용. 필요 시 `ENFORCE_EAGER=0` env 로 graph 재활성화 가능.
5. **`--max-num-seqs 512`** — 배치 제한을 풀어서 throughput 최대.
6. **양자화 미사용** — 3B FP16 weights 가 T4/L4 모두에 native 로 들어가서 AWQ 등 dequant 단계 불필요. 양자화 노이즈 제거.
7. **Greedy decoding 강제 (`temperature=0`, `top_p=1.0`)** — `vllm bench serve` 가 `--extra-body` 로 전달. 모델 `generation_config.json` 의 `temperature=0.7` 등 기본값을 override 해서 deterministic / reproducible 결과 보장.
8. **`VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`** — P2pNcclConnector 의 tensor_id 일치를 위해 필수. 자세한 내용은 §2.3 의 warning 참고.

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
| 5b. 시스템 도구 | `ifstat`, `s5cmd`, `nvtop` |
| 6. chrony | 존재 여부 확인만 (baseline 스냅샷은 launch_configs.sh 가 실험 단위로 생성) |
| 7. DCGM exporter | Prometheus `:9400` 시작 (이미 도는 경우 skip) |
| 9. 검증 | `import vllm, lmcache` |

> **변경 (2026-05)**: 시스템 metric collector (`nvidia-smi dmon`, `ifstat`, DCGM scrape)
> 는 setup.sh 가 시작하지 않습니다. 실험 단위로 깨끗한 `$RUN_DIR/system_logs/` 폴더에
> 출력되도록 **`launch_configs.sh` 가 vllm serve 시작과 동시에 collector 도 시작**
> 합니다. PID 파일은 `$RUN_DIR/.pid_*`.

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
--no-enable-prefix-caching --dtype half --served-model-name qwen2.5-3b
--max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL
--max-num-seqs ${MAX_NUM_SEQS:-512}
```
**CUDA Graph OFF (기본)** — `--enforce-eager` 자동 추가. P2pNccl 안정성을 위해
기본 OFF, `ENFORCE_EAGER=0 bash launch_configs.sh ...` 로 재활성화 가능.
**`--dtype half` 강제** — T4 GPU 가 BF16 미지원이라 동일 launch script 로
T4/L4 모두 돌리기 위해 FP16 으로 고정. 양자화는 사용하지 않음.

`launch_configs.sh` 가 시작 시 export 하는 env:
```
PYTHONHASHSEED=123                          # cross-node 직렬화 결정성
VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1     # tensor_id 일치 (§2.3 참고)
NCCL_IB_DISABLE=1, NCCL_SOCKET_IFNAME=...   # NCCL transport 강제
```

#### 4.2.1 실험 단위 폴더 구조 (RUN_DIR)
`launch_configs.sh` 가 호출되면 자동으로 다음 폴더를 만들고 vllm 로그·proxy 로그·
ASAN 로그·sweep 결과는 `results/`, 시스템 collector 출력은 `system_logs/` 에 떨어집니다.

```
$EXP_LOG_DIR/{CONFIG}-{SERVED_MODEL_NAME}/      ← 예: ~/exp-logs/D-qwen2.5-3b/
├── system_logs/
│   ├── nvidia_smi.csv          (1Hz)
│   ├── ifstat.csv              (1Hz)
│   ├── dcgm.log                (2s scrape)
│   └── clock_baseline_*.txt    (chrony 스냅샷)
└── results/
    ├── vllm_configD_prefill_<host>.log     (Prefill 노드만)
    ├── vllm_configD_decode_<host>.log      (Decode 노드만)
    ├── pd_proxy_<host>.log                 (proxy 띄운 노드만)
    ├── asan_prefill*.log
    ├── p2048_d64_r1.0.json                 (sweep 결과)
    ├── p2048_d64_r1.0.log
    ├── p2048_d64_r1.0.metrics.csv
    ├── p2048_d64_r1.0.metrics.json
    └── .done_p2048_d64_r1.0
```
collector 와 s3 sync daemon 의 PID 는 `$RUN_DIR/.pid_*` 에 저장되고 launch
스크립트가 종료 (Ctrl+C 또는 vllm exit) 되면 trap 으로 자동 정리됩니다.

#### 4.2.2 S3 sync (양쪽 노드 자동)
`launch_configs.sh` 가 collector 와 함께 background `s5cmd sync` daemon 도
시작합니다. **Prefill / Decode 양쪽 노드** 각자 자기 `$RUN_DIR/` 을 30초마다
S3 로 push:

```
s3://$S3_BUCKET/raw/official/{RUN_TAG}/{VLLM_HOST_IP}/{CONFIG}-{MODEL}/
├── system_logs/   ← 그대로 mirror
└── results/       ← 그대로 mirror
```

- `S3_BUCKET=""` 으로 export 하면 sync 비활성화
- `s5cmd` 미설치 시 자동 skip (경고만)

**Port plan**:
- Prefill HTTP: 8100, kv_port: 14600 (+rank)
- Decode HTTP : 8200, kv_port: 14700 (+rank)
- Proxy HTTP : 8000, ZMQ control: 30001

### 4.3 `sweep_official.py` — 워크로드 그리드 자동화

각 (prefill_len, decode_len, rate) 포인트마다 다음을 실행:

```
vllm bench serve \
  --backend openai --base-url <url> \
  --endpoint /v1/completions --model qwen2.5-3b \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --dataset-name random --random-input-len $PL --random-output-len $DL \
  --random-range-ratio 0.0 \
  --num-prompts 300 --request-rate $RATE --burstiness 1.0 \
  --ignore-eos --seed 0 \
  --extra-body '{"temperature": 0, "top_p": 1.0}' \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --save-detailed \
  --result-dir $EXP_LOG_DIR/<config>/ \
  --result-filename p{prefill}_d{decode}_r{rate}.json \
  --metadata config=<C> prefill_len=<PL> decode_len=<DL> rate=<R> point_id=<id> \
              model_path=Qwen/Qwen2.5-3B-Instruct dtype=half
```
*(주의: Proxy의 `max_tokens=1` 정책 충돌 우회 및 400 Bad Request 에러를 방지하기 위해 payload에서 `min_tokens` 주입을 제거하였습니다. 또한, Proxy 서버에 `/health` 엔드포인트가 없는 점을 고려하여 404/405 응답도 서버 활성화로 간주하도록 헬스체크가 수정되었습니다.)*

`--metadata` 에 `model_path`, `dtype` 을 박아 vllm bench serve 의 `model_id`
(=served-model-name) 가 구분 못 하는 양자화/dtype 변형까지 결과 JSON 에 기록.
이 값들은 `sweep_official.py` 상단의 `MODEL_NAME` / `MODEL_PATH` / `MODEL_DTYPE`
**모듈 상수** 가 단일 진실원 (env 의존 X).

Resume: `.done_<point>` / `.failed_<point>` 마커. S3 sync 백그라운드 (`raw/official/{date}/{host}/{config}/`).

### 4.4 `sweep_official.py` — per-side `/metrics` 스크래퍼

`vllm bench serve` 의 결과 JSON 은 **proxy 가 본 end-to-end** (`request_throughput`,
`output_throughput`, TTFT/TPOT/ITL/E2EL) 만 제공합니다. Disaggregated setup 에서는
"Prefill 노드 / Decode 노드 각각이 무슨 일을 했는지" 가 안 보임.

이를 보완하기 위해 sweep 이 각 point 가 도는 동안 **양쪽 vLLM 인스턴스의
Prometheus `/metrics` 엔드포인트** 를 1초 주기로 polling 합니다.

**캡쳐 metric**:

| 종류 | metric | 의미 |
|---|---|---|
| Gauge | `vllm:num_requests_running` | 현재 처리 중인 요청 수 (= 배치 크기) |
| Gauge | `vllm:num_requests_waiting` | 큐 대기 중인 요청 수 |
| Gauge | `vllm:kv_cache_usage_perc` | GPU KV cache 사용률 |
| Counter | `vllm:prompt_tokens_total` | 누적 prompt 토큰 |
| Counter | `vllm:generation_tokens_total` | 누적 생성 토큰 |
| Counter | `vllm:request_success_total` | 누적 성공 요청 수 |

**Rate 계산 원리**: Counter 는 monotonic 누적값이므로 active 구간
(prefill+decode `running > 0`) 의 첫·마지막 sample 차분을 duration 으로 나눠
per-second rate 산출. 두 노드의 `/metrics` 가 독립이라 **자동으로 per-side**
RPS / prefill_TPS / decode_TPS 가 분리되어 나옴.

**출력 파일 (per point, S3 자동 sync)**:
- `{point}.metrics.csv` — 1초 sampling 시계열 raw (prefill + decode 컬럼 분리)
- `{point}.metrics.json` — summary. `_derived` 키에 `prefill_rps`, `prefill_prompt_tps`, `decode_rps`, `decode_generation_tps` 정리

**stdout 출력** (매 point 끝날 때):
```
[1/9] p2048_d64_r1.0
  completed=300 rps=0.97 tok/s=62.4 ttft_p50_ms=85.10 ...                    ← end-to-end
  thru  — prefill rps=0.97 prompt_tps=1985.40 | decode rps=0.97 gen_tps=62.32 ← per-side
  batch — prefill mean=1.8 p99=4.0 max=5 | decode mean=8.2 p99=14.0 max=15
```

**CLI 옵션**:
```
--prefill-metrics-url   default: http://127.0.0.1:8100/metrics
--decode-metrics-url    default: ""  (빈 값 → decode 측 스크래핑 비활성화)
--metrics-interval      default: 1.0 (초)
```
Cross-node config D 의 경우 sweep 이 Prefill 노드에서 도는 가정하에
`--decode-metrics-url http://<decode-private-ip>:8200/metrics` 를 반드시 전달.

### 4.5 `analyze_official.py`

`{EXP_LOG_DIR}/{config}/p*_d*_r*.json`을 읽어 표 + matplotlib plot 출력.

- **`--skip-warmup 50`** (기본): `--save-detailed`의 per-request `ttfts[]` 배열에서 앞 50개 버리고 percentile 재계산.
- **`--skip-warmup 0`**: 벤치마크의 자체 집계 그대로 사용.
- **`--compare-custom <dir>`**: `experiment/tier1` 브랜치의 `sweep.py` JSONL 결과와 side-by-side 비교 (교차 검증용).

---

## 5. 워크로드 그리드

```python
PD_PAIRS = [
    (2048, 64),
    (512, 512),
    (128, 1024)
]
RATES = [1.0, 2.0, 4.0]

Total: 9 points/config (3 pairs × 3 rates)
num_prompts = 300 (warmup 10 + measured 290, analyze에서 앞 10 skip)
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

공식 xpyd 예제가 `--enforce-eager` 를 명시하고, P2pNcclConnector send/recv 와
graph capture 의 상호작용 가능성 우려가 있어 **기본 OFF (`--enforce-eager` ON)** 으로 운영.
`launch_configs.sh` 의 `ENFORCE_EAGER` 기본값이 1 이라 그냥 `bash launch_configs.sh ...`
하면 자동 적용. CUDA Graph 캡처 재활성화는:
```bash
ENFORCE_EAGER=0 bash launch_configs.sh configC1 prefill
```
페널티는 per-step kernel launch overhead 만큼 (~5~15%) 이지만 양쪽 노드 동일하게
적용되므로 P/D split 비교 자체에는 영향 X.

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

> **RUN_TAG 사용법 (한 줄)**: 새 실험 시작할 때 양쪽 노드에서
> `export RUN_TAG=20260526-1530-baseline` 같이 **동일한 값**을 export. 그 뒤로는
> 평소 명령어 그대로 — 모든 컴포넌트 (launch_configs.sh, sweep_official.py) 가
> RUN_TAG env 를 자동 인식. 미지정 시 `YYYYMMDD-HHMM` 분 단위 자동 생성 (양쪽
> 노드가 같은 분에 띄우면 합쳐짐).

**Decode node:**
```bash
# (실험마다 한 번) 실험 식별자 — Prefill 과 동일하게
export RUN_TAG=20260526-1530-baseline

export VLLM_HOST_IP=10.0.x.y       # 이 노드의 private IP
export PROXY_IP=10.0.x.z           # ← prefill/proxy 노드의 private IP (필수)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens5     # 실제 NIC 이름으로

bash disagg-exp/launch_configs.sh configD decode
# 시작 시 stdout 에서 확인:
#   [launch] RUN_TAG=20260526-1530-baseline
#   [launch] S3 dest=s3://.../raw/official/20260526-1530-baseline/10.0.x.y/D-qwen2.5-3b/
```

**Prefill node:**
```bash
# (실험마다 한 번) Decode 와 동일한 값
export RUN_TAG=20260526-1530-baseline

export VLLM_HOST_IP=10.0.x.z       # 이 노드의 private IP
export DECODER_HOST=10.0.x.y       # decode 노드의 private IP
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens5

bash disagg-exp/launch_configs.sh configD prefill
```

**Prefill node (또 다른 터미널, 둘 다 ready 후):**
```bash
# RUN_TAG 이 새 셸엔 없을 수 있으니 다시 export (또는 .bashrc 등록)
export RUN_TAG=20260526-1530-baseline

# Proxy 띄우기 (이 셸도 자기 RUN_DIR 에 s3 sync 시작)
bash disagg-exp/launch_configs.sh configD proxy
```

**Prefill node (또또 다른 터미널, sweep):**
```bash
export RUN_TAG=20260526-1530-baseline
export VLLM_HOST_IP=10.0.x.z

# Sweep — Decode 노드의 /metrics URL 을 반드시 전달 (per-side 측정용)
.venv/bin/python disagg-exp/sweep_official.py \
  --config D \
  --base-url http://127.0.0.1:8000 \
  --decode-metrics-url http://10.0.x.y:8200/metrics
# 또는 명시적으로 --run-tag 전달:
#   --run-tag 20260526-1530-baseline
```

> ⚠️ `--decode-metrics-url` 을 안 주면 `{point}.metrics.json` 의
> `_derived.decode_rps`, `_derived.decode_generation_tps` 가 **null** 로 떨어집니다.
> 사전 검증: `curl -s http://10.0.x.y:8200/metrics | grep -E "^vllm:num_requests_running"`
> 가 라인을 반환해야 함 (안 나오면 AWS SG 에 8200/tcp 인바운드 필요).

> 💡 **새 실험 돌릴 때마다 RUN_TAG 만 바꾸면 됨**: `export RUN_TAG=20260526-1700-exp02-bigbatch`
> 양쪽 노드에서 동일하게 export 한 다음 launch 다시 시작. 결과는 자동으로 다른
> S3 폴더로 분리됨.

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
aws s3 sync s3://hdjung-disaggregation-result/raw/official/{RUN_TAG}/ ./data/{RUN_TAG}/
# 예: aws s3 sync s3://hdjung-disaggregation-result/raw/official/20260526-1530-baseline/ ./data/20260526-1530-baseline/

# 표 출력
python disagg-exp/analyze_official.py --log-dir ./data --configs A1 A2 A3 B C1 D --plot

# 커스텀 브랜치 결과와 cross-check
python disagg-exp/analyze_official.py --log-dir ./data --configs C1 D \
    --compare-custom /path/to/custom_branch_results/
```

`--plot`은 `(prefill, decode)` 조합별로 TTFT/TPOT/$/M-tokens 차트를 `plots_official/` 하위에 PNG로 저장.

---

## 9. 로그 구조

### 9.1 로컬 (각 노드)
```
$EXP_LOG_DIR/
└── D-qwen2.5-3b/                              ← {CONFIG}-{SERVED_MODEL_NAME}
    ├── system_logs/
    │   ├── nvidia_smi.csv                     # 1Hz
    │   ├── ifstat.csv                         # 1Hz
    │   ├── dcgm.log                           # 2s scrape
    │   └── clock_baseline_<host>.txt          # chrony snapshot
    ├── results/
    │   ├── vllm_configD_{prefill|decode}_<host>.log
    │   ├── pd_proxy_<host>.log                # proxy 띄운 노드만
    │   ├── asan_{prefill|decode}*.log         # ASAN 로그
    │   ├── p2048_d64_r1.0.json                # vllm bench serve 결과 (end-to-end)
    │   ├── p2048_d64_r1.0.log                 # bench subprocess stdout
    │   ├── p2048_d64_r1.0.metrics.csv         # /metrics scraper 시계열
    │   ├── p2048_d64_r1.0.metrics.json        # per-side summary (_derived RPS/TPS)
    │   └── .done_p2048_d64_r1.0               # resume 마커
    └── .pid_*                                  # collector / s3-sync PIDs (자동 정리됨)
```

### 9.2 S3 (양쪽 노드의 RUN_DIR 이 같은 RUN_TAG 폴더로 합쳐짐)
```
s3://hdjung-disaggregation-result/raw/official/
└── {RUN_TAG}/                                  ← 예: 20260526-1530-baseline
    ├── 172.31.49.208/                          ← Prefill 노드 (VLLM_HOST_IP)
    │   └── D-qwen2.5-3b/
    │       ├── system_logs/                    ← Prefill 노드의 시스템 로그
    │       └── results/                        ← Prefill 서버 로그 + proxy 로그 + sweep 결과 전부
    └── 172.31.48.200/                          ← Decode 노드 (VLLM_HOST_IP)
        └── D-qwen2.5-3b/
            ├── system_logs/                    ← Decode 노드의 시스템 로그
            └── results/                        ← Decode 서버 로그만
```

- **`{RUN_TAG}` 기본값**: `$(date +%Y%m%d-%H%M)` (분 단위). 양쪽 노드를 같은 분에
  띄우면 자동으로 같은 폴더에 합쳐짐. 다른 분이면 폴더 두 개로 갈라짐 — 사후에
  사람이 합치거나, 명시적으로 `export RUN_TAG=...` 동일하게 설정 권장.
- **per-node 자체 sync**: 각 노드의 `launch_configs.sh` 가 자기 `$RUN_DIR/` 을
  30초마다 S3 로 push (양쪽 노드 모두 `s5cmd` 설치 필요).
- **sweep 결과** (`p*.json`, `.metrics.*`) 는 sweep_official.py 가 도는 노드
  (보통 Prefill) 의 `results/` 에만 들어감. Decode 의 `results/` 엔 자기 vllm log
  만 있음.

S3 sync 경로: `s3://hdjung-disaggregation-result/raw/official/{RUN_TAG}/{VLLM_HOST_IP}/{CONFIG}-{MODEL}/{system_logs|results}/`

(커스텀 브랜치는 `raw/custom/...`, 이 브랜치는 `raw/official/...` — 안 섞임.)

---

## 10. KV 캐시 통신 오버헤드 로깅

`p2p_nccl_connector.py`에는 큐(Queue) 대기 시간을 포함한 시스템 전체의 엔드투엔드(End-to-End) KV 캐시 전송 지연 시간을 측정하도록 로깅이 추가되어 있습니다. 이 시간은 순수 네트워크 통신 뿐만 아니라 버퍼 동기화와 큐 대기로 인해 소비되는 전체 파이썬 오버헤드를 포함합니다.

- **Prefill (송신)**: `큐 대기 포함 전체 Send 시간: OOO.OO ms`
- **Decode (수신)**: `대기 시간 포함 전체 Recv 시간: OOO.OO ms`

결과 폴더 내의 `prefill_stdout.log` 및 `decode_stdout.log`를 통해 요청당 오버헤드 병목 현상을 정확히 파악할 수 있습니다.

---

## 11. 변경 이력 (최근 작업)

| 커밋 | 내용 |
|---|---|
| (이번 변경) | **S3 폴더 구조 재설계** — `raw/official/{RUN_TAG}/{ip}/{config}-{model}/{system_logs,results}/`. Decode 노드도 background s5cmd sync 시작 → Decode 서버 log 도 S3 에 자동 업로드. setup.sh 의 collector 시작은 launch_configs.sh 로 이동 (실험 단위로 system_logs 분리). `RUN_TAG` env 도입 (기본 `YYYYMMDD-HHMM`). |
| `0f7eb19ab` | **CUDA Graph 기본 비활성화** — P2pNcclConnector 안정성 우선. `ENFORCE_EAGER` 기본 1 |
| `3c698c672` | **sweep 메트릭에 per-side RPS/TPS 추가** — `/metrics` scraper 도입. `{point}.metrics.csv` / `{point}.metrics.json` 자동 생성. `vllm:gpu_cache_usage_perc` → `vllm:kv_cache_usage_perc` 버그 fix |
| `054d1ee70` | **모델을 Qwen2.5-3B-Instruct 로 변경** — 양자화(AWQ) 제거. `MODEL_NAME` / `MODEL_PATH` / `MODEL_DTYPE` 를 sweep_official.py 모듈 상수로 통합. PD_PAIRS: `(2048,128)/(1024,512)/(128,2048)` → `(2048,64)/(512,512)/(128,1024)` |
| `bb6f24166` | 결과 JSON metadata 에 `model_path`/`dtype` 추가. `RATES` 그리드 변경 |
| `10c093308` | **Decode hang 해결** (request_id randomization 버그). `--extra-body '{"temperature": 0}'` 로 greedy decoding 강제 |
| `18224a753` | `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` 자동 export 추가 |
| `90b5cfe2a` | configD decode 가 self-IP 로 fallback 하던 문제 → `PROXY_IP` 명시 강제 |

### 알려진 함정 (debugging 시 우선 확인)

1. **Decode 가 첫 요청에서 영원히 hang** → §2.3 의 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` warning 참고. `launch_configs.sh` 가 자동 export 하지만 직접 띄울 때는 명시 필요.
2. **`{point}.metrics.json` 의 `decode_rps`/`decode_generation_tps` 가 `null`** → sweep 돌릴 때 `--decode-metrics-url` 누락. §7.4 참고.
3. **NCCL handshake 후 hang** → CUDA Graph 잔재. `ENFORCE_EAGER=1` (기본값) 확실히 들어갔는지 vllm 로그 첫 줄에서 `enforce_eager=True` 확인.
4. **결과 JSON 의 `model_id="qwen2.5-3b"` 만 보고 모델 식별 어려움** → metadata 의 `model_path`, `dtype` 같이 확인.
5. **S3 에 Prefill / Decode 폴더가 다른 RUN_TAG 로 갈라짐** → 양쪽 노드에서 export RUN_TAG 안 하고 default (분 단위 timestamp) 가 어긋난 경우. 둘 다 같은 분에 띄우지 못했다면 사후에 사람이 폴더 합치거나, 다음 실험 시 RUN_TAG 명시.
6. **Decode 노드 vllm 서버 log 가 S3 에 없음** → Decode 노드에 s5cmd 미설치이거나 `S3_BUCKET=""` 으로 비활성화된 경우. setup.sh 가 s5cmd 자동 설치하므로 보통 자동 동작. 확인: Decode 노드에서 `command -v s5cmd && echo OK`.
