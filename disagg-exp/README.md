# Disagg-Exp Tier-1 — Prefill/Decode 분리 실험

vLLM fork (`releases/v0.21.0`) 기반의 prefill/decode disaggregation 비용대비가치 평가 실험.
모델은 `meta-llama/Llama-3.1-8B-Instruct` (BF16, ~16GB), 리전은 us-west-2.

> **핵심 질문**: 같은 예산이면 큰 GPU 한 장이 나은가, 작은 GPU 여러 장 + PD 분리가 나은가?
>
> 같은 워크로드를 7가지 토폴로지에 던지고 **TTFT / TPOT / $/M-tokens**로 비교한다.

---

## 1. 개념 정리 (왜 이렇게 짰는가)

### 1.1 Prefill vs Decode

LLM 추론은 두 단계로 나뉜다:

- **Prefill**: 입력 프롬프트 전체를 한 번에 통과시켜 KV cache를 만든다. 계산 집약적(compute-bound), GPU 활용도 높음.
- **Decode**: 토큰을 하나씩 생성한다. 메모리 대역폭 집약적(memory-bound), GPU 한 번에 별로 못 씀.

**Monolithic**: 두 단계를 같은 인스턴스에서 처리 (배치를 섞음).
**PD Disaggregation**: prefill 전용 노드와 decode 전용 노드를 분리 → 각 단계를 최적 GPU에 배치 가능.

### 1.2 Tensor Parallel (TP) vs Pipeline Parallel (PP)

- **TP=N**: 모델의 한 레이어를 N개 GPU가 분담 (행렬 곱을 행/열로 자름). 통신 빈도 ↑, latency ↓.
- **PP=N**: 모델 레이어를 N개 GPU에 순차 배치 (1~10번 레이어 GPU0, 11~20 GPU1, …). 통신 빈도 ↓, throughput ↑.
- **TP × PP = 총 GPU 수**. 8B 모델은 T4 16GB 단독에 안 올라가서 **TP ≥ 2**가 필수.

### 1.3 NIXL / LMCache / UCX

PD 분리 시 prefill이 만든 KV를 decode에게 전달해야 한다. 이 통로가 **NIXL** (NVIDIA의 GPU-to-GPU 전송 라이브러리, **UCX** 위에서 동작).

- **UCX_TLS**: 전송 계층 선택. same-node는 `cuda_copy,shm`, cross-node는 `tcp`.
- **LMCache**: KV 캐시 라이프사이클 관리 라이브러리. vLLM과 NIXL을 연결.
- 본 실험은 LMCache 0.3.9+ 사용. `LMCacheConnectorV1`이 vLLM과 LMCache의 어댑터.

### 1.4 워크로드 측정 — 2-phase warmup/measured

각 sweep point는 두 단계:

1. **warmup** (50 요청): 시스템 warm-up + overload 판정용. `fail_rate > 0.3` 또는 `ttft_p99 > 30s`면 measured phase **skip**, `.failed_*` 마커 작성.
2. **measured** (200 요청): 본 측정. JSONL에 모두 기록.

이게 없으면 overload된 point가 통계를 오염시킨다.

### 1.5 Throughput 정의 (2가지 동시 기록)

- `thr_tok_s_e2e = tokens / (max(recv) − min(send))` — **service window**. overload에서 long-tail이 분모를 부풀려 작게 나옴.
- `thr_tok_s_send = tokens / (max(send) − min(send))` — **arrival window**. 실제 처리 rate에 가까움.
- `achieved_rate = N_ok / arrival_window` — 요청 rate 대비 실제 처리 → saturation 감지.

### 1.5 변인 통제 및 결과 오염 방지 장치 (Anti-Contamination Mechanisms)

본 벤치마크는 인프라 수준부터 어플리케이션 수준까지, 실험 결과(Latency, Throughput)가 외부 요인에 의해 오염되는 것을 막기 위해 다음과 같은 엄격한 변인 통제 장치를 적용했습니다.

1. **캐시 오염 방지 (`--no-enable-prefix-caching`)**
   - **이유**: 동일한 프롬프트 요청 시 vLLM이 기존 KV 캐시를 재사용하여 연산을 건너뛰는 것을 방지합니다. 
   - **효과**: 모든 요청이 100% 정직하게 전체 Prefill 연산과 KV 전송을 거치도록 강제하여, "순수 처리량"만을 정확히 측정합니다.

2. **파이썬 해시 및 난수 시드 고정 (`PYTHONHASHSEED=123`)**
   - **이유**: Prefill 노드와 Decode 노드가 물리적으로 분리된 환경(Config D)에서, 객체의 직렬화(Serialization) 순서나 내부 랜덤성이 엇갈리면 통신 레이턴시가 튀거나 패킷이 꼬일 수 있습니다.
   - **효과**: 두 서버를 완벽히 동일한 난수/해시 환경으로 동기화하여 통신 병목 변수를 제거합니다.

3. **JIT 컴파일 딜레이 격리 (2-Phase Warm-up)**
   - **이유**: PyTorch/CUDA 프로그램 특성상 처음 1~10개의 요청은 메모리 할당 및 커널 JIT 컴파일 때문에 비정상적으로 느립니다. (수십 초 소요)
   - **효과**: `sweep.py`는 실제 측정(Measured phase) 전에 50개의 가짜 트래픽(Warm-up phase)을 먼저 쏴서 파이프라인을 완전히 데우고(Warming), 그 이후의 200개 데이터만 통계에 반영합니다.

4. **로깅 I/O 병목 방지 (TP Rank 0 Filtering)**
   - **이유**: Tensor Parallel(TP=2, 4) 환경에서는 GPU마다 똑같은 워커 프로세스가 뜹니다. 모든 워커가 동시에 `.jsonl` 로그 파일에 접근해 기록하려고 하면 하드디스크 I/O 경합이 발생하여 KV 전송 시간(duration_ms)이 인위적으로 늘어납니다.
   - **효과**: `instrumented_connector.py`에서 `if rank == 0:` 조건을 걸어 오직 대장 프로세스 하나만 디스크에 접근하도록 막았습니다.

5. **좀비 프로세스 청소 (`setup.sh`의 `_kill_pid_file`)**
   - **이유**: 이전 실험에서 죽지 않고 백그라운드에 남아있는 vLLM이나 모니터링 툴(DCGM)이 다음 실험의 GPU 메모리/대역폭을 몰래 갉아먹는 것을 막습니다.
   - **효과**: 실험을 재시작할 때마다(Idempotent) 이전 상태를 완전히 초기화(Clean State)하여 독립된 실험 환경을 보장합니다.

---

## 2. 7가지 Config

| Config | Instance | GPU 구성 | Topology | $/hr OD |
|---|---|---|---|---:|
| **A1** | g4dn.12xlarge | 4× T4 16GB | Monolithic **TP=2 PP=2** | $3.91 |
| **A2** | g4dn.12xlarge | 4× T4 16GB | Monolithic **TP=4 PP=1** | $3.91 |
| **A3** | g4dn.12xlarge | 4× T4 16GB | Monolithic **TP=1 PP=4** | $3.91 |
| **B**  | g6e.xlarge | 1× L40S 48GB | Monolithic TP=1 | $1.86 |
| **C1** | g4dn.12xlarge | 4× T4 16GB | Same-node PD **TP=2 PP=1**: prefill GPU 0-1 + decode GPU 2-3. NIXL via shm | $3.91 |
| **C2** | g4dn.12xlarge | 4× T4 16GB | Same-node PD **TP=1 PP=2**: prefill GPU 0-1 + decode GPU 2-3. NIXL via shm | $3.91 |
| **D**  | 2× g6.xlarge | 2× L4 24GB | Cross-node PD: prefill TP=1 + decode TP=1. NIXL via TCP | $1.61 |

### 비교 의미

- **A1↔A2↔A3**: 동일 하드웨어에서 TP/PP 조합만 다름 → Monolithic의 최적 병렬화 전략 탐색.
- **C1↔C2**: 동일 하드웨어 + 동일 PD 토폴로지에서 TP/PP만 다름 → PD에서의 최적 병렬화 전략 탐색.
- **A↔C**: 동일 4× T4. Monolithic vs same-node PD → 순수 소프트웨어 비교.
- **B↔A**: 큰 GPU 1장 vs 작은 GPU 4장 → 멀티-GPU 통신 오버헤드 평가.
- **C↔D**: 둘 다 PD. T4+shm vs L4+TCP → $/M-tokens 정규화로 비교.
- **B↔D**: 비슷 가격대. 큰 GPU 1장 vs 작은 GPU 2장 + PD.

---

## 3. 파일 구성

```
disagg-exp/
├── setup.sh                       # 노드 부트스트랩
├── launch_configs.sh              # vllm serve 디스패처
├── instrumented_connector.py      # PD prefill KV 전송 시간 기록
├── sweep.py                       # 클라이언트 워크로드 + S3 sync 내장
├── analyze.py                     # JSONL → 표/플롯
└── README.md                      # 이 문서
```

### 3.1 `setup.sh` — 노드 부트스트랩 (idempotent)

재실행 안전. 한 번 돌리면 다음이 갖춰진다:

| 단계 | 내용 |
|---|---|
| 1. `uv` 설치 | Astral의 빠른 Python 패키지 매니저 |
| 2. `.venv` 생성 | Python 3.12 가상환경 (`$REPO/.venv`) |
| 3. vLLM editable install | `VLLM_USE_PRECOMPILED=1 uv pip install -e .` |
| 4. LMCache ≥ 0.3.9 | 버전 체크 후 필요시 설치 |
| 5. Python 의존성 | httpx, fastapi, aiohttp, huggingface_hub, … |
| 5b. 시스템 도구 | `ifstat`, `s5cmd` (apt/wget) |
| 6. chrony | `chronyc tracking` 스냅샷 저장 |
| 7. DCGM exporter | Prometheus endpoint `:9400` 기동 (DLAMI에 이미 설치돼있다고 가정) |
| 8. Metric collector 3종 | `nvidia-smi dmon` (1Hz), `ifstat` (1Hz), DCGM scrape loop (2s). 각각 PID 파일로 관리 |
| 9. 검증 | `import vllm, lmcache` 버전 출력 |

PID 파일: `$EXP_LOG_DIR/.pid_{nvidia_dmon,ifstat,dcgm_loop}` → 재실행 시 안전하게 kill 후 재시작.

### 3.2 `launch_configs.sh` — `vllm serve` 디스패처

각 config + role을 명령행 한 줄로 실행. 환경변수와 `--kv-transfer-config` JSON을 자동 조립한다.

| 함수 | 역할 |
|---|---|
| `configA1()` | TP=2 PP=2, 4× T4, port 8000 |
| `configA2()` | TP=4 PP=1, 4× T4, port 8000 |
| `configA3()` | TP=1 PP=4, 4× T4, port 8000 |
| `configA()` | configA1의 alias (back-compat) |
| `configB()` | TP=1, 1× L40S, port 8000 |
| `configC1_prefill()` | TP=2 PP=1 (GPU 0-1), LMCache sender, port 8100. UCX_TLS=cuda_copy,shm,tcp |
| `configC1_decode()` | TP=2 PP=1 (GPU 2-3), LMCache receiver, port 8200 |
| `configC2_prefill()` | TP=1 PP=2 (GPU 0-1), LMCache sender, port 8100. UCX_TLS=cuda_copy,shm,tcp |
| `configC2_decode()` | TP=1 PP=2 (GPU 2-3), LMCache receiver, port 8200 |
| `configC_prefill/decode()` | configC1의 alias (back-compat) |
| `configD_prefill()` | TP=1, LMCache sender, peer=`$DECODER_HOST`. UCX_TLS=tcp |
| `configD_decode()` | TP=1, LMCache receiver, port 8200 |
| `launch_proxy()` | `disagg_proxy_server.py` 실행 (port 8000, prefill 8100, decode 8200 묶음) |

공통 flag (`COMMON_FLAGS`): `--no-enable-prefix-caching --dtype bfloat16 --served-model-name llama-3.1-8b`. CUDA Graph **ON** (enforce-eager 안 씀).

LMCache YAML은 `$LOG_DIR/lmcache_{prefill,decode}_{C,D}.yaml`에 런타임 생성. peer 주소(127.0.0.1 / `$DECODER_HOST`)도 자동 주입.

### 3.3 `instrumented_connector.py` — KV 전송 시간 기록

`LMCacheConnectorV1` 상속. PD prefill role만 사용 (configC1/C2/D prefill).

| 메서드 | 역할 |
|---|---|
| `__init__` | `vllm_config.parallel_config.rank` 저장. 0만 로그 기록 (TP=2 동시 쓰기 방지) |
| `save_kv_layer` | 첫 호출 시점 기록 (`_save_start`). 레이어 카운트 |
| `wait_for_save` | KV 저장 완료 대기 후 duration_ms 측정. decode-only 배치는 skip |
| `get_finished` | 완료된 req_id 기록. lmcache_engine 메서드 introspection (1회) |
| `build_connector_meta` | 스케줄러 측 metadata 타입 로그 (1회) |

출력: `$EXP_LOG_DIR/kv_transfer.jsonl`
```json
{"ts_utc": 1747526400.12, "event": "wait_for_save_done", "duration_ms": 23.5, "layers_saved": 32}
{"ts_utc": 1747526400.30, "event": "send_finished", "req_id": "C_2048_512_4.0_measured_42"}
```

`launch_configs.sh`가 `--kv-transfer-config`에 `"kv_connector_module_path":"instrumented_connector"`를 자동으로 넣어준다. PYTHONPATH에 `disagg-exp/`가 들어가서 import 가능.

### 3.4 `sweep.py` — 클라이언트 워크로드 드라이버

| 영역 | 역할 |
|---|---|
| `_do_request()` | 단일 요청. token-id 직접 전달 (BOS는 서버가 prepend), SSE 스트림 파싱, TTFT/e2e/usage 기록 |
| `run_point()` | 한 sweep point (prefill×decode×rate 조합) 실행. Poisson 도착, 2-phase warmup→measured |
| `wait_for_health()` | sweep 전에 `/health` 200 대기 (vLLM 모델 로드 ~수분) |
| `S3Syncer` | 백그라운드 thread로 30초마다 `$EXP_LOG_DIR/` → S3 sync. atexit로 비정상 종료에도 final sync |
| `main()` | 그리드 빌드 → S3Syncer 시작 → health 대기 → 점별 실행. 마커(.done/.failed)로 resume |

요청 페이로드:
```json
{
  "prompt": [1, 2, ..., N],
  "max_tokens": D, "min_tokens": D, "ignore_eos": true,
  "temperature": 0, "top_p": 1.0,
  "stream": true, "stream_options": {"include_usage": true}
}
```

출력: `$EXP_LOG_DIR/<config>/p{prefill}_d{decode}_r{rate}.jsonl` (요청당 1줄).

### 3.5 `analyze.py` — 사후 분석 (로컬 실행)

| 함수 | 역할 |
|---|---|
| `load_points()` | config 폴더에서 JSONL 모두 읽어 measured phase만 추출 |
| `analyze_point()` | p50/p99 TTFT, p50/p99 TPOT, throughput 2종, achieved_rate 계산 |
| `dollar_per_m_tokens()` | `COST_PER_HR[config] / (thr_e2e × 3600 / 1e6)` |
| `_warn_pt_delta()` | `prompt_tokens` vs `prefill_len` delta > 1이면 경고 |
| `print_table()` | 표 출력 |
| `plot_comparison()` | matplotlib 3-panel: TTFT / TPOT / $/M-tokens vs rate |

---

## 4. 워크로드 그리드

```python
PREFILL_LENS = [512, 2048, 8192]
DECODE_LENS  = [128, 512, 1024, 4096]
RATES        = [1.0, 2.0, 4.0]

Cross 1 (prefill × rate, decode=512):  3 × 3 =  9 points
Cross 2 (decode × rate, prefill=2048): 4 × 3 = 12 points
                  (중복 3 제거) → total = 18 points/config
warmup=50, measured=300 → 350 req/point
```

Per config: **6,300 요청**. 7 configs → 약 **44,100 요청** (총 126 points).

Env로 override 가능:
```bash
SWEEP_PREFILL_LENS=512 SWEEP_DECODE_LENS=128 SWEEP_RATES=1.0 \
SWEEP_WARMUP_N=3 SWEEP_MEASURED_N=5 \
python sweep.py --config A1 --base-url http://localhost:8000
```

---

## 5. 로그 구조

```
$EXP_LOG_DIR/
├── <config>/                       # A1, A2, A3, B, C1, C2, D
│   ├── p512_d512_r1.0.jsonl        sweep per-request data
│   ├── .done_p512_d512_r1.0        완료 마커 (resume용)
│   └── .failed_p512_d512_r4.0      overload 또는 예외
├── kv_transfer.jsonl               C1/C2/D prefill만, instrumented_connector
├── nvidia_smi.csv                  1Hz Power/Util/Memory
├── ifstat.csv                      1Hz NIC throughput
├── dcgm.log                        2s DCGM 메트릭
├── s3_sync.log                     S3Syncer 로그
├── clock_baseline_<host>.txt       chrony snapshot
├── vllm_configA1_<host>.log               Config A1 vLLM 서버
├── vllm_configA2_<host>.log               Config A2
├── vllm_configA3_<host>.log               Config A3
├── vllm_configB_<host>.log                Config B
├── vllm_configC1_{prefill,decode}_<host>.log   Config C1
├── vllm_configC2_{prefill,decode}_<host>.log   Config C2
├── vllm_configD_{prefill,decode}_<host>.log    Config D
├── pd_proxy_<host>.log                         C1/C2/D proxy
├── lmcache_{prefill,decode}_{C1,C2,D}.yaml     자동 생성
└── .pid_{nvidia_dmon,ifstat,dcgm_loop}
```

---

## 6. 실행 순서 (전체 흐름)

### 흐름 한눈에

```
[AWS 사전 준비] → [Step 1~5: AMI builder 인스턴스에서 검증/모델다운/AMI 굽기]
                       ↓
                  [Step 6~7: 각 config별 인스턴스 띄움 + sweep 실행]
                       ↓
                  [Step 8: 결과 회수 + analyze.py]
                       ↓
                  [Step 9: 인스턴스 종료 + 비용 확인]
```

### 6.0 AWS 사전 준비 (한 번만)

- Security Group `hdjung-vllm-try`: All TCP 자기 SG + SSH 22 + 6379 + 8265
- S3 버킷: `hdjung-disaggregation-result`
- IAM Role: `hdjung_disaggregation_result` (EC2 → AmazonS3FullAccess)
- Key pair 준비

### Step 1 — AMI builder 인스턴스 띄우기

EC2 콘솔 → Launch instances:

| 항목 | 값 |
|---|---|
| Name | `disagg-exp-ami-builder` |
| AMI | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04) |
| Instance type | **g4dn.xlarge** (T4 1장, spot ~$0.20/hr) |
| Network | SG = `hdjung-vllm-try`, Auto-assign public IP = Enable |
| Storage | 200 GiB gp3 |
| IAM instance profile | `hdjung_disaggregation_result` |
| Purchasing option | Spot |

### Step 2 — SSH 접속 + 환경 세팅

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<public-ip>

git clone https://github.com/<your>/vllm-disaggregation.git
cd vllm-disaggregation
git checkout experiment/tier1

export EXP_LOG_DIR=/home/ubuntu/exp-logs
bash disagg-exp/setup.sh
source .venv/bin/activate
```

**검증 체크리스트**:
```bash
python -c "import vllm, lmcache; print(vllm.__version__, lmcache.__version__)"
# 기대: 0.21.x  0.3.9+

curl -s localhost:9400/metrics | grep -m1 DCGM_FI_DEV_FB_USED
chronyc tracking | grep "Last offset"
sleep 5 && wc -l $EXP_LOG_DIR/nvidia_smi.csv   # > 0 이면 OK
```

### Step 3 — Smoke test

g4dn.xlarge는 T4 1장 → A1/A2/A3 불가 (4 GPU 필요). TP=1로 최소 동작만:

```bash
# 터미널 1
MAX_MODEL_LEN=2048 GPU_MEM_UTIL=0.90 \
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --served-model-name llama-3.1-8b \
  --max-model-len 2048 --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching --dtype bfloat16 \
  --tensor-parallel-size 1 --port 8000

# 터미널 2 (서버 ready 후)
SWEEP_PREFILL_LENS=512 SWEEP_DECODE_LENS=128 SWEEP_RATES=1.0 \
SWEEP_WARMUP_N=3 SWEEP_MEASURED_N=5 \
python disagg-exp/sweep.py --config test --base-url http://localhost:8000 --s3-bucket ""
```

`$EXP_LOG_DIR/test/p512_d128_r1.0.jsonl`에 `"status":"success"` 있으면 통과.

⚠️ T4 단독에 8B OOM 나면 `MAX_MODEL_LEN=1024` 또는 TinyLlama로 대체.

### Step 4 — 모델 다운로드 + S3 캐시

```bash
huggingface-cli login   # 토큰 입력
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ~/models/llama-3.1-8b

# 다음 인스턴스에서 빠르게 받기용
s5cmd sync ~/models/llama-3.1-8b/ s3://hdjung-disaggregation-result/models/llama-3.1-8b/
```

### Step 5 — AMI 굽기

EC2 콘솔에서:
1. Instances → `disagg-exp-ami-builder` 선택
2. Actions → Image and templates → Create image
3. Image name: `disagg-exp-v0.21-ready`
4. `available` 대기 (~10~20분) 후 AMI ID 메모
5. builder 인스턴스 **Terminate**

### Step 6 — 실제 실험 (각 config별 인스턴스 띄움)

각 인스턴스는 IAM Role `hdjung_disaggregation_result`를 꼭 붙여서 띄울 것 (sweep.py가 자동으로 S3 sync 함).

#### Config A1/A2/A3 — 같은 g4dn.12xlarge에서 순차 실행

```bash
export EXP_LOG_DIR=/home/ubuntu/exp-logs
export MODEL=~/models/llama-3.1-8b   # 또는 HF ID

# ── A1: TP=2 PP=2 ──
bash disagg-exp/launch_configs.sh configA1 &
# "Application startup complete." 기다린 후 (tail -f $EXP_LOG_DIR/vllm_configA1_$(hostname).log)
python disagg-exp/sweep.py --config A1 --base-url http://localhost:8000
kill %1; sleep 5

# ── A2: TP=4 PP=1 ──
bash disagg-exp/launch_configs.sh configA2 &
python disagg-exp/sweep.py --config A2 --base-url http://localhost:8000
kill %1; sleep 5

# ── A3: TP=1 PP=4 ──
bash disagg-exp/launch_configs.sh configA3 &
python disagg-exp/sweep.py --config A3 --base-url http://localhost:8000
kill %1
```

#### Config B — g6e.xlarge (1× L40S)

```bash
bash disagg-exp/launch_configs.sh configB &
python disagg-exp/sweep.py --config B --base-url http://localhost:8000
```

#### Config C1/C2 — g4dn.12xlarge same-node PD (순차 실행)

같은 g4dn.12xlarge 인스턴스에서 C1 끝나면 C2 실행. 둘 다 터미널 3개 필요 (decode → prefill → proxy).

```bash
# ── C1: TP=2 PP=1 ──
# 터미널 1: decode 먼저 (receiver가 sender 연결을 기다리는 구조)
bash disagg-exp/launch_configs.sh configC1 decode

# 터미널 2: prefill
bash disagg-exp/launch_configs.sh configC1 prefill

# 터미널 3: 양쪽 "Application startup complete." 확인 후 proxy
bash disagg-exp/launch_configs.sh configC1 proxy

# 터미널 4: sweep (proxy 통해 port 8000)
python disagg-exp/sweep.py --config C1 --base-url http://localhost:8000

# C1 sweep 끝나면 vllm/proxy 모두 kill 후 C2 실행
# ── C2: TP=1 PP=2 (같은 흐름) ──
bash disagg-exp/launch_configs.sh configC2 decode
bash disagg-exp/launch_configs.sh configC2 prefill
bash disagg-exp/launch_configs.sh configC2 proxy
python disagg-exp/sweep.py --config C2 --base-url http://localhost:8000
```

#### Config D — g6.xlarge × 2대 (같은 AZ 필수)

**decode 노드**:
```bash
export EXP_LOG_DIR=/home/ubuntu/exp-logs
bash disagg-exp/setup.sh
bash disagg-exp/launch_configs.sh configD decode
```

**prefill 노드** (decode 노드의 private IP를 알고 있어야 함):
```bash
export DECODER_HOST=<decode-private-ip>
export EXP_LOG_DIR=/home/ubuntu/exp-logs
bash disagg-exp/setup.sh
bash disagg-exp/launch_configs.sh configD prefill &
# 양쪽 ready 확인 후
export PREFILL_HOST=127.0.0.1
export DECODE_HOST=<decode-private-ip>
bash disagg-exp/launch_configs.sh configD proxy &
python disagg-exp/sweep.py --config D --base-url http://localhost:8000
```

### Step 7 — Resume / 재시도

`.done_<point>` 마커가 있으면 sweep.py가 자동 skip. 실패한 point만 재시도하려면:
```bash
rm $EXP_LOG_DIR/A1/.failed_p2048_d4096_r8.0
python disagg-exp/sweep.py --config A1 --base-url http://localhost:8000
```

### Step 8 — 결과 회수 + 분석 (로컬에서)

```bash
aws s3 sync s3://hdjung-disaggregation-result/raw/ ./data/

python disagg-exp/analyze.py --log-dir ./data --plot
# 기본 configs: ["A1","A2","A3","B","C1","C2","D"]
# 출력: 표 + ./data/plots/plot_p*_d*.png
```

### Step 9 — 인스턴스 종료 (잊지 말 것)

**Spot도 종료까지 과금됨.**

```bash
aws ec2 terminate-instances --instance-ids i-xxx i-yyy
aws ce get-cost-and-usage \
  --time-period Start=2026-05-17,End=2026-05-24 \
  --granularity DAILY --metrics UnblendedCost
```

---

## 7. S3 자동 sync (sweep.py 내장)

별도 명령 불필요. `sweep.py` 실행 시 자동으로 백그라운드 스레드가 30초 주기로 `$EXP_LOG_DIR/`를 S3로 sync한다.

| 항목 | 값 |
|---|---|
| 기본 버킷 | `hdjung-disaggregation-result` |
| 목적지 | `s3://{bucket}/raw/{YYYYMMDD-UTC}/{hostname}/` |
| 주기 | 30초 (`S3_SYNC_INTERVAL` env로 override) |
| 도구 | `s5cmd` 우선, 없으면 `aws s3 sync` fallback |
| 종료 처리 | 정상 종료 시 final sync + `atexit`로 비정상 종료(Ctrl-C, 예외)에서도 final sync |
| 실패 처리 | sync 실패해도 sweep는 계속. 에러는 `s3_sync.log`에 기록 |

```bash
# 기본 (자동 sync)
python disagg-exp/sweep.py --config A1 --base-url http://localhost:8000

# 다른 버킷 사용
python disagg-exp/sweep.py --config A1 ... --s3-bucket my-other-bucket

# S3 sync 끔
python disagg-exp/sweep.py --config A1 ... --s3-bucket ""
```

S3 접근 권한은 인스턴스 IAM Role(`hdjung_disaggregation_result`)에서 부여. 인스턴스 시작 시 Advanced details → IAM instance profile에서 선택.

---

## 8. 실행 체크리스트

**사전 준비**
- [ ] SG, S3 버킷, IAM Role 생성
- [ ] 코드 fork에 push 완료
- [ ] g4dn.xlarge AMI builder에서 setup.sh 통과
- [ ] smoke test 성공 (`"status":"success"` 라인 있음)
- [ ] 모델 다운로드 + S3 캐시
- [ ] AMI 굽기 (AMI ID 메모)

**Config A1/A2/A3** (g4dn.12xlarge, 같은 인스턴스 순차)
- [ ] IAM Role `hdjung_disaggregation_result` 붙음
- [ ] `launch_configs.sh configA1` → `/health` 200
- [ ] small sweep 1 point 성공
- [ ] A1 full sweep (18 points)
- [ ] A2 full sweep
- [ ] A3 full sweep
- [ ] 인스턴스 종료

**Config B** (g6e.xlarge)
- [ ] 동일 flow

**Config C1** (g4dn.12xlarge, same-node PD, TP=2 PP=1)
- [ ] decode → prefill → proxy 순으로 기동
- [ ] `kv_transfer.jsonl`에 `wait_for_save_done` 라인 들어옴
- [ ] `vllm_configC1_prefill_<host>.log`에 `connector lmcache_engine type=...` 1회
- [ ] TP+PD sanity test (작은 sweep 먼저)
- [ ] full sweep

**Config C2** (g4dn.12xlarge, same-node PD, TP=1 PP=2)
- [ ] C1 완전히 정지 후 (같은 인스턴스 재사용)
- [ ] 동일 flow

**Config D** (g6.xlarge × 2, 같은 AZ)
- [ ] decode 노드 → prefill 노드 → proxy 순서
- [ ] cross-node 시간 동기 (offset < 50ms)

**분석**
- [ ] `aws s3 sync` 로 데이터 회수
- [ ] `analyze.py --plot`
- [ ] $/M-tokens 비교표

---

## 9. Gotcha (실험 무의미해질 수 있는 함정들)

| 함정 | 대처 |
|---|---|
| **TP+PD over NIXL with TP>1** vLLM 0.21에서 작동 보장 불확실 | Config C1(TP=2) 첫 요청으로 sanity test. 안 되면 C2(TP=1 PP=2)로 fallback (TP=1이면 NIXL이 단순해짐) |
| **T4 VRAM 빡빡** TP=2 → 8GB/GPU | OOM 시 `MAX_MODEL_LEN=2048` 또는 `GPU_MEM_UTIL=0.80` |
| **UCX_TLS 잘못 고름** `cuda_ipc`는 `CUDA_VISIBLE_DEVICES` split 시 불가 | same-node = `cuda_copy,shm,tcp`, cross-node = `tcp` |
| **BOS +1** 서버가 BOS prepend | `prompt_tokens = prefill_len + 1`은 정상. +1 초과 시 analyze.py 경고 |
| **Config D 다른 AZ** 인스턴스 두 대 다른 AZ면 TCP latency 폭증 | 같은 AZ에 두 대 |
| **Clock sync** | `clock_baseline_<host>.txt` offset < 50ms 확인 |
| **enforce-eager** CUDA Graph 끄면 throughput 20~40% 떨어짐 | `COMMON_FLAGS`에 안 들어가 있음 (확인 완료) |
| **chunked-prefill** | PD prefill role만 `--no-enable-chunked-prefill`, monolithic은 default(ON) |
| **`kv_connector_module_path`** 빠지면 factory가 `ValueError` | `launch_configs.sh`가 자동으로 넣음 |
| **TP=2 시 instrumented_connector 동시 쓰기** | rank-0만 log 작성 (코드에 반영) |
| **Spot 종료** | 실험 끝나면 즉시 terminate. 비용 확인은 `aws ce get-cost-and-usage` |

---

## 10. 산출물

- Config 대결 표: 18 points × 7 configs의 TTFT p50/p99 / TPOT p50/p99 / $/M-tokens (총 126 points)
- Panel 1: TTFT vs rate (config별 선, prefill_len별 facet)
- Panel 2: $/M-tokens vs rate (config별 선)
- Panel 3: KV 전송 시간 (C1 vs C2 vs D, `kv_transfer.jsonl` 기반)
- 결론: "같은 예산이면 어떤 구성이 언제 유리한가" 한 줄 결론
