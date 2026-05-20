## 실험 1차 — Tier 1 Configuration 대결 (v4)

### 0. 핵심 질문

같은 예산이면 큰 GPU 한 장이 나은가, 작은 GPU 여러 장 + PD 분리가 나은가? 같은 4-GPU에서 TP/PP 비율은?

7가지 토폴로지에서 같은 워크로드를 돌려 TTFT / TPOT / $/M-tokens 비교.

---

### 1. 실험 환경

- **모델**: `meta-llama/Llama-3.1-8B-Instruct` (BF16, ~16GB weights)
- **vLLM**: 로컬 fork `releases/v0.21.0` base, editable install
- **LMCache**: ≥ 0.3.9
- **리전**: us-west-2 (Oregon), DLAMI Ubuntu 22.04, Python 3.12
- **S3 버킷**: `hdjung-disaggregation-result` ([sweep.py](http://sweep.py/) 내장 자동 sync)
- **IAM Role**: `hdjung_disaggregation_result`
- **SG**: `hdjung-vllm-try`
- **비교 정규화**: $/M-tokens
- **AMI 이름:** disagg-exp-tier1-ready
- **AMI 태그:** ami-032b0c7aa63b8fe2a

---

### 2. 7가지 Config

| Config | Instance | GPU | Topology | $/hr spot |
| --- | --- | --- | --- | --- |
| **A1** | g4dn.12xlarge | 4× T4 16GB | Monolithic **TP=2 PP=2** | ~$1.14 |
| **A2** | g4dn.12xlarge | 4× T4 16GB | Monolithic **TP=4 PP=1** | ~$1.14 |
| **A3** | g4dn.12xlarge | 4× T4 16GB | Monolithic **TP=1 PP=4** | ~$1.14 |
| **B** | g6e.xlarge | 1× L40S 48GB | Monolithic single GPU | ~$0.60 |
| **C** | g4dn.12xlarge | 4× T4 16GB | Same-node PD. prefill TP=2 (GPU 0-1) + decode TP=2 (GPU 2-3). NIXL via shm | ~$1.14 |
| **D** | 2× g6.xlarge | 2× L4 24GB | Cross-node PD. prefill TP=1 + decode TP=1. NIXL via TCP | ~$0.54 |

**비교 의미**:

- **A1 vs A2 vs A3**: 같은 4× T4에서 TP/PP 비율이 throughput/latency에 미치는 영향
- **best-A vs C1/C2**: 최적 monolithic vs same-node PD → 순수 소프트웨어 비교
- **C1 vs C2**: 같은 PD, 같은 GPU 배치에서 TP vs PP 비교
- **B vs best-A**: 1 큰 GPU vs 4 작은 GPU → 멀티-GPU 오버헤드 평가
- **C vs D**: 둘 다 PD. link 다름(shm vs TCP) → $/M-tokens 정규화
- **B vs D**: 비슷 가격대. 1 강한 GPU vs 2 작은 GPU + PD

### 2-1. 모델 fit

- T4 (16GB): 단독 ❌ → TP≥2 필수. TP=1 PP=4일 때 각 stage weight ~4GB/GPU. `gpu-memory-utilization 0.85`, `max-model-len 4096`
- L4 (24GB): 단독 ✅ (8GB 여유)
- L40S (48GB): 단독 ✅ 넉넉

---

### 3. Noise Controls

**Server side** (`launch_configs.sh` COMMON_FLAGS):

- `-no-enable-prefix-caching` — KV 캐시 재사용 차단 (같은 프롬프트 실험 시 캐시 오염 방지)
- `-no-enable-chunked-prefill` → PD prefill role에만 적용
- `-served-model-name llama-3.1-8b`
- **`-dtype half` (float16)** — 모든 config에서 명시적 통일. T4 native, L40S/L4는 bfloat16에서 downgrade. dtype 혼재 시 dtype 효과가 throughput/latency 차이에 섞여 cross-config 비교 왜곡.
- `-max-model-len 4096` (env override 가능)
- `-gpu-memory-utilization 0.85`
- **`-max-num-seqs 512`** (기본값 256은 큐 병목 유발)
- ⚠️ **`-max-num-batched-tokens` 미설정** (default가 auto이고 auto는 가능한 최대를 쓰는거라면 미설정해도 쓰루풋에 영향을 안미치기 때문)
- **CUDA Graph ON** (enforce-eager 안 씀)

**GPU dtype: 명시적 float16 통일**:
- **이유**: T4 (CC 7.5) → float16 native, L40S/L4 (CC ≥ 8.0) → bfloat16 native인데, 섞어 쓰면 dtype 정밀도 차이가 throughput/latency에 반영되어 configs 비교가 왜곡됨.
- **결정**: 모든 config에서 `--dtype half` (float16)으로 통일 → 순수 토폴로지/병렬화 효과 측정 가능.

**CUDA/Python 빌드 요구사항**:
- **python3.12-dev**: FlashInfer JIT 컴파일용 헤더 필요
- **CUDA bundled path**: DLAMI PyTorch 2.11+는 site-packages/nvidia/cu13 아래 CUDA 번들 → `CUDA_HOME` export + `lib64` 심링크 필수

**Client side** (`sweep.py`):

- 최대 배치 512(배치제한을 없애서 쓰루풋 최대)
- `prompt: list[int]` (token ID 직접 전달)
- `min_tokens = max_tokens = decode_len` + `ignore_eos = true`
- `temperature = 0`, `top_p = 1.0`
- `stream = true`, `stream_options = {include_usage: true}`

---

### 4. 워크로드 그리드

```jsx
PREFILL_LENS = [512, 2048]              ← 축소 (max-model-len=4096 제약)
DECODE_LENS  = [128, 512, 1024]         ← 축소 (max-model-len=4096 제약)
RATES        = [1.0, 2.0, 4.0]  ← 축소 (0.5,6,8 제거)

Cross 1 (prefill × rate, decode=512 고정):  2 × 3 = 6 points
Cross 2 (decode × rate, prefill=2048 고정): 3 × 3 = 9 points (중복 1점 제거)
Total: 14 points/config × 7 configs = 98 points
warmup=50, measured=300 → 350 req/point  ← measured 2배 증가 (p99 신뢰도↑)
```

**max-model-len=4096 제약**:
- T4 (16GB VRAM) 환경에서 max_model_len=4096 고정 (vLLM 요청 시 context window 제한)
- prefill + decode 합이 4096을 초과하면 OOM 발생 → 실험 불가
- 최대 안전 조합: prefill=2048 + decode=1024 = 3072 tokens
- prefill=512, decode=4096 등의 극단 조합 제거

**RATES 축소 이유**: 0.5는 너무 저부하(noise), 6.0+는 극한/dropout 유발 → 1.0(low), 2.0(mid), 4.0(high saturation) 3점으로 충분

**measured=300 증가 이유**: 원래 200 → p50 신뢰도 낮음 → 300으로 p99 신뢰도 향상

A1, A2, A3, C1, C2는 같은 인스턴스에서 순차 실행 → 18 × 5 = 90 points, 인스턴스 비용은 1대분.

### 4-1. Overload 보호 (2-phase)

warmup 50건 끝나면: `fail_rate > 0.3` 또는 **`ttft_p99 > 180s`** ← 30s → 180s 완화 (T4 극한 환경 대응)

### 4-2. Throughput 정의

- `thr_tok_s_e2e = tokens / (max(recv) − min(send))`
- `thr_tok_s_send = tokens / (max(send) − min(send))`
- `achieved_rate = N_ok / arrival_window`

---

### 5. 코드 구성

| 파일 | 역할 |
| --- | --- |
| `setup.sh` | 노드 부트스트랩. uv venv → vLLM → lmcache → chrony → DCGM → metric collectors → s5cmd |
| `launch_configs.sh` | 디스패처 (A1/A2/A3/B/C1/C2/D × role) |
| `instrumented_connector.py` | PD prefill KV-transfer 시간 기록 |
| `sweep.py` | Poisson 워크로드 + **S3 자동 sync 내장** |
| `analyze.py` | p50/p99, $/M-tokens 표 + plot |

---

### 6. 핵심 gotcha & 구현 특이사항

**dtype & 빌드**:
- **GPU dtype**: nvidia-smi의 compute_cap으로 CC 감지 → bc 필요 (부동소수점 비교)
- **T4 → float16**: bfloat16 미지원 경고 + auto-fallback
- **python3.12-dev**: FlashInfer/CUDA JIT 컴파일 헤더 필수 (apt-get)
- **CUDA bundled path**: DLAMI PyTorch 2.11+는 site-packages/nvidia/cu13 아래 → 감지 + lib→lib64 symlink 생성

**vLLM 0.21 connector & PD**:
- **connector 메서드**: 첫 호출 시 introspection으로 메서드명 확인 (0.21 variant compatibility)
- **TP + PD 위험**: NIXL + TP > 1 → Config C1/C2 첫 요청으로 sanity test 필수
- **TP=1 PP=4 (A3)**: PP bubble 발생 가능. 각 stage weight ~4GB/GPU, 파이프라인 불균형

**메모리 & 리소스**:
- **T4 VRAM**: TP=2 → 8GB/GPU 빡빡. 최대 context window 제약 (max-model-len=4096)
- **max-model-len=4096 제약**: prefill + decode 합이 4096을 넘으면 OOM 발생. 따라서 PREFILL_LENS=[512,2048], DECODE_LENS=[128,512,1024] (최대 조합: 2048+1024=3072 ≤ 4096)
- **OOM 회피**: env 오버라이드로 `MAX_MODEL_LEN=2048` or `GPU_MEM_UTIL=0.75` 사용 가능 (측정값에 영향)
- **UCX_TLS**: Same-node (C1/C2) = `cuda_copy,shm,tcp`. Cross-node (D) = `tcp` 명시

**S3 sync & 로그**:
- **S3Syncer**: threading 기반 30초 interval, s5cmd/aws cli auto-fallback, atexit에서 final sync (graceful shutdown)
- **S3 경로**: `s3://bucket/raw/custom/{YYYYMMDD}/{hostname}/{config}/` (shared bucket 내 isolation)

**워크로드 & 데이터**:
- **BOS +1**: `prompt_tokens = prefill_len + 1` 정상 (모델이 BOS 자동 prepend)
- **RATES**: [0.5, 6.0, 8.0] 제거 → [1.0, 2.0, 4.0]로 축소 (노이즈↓, 의미있는 범위)
- **measured=300**: 기존 200 → p99/p95 신뢰도 향상
- **LMCache**: ≥0.3.9 strictly enforced (version check in setup.sh)

**Config 명명**:
- **A1/A2/A3**: 기존 “A” → 명시적 TP/PP 라벨 (2,2 / 4,1 / 1,4)
- **C1/C2**: 기존 “C” → PD topology 내 TP/PP 구분 (2,1 / 1,2)
- **backward compat**: configA/configC → A1/C1 alias로 유지

---

### 7. AWS 사전 자원 (완료)

- [x]  SG: `hdjung-vllm-try`
- [x]  S3: `hdjung-disaggregation-result`
- [x]  IAM Role: `hdjung_disaggregation_result`
- [x]  코드 작성 완료

---

### 8. 실행 가이드

### 8-1. 첫 인스턴스 (검증 + AMI)

EC2 콘솔 → Launch instances:

- AMI: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)`
- Instance: `g4dn.xlarge` (~$0.20/hr spot)
- SG: `hdjung-vllm-try`, IAM: `hdjung_disaggregation_result`
- Storage: 200 GiB gp3, Spot

### 8-2. SSH + 검증

```bash
ssh ubuntu@<ip>
git clone <fork> && cd vllm-disaggregation
export EXP_LOG_DIR=/home/ubuntu/exp-logs

# setup.sh는 idempotent (재실행 안전)
# - python3.12-dev 설치 (apt-get)
# - CUDA bundled path 자동 감지 (site-packages/nvidia/cu13)
bash disagg-exp/setup.sh && source .venv/bin/activate

# 검증
python -c "import vllm, lmcache; print(vllm.__version__, lmcache.__version__)"
# 기대: vllm 0.21.x  lmcache 0.3.9+

# 시스템 체크
which nvidia-smi s5cmd  # GPU 감지, S3 sync용
```

### 8-3. Smoke test → 모델 다운 → AMI 굽기

상세 절차는 이전 v3 참고. AMI 완성 후 builder 인스턴스 Terminate.

### 8-4. 실험 실행 (AMI로)

**Config A1, A2, A3, C1, C2** (g4dn.12xlarge 1대):

```bash
# dtype auto-detection + CUDA bundled path가 자동 설정됨
# launch_configs.sh에서 nvidia-smi CC 감지 → T4면 float16, L40S/L4면 bfloat16

# A1: TP=2 PP=2 (통신 많음, 낮은 latency)
bash disagg-exp/launch_configs.sh configA1
python disagg-exp/sweep.py --config A1 --base-url http://localhost:8000 --s3-bucket hdjung-disaggregation-result

# A2: TP=4 PP=1 (uniform communication)
bash disagg-exp/launch_configs.sh configA2
python disagg-exp/sweep.py --config A2 --base-url http://localhost:8000

# A3: TP=1 PP=4 (버블 주의, 낮은 latency 기대)
bash disagg-exp/launch_configs.sh configA3
python disagg-exp/sweep.py --config A3 --base-url http://localhost:8000

# C1: same-node PD TP=2 PP=1 prefill/decode
bash disagg-exp/launch_configs.sh configC1 prefill &
sleep 10
bash disagg-exp/launch_configs.sh configC1 decode &
sleep 10
bash disagg-exp/launch_configs.sh configC1 proxy
# → 다른 터미널에서
python disagg-exp/sweep.py --config C1 --base-url http://localhost:8000

# C2: same-node PD TP=1 PP=2 prefill/decode
# (C1과 동일 절차)
```

**Config B** (g6e.xlarge 1대): 단일 L40S, TP=1

**Config D** (g6.xlarge × 2대, 같은 AZ):
- 각 인스턴스에서 prefill/decode 분리
- DECODER_HOST 환경변수 필수 (decode 노드 private IP)

### 8-5. 분석

```bash
python disagg-exp/analyze.py --log-dir $EXP_LOG_DIR --configs A1 A2 A3 B C1 C2 D --plot
```

---

### 9. 체크리스트

- [x]  SG, S3, IAM
- [x]  코드 완성 (v6: dtype auto-detect, python3.12-dev, CUDA bundled, RATES축소, measured=300)
- [ ]  첫 인스턴스 setup.sh 검증 (bc, python3.12-dev, CUDA bundled path)
- [ ]  smoke test (dtype 자동 선택 확인)
- [ ]  모델 다운 + S3 캐시
- [ ]  AMI 굽기
- [ ]  Config A1/A2/A3 + C1/C2 (g4dn.12xlarge)
- [ ]  Config B (g6e.xlarge, single L40S)
- [ ]  Config D (2× g6.xlarge, cross-node PD via TCP)
- [ ]  분석 + 결론

### 10. 산출물

- **Panel 0**: A1 vs A2 vs A3 — TP/PP 비교 (같은 하드웨어, 같은 GPU 메모리)
- **Panel 0b**: C1 vs C2 — PD 내 TP vs PP 비교
- **Panel 1**: TTFT p50/p99 vs prefill_len (config별, rate별 facet)
- **Panel 2**: TPOT p50 vs rate (config별)
- **Panel 3**: $/M-tokens vs rate (cost-normalized 비교)
- **Panel 4**: KV 전송 시간 분포 (C1 vs C2 vs D) — instrumented_connector.py 수집
- **요약 테이블**: 18점 grid × 7 configs, TTFT/TPOT/throughput/cost 정규화

**분석 스크립트**:

```bash
python disagg-exp/analyze.py --log-dir $EXP_LOG_DIR --configs A1 A2 A3 B C1 C2 D --plot
# → ./results/plots/{plot_p512_d128, plot_p512_d512, ...}.png
```

**구글 시트 예상**:
- 각 config별 격자 18×7 표
- 드래그 다운 pivot: config × prefill_len × rate → TTFT/TPOT/cost 분해
- 스캐터: rate vs cost (B/D/C/A 클러스터링)