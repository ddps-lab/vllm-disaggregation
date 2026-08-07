# HWProfiling — MoE Expert FFN Micro-Benchmark

MoE expert 1개의 FFN 연산을 vLLM과 동일한 dataflow로 mimic하여, batch size(M =
expert에 라우팅된 토큰 수)별 **effective FLOPS / memory bandwidth**를 GPU별로
측정한다.

## 측정하는 것

vLLM의 default bf16 triton `fused_moe` 경로는 FFN chain을 fusion하지 않는다
(`vllm/model_executor/layers/fused_moe/fused_moe.py`의 `fused_experts_impl`):

| stage | 연산 | HBM traffic |
|---|---|---|
| `w13_gemm` | `x[M,K] @ w13[2I,K]ᵀ → h[M,2I]` | read x,w13 / **write h** |
| `silu_and_mul` | `silu(h[:, :I]) * h[:, I:] → a[M,I]` (별도 커널) | **read h / write a** |
| `w2_gemm` | `a[M,I] @ w2[K,I]ᵀ → y[M,K]` | **re-read a**, read w2 / write y |
| `full_chain` | 위 3개 연속 실행 | 합 |

따라서 순수 PyTorch `GEMM → silu_and_mul → GEMM` 시퀀스는 vLLM serving과 동일한
activation memory traffic을 가진다. weight layout도 vLLM과 동일:
`w13 [2I, K]` (gate 먼저, up 나중), `w2 [K, I]`, bias 없음.

## 대상 모델 / 인스턴스

| model | K (hidden) | I (moe inter) | experts | shared | MoE layers |
|---|---|---|---|---|---|
| mixtral-8x22b | 6144 | 16384 | 8 (top-2) | — | 56 |
| qwen3-30b-a3b | 2048 | 768 | 128 (top-8) | — | 48 |
| deepseek-v2-lite | 2048 | 1408 | 64 (top-6) | 2 | 26 |
| glm-5.2 | 6144 | 2048 | 256 (top-8) | 1 | 75 |

config는 실행 시 HuggingFace `AutoConfig`로 로드한다 (weight 다운로드 없음 —
random weight 사용; 값은 GEMM 시간에 영향 없음). 네트워크 실패 시 하드코딩된
fallback을 쓰고 JSON에 `config_source: "fallback_dims"`로 표시된다.

인스턴스: g4dn.xlarge(T4) · g5.xlarge(A10G) · g6.xlarge(L4) · g6e.xlarge(L40S).

## 실행 (인스턴스에서만 — 로컬 실행 금지)

```bash
# 로컬 → 인스턴스 동기화 (one-way)
rsync -avz --exclude results/ experiment/HWProfiling/ <ip>:~/vllm-disaggregation/experiment/HWProfiling/

# 인스턴스에서
cd ~/vllm-disaggregation
uv run python experiment/HWProfiling/main.py --model all
uv run python experiment/HWProfiling/main.py --model qwen3-30b-a3b --batch-sizes 1,16,256,4096

# 결과 회수
rsync -avz <ip>:~/vllm-disaggregation/experiment/HWProfiling/results/ experiment/HWProfiling/results/
```

주요 옵션: `--dtype {auto,bf16,fp16}` (auto는 SM<80에서 fp16 fallback — T4),
`--warmup 10 --iters 50`, `--instance-type <label>` (IMDS 미검출 시),
`--mem-budget-gb`, `--cooldown <s>` (T4 thermal throttle 대응), `--no-shared`,
`--no-vllm-op`, `-v`.

## 결과 레이아웃 (실행 시 동적 생성)

```
results/<model>/<instance_type>/
├── run_<timestamp>.log        # 전체 실행 로그
├── results_<timestamp>.json   # config/환경 메타데이터 + 전체 측정치
└── results_<timestamp>.csv    # 사람이 읽기 쉬운 flat 요약
```

## 측정 방법론 노트

- **L2 cache 대책 (replica 회전)**: 작은 expert(Qwen3 w13 = 6 MB)는 L4의 48 MB
  L2에 통째로 들어가므로, 한 weight로 반복 측정하면 HBM이 아닌 L2 BW가 나온다.
  실제 serving(AFD의 FFN server)처럼 **MoE layer 수만큼 서로 다른
  weight+input replica**를 만들어 iteration마다 갈아끼운다. VRAM 예산 초과 시
  clamp하며, 회전 footprint < 2×L2이면 `rotation_ok=false`로 표시된다.
  예외: Mixtral-8x22B는 expert 1개가 604 MB라 `replica_cap=8`로 상한
  (weight 4.8 GB — 여전히 2×L2를 크게 상회).
  `achieved_gbps > 1.15×peak`이면 회전 부족 경고가 로그에 남는다.
- **silu_and_mul**: vLLM 설치 시 동일 CUDA 커널(`torch.ops._C.silu_and_mul`)을
  쓰고, 없으면 torch.compile fusion, 최후에 eager (JSON `silu_impl`에 기록 —
  eager는 activation BW가 부풀려짐).
- **timing**: CUDA event pair로 iteration별 latency 기록, **median**이 주 통계
  (clock ramp/throttle에 강건). warmup은 (stage, M)마다 수행 — cuBLAS heuristic
  선택이 shape별 첫 호출에 일어나기 때문.
- **M=1의 silu_and_mul**은 kernel launch latency 지배적이라 BW 수치가 의미
  없음 — `min` latency를 함께 참고.
- peak 수치는 NVIDIA datasheet의 **dense** (non-sparsity) FP16/BF16 tensor-core
  TFLOPS 기준 (`profiler/device.py`의 `PEAK_TABLE`).
