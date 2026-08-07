# HWProfiling — MoE Expert FFN Micro-Benchmark

MoE expert 1개의 FFN 연산을 vLLM과 동일한 dataflow로 mimic하여, batch size(M =
expert에 라우팅된 토큰 수)별 **effective FLOPS / memory bandwidth**를 GPU별로
측정한다.

## 측정하는 것

vLLM의 default bf16 triton `fused_moe` 경로는 FFN chain을 fusion하지 않는다
(`vllm/model_executor/layers/fused_moe/fused_moe.py`의 `fused_experts_impl`):

| stage            | 연산                                                | HBM traffic                            |
| ---------------- | --------------------------------------------------- | -------------------------------------- |
| `w13_gemm`     | `x[M,K] @ w13[2I,K]ᵀ → h[M,2I]`                 | read x,w13 /**write h**          |
| `silu_and_mul` | `silu(h[:, :I]) * h[:, I:] → a[M,I]` (별도 커널) | **read h / write a**             |
| `w2_gemm`      | `a[M,I] @ w2[K,I]ᵀ → y[M,K]`                    | **re-read a**, read w2 / write y |
| `full_chain`   | 위 3개 연속 실행                                    | 합                                     |

따라서 순수 PyTorch `GEMM → silu_and_mul → GEMM` 시퀀스는 vLLM serving과 동일한
activation memory traffic을 가진다. weight layout도 vLLM과 동일:
`w13 [2I, K]` (gate 먼저, up 나중), `w2 [K, I]`, bias 없음.

## 대상 모델 / 인스턴스

| model            | K (hidden) | I (moe inter) | experts     | shared | MoE layers |
| ---------------- | ---------- | ------------- | ----------- | ------ | ---------- |
| mixtral-8x22b    | 6144       | 16384         | 8 (top-2)   | —     | 56         |
| qwen3-30b-a3b    | 2048       | 768           | 128 (top-8) | —     | 48         |
| deepseek-v2-lite | 2048       | 1408          | 64 (top-6)  | 2      | 26         |
| glm-5.2          | 6144       | 2048          | 256 (top-8) | 1      | 75         |

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

### Plot (로컬, matplotlib만 필요)

```bash
python experiment/HWProfiling/plot.py   # results/<model>/<instance>/plots/<kind>.png
```

(모델, instance)별 최신 결과에서 expert_kind별 그림 하나를 그린다. stage별
subplot에 achieved TFLOPS(왼쪽 축, 파랑 실선 ●)와 achieved GB/s(오른쪽 축,
주황 파선 ■)를 함께 표시하고, 점선은 datasheet peak. silu_and_mul은 pointwise
연산이라 bandwidth만 그린다.

주요 옵션: `--dtype {auto,bf16,fp16}` (auto는 SM<80에서 fp16 fallback — T4),
`--warmup 10 --iters 50`, `--instance-type <label>` (IMDS 미검출 시),
`--mem-budget-gb`, `--cooldown <s>` (T4 thermal throttle 대응), `--no-shared`,
`--no-vllm-op`, `-v`.

## 결과 레이아웃 (실행 시 동적 생성)

```
results/<model>/<instance_type>/
├── run_<timestamp>.log        # 전체 실행 로그
├── config_<timestamp>.json    # 실험 config/환경 메타데이터만
└── results_<timestamp>.csv    # 전체 측정치 (측정 데이터의 단일 소스)
```

주의: 저장소 루트 .gitignore(vLLM 본체)가 `*.log`(92행)와 `*.csv`(219행)를
전역으로 무시하므로 results의 log/csv는 git에 잡히지 않는다.

## 측정 방법론 노트

- **L2 cache 대책 (replica 회전)**: 작은
  expert(Qwen3 w13 = 6 MB)는 L4의 48 MB L2에 통째로 들어가므로, 한 weight로
  반복 측정하면 HBM이 아닌 L2 BW가 나온다. 실제 serving(AFD의 FFN server)처럼
  **MoE layer 수만큼 서로 다른 weight+input replica**를 만들어 iteration마다
  갈아끼운다. VRAM 예산 초과 시 clamp하며, 회전 footprint < 2×L2이면
  `rotation_ok=false`로 표시된다. 예외: Mixtral-8x22B는 expert 1개가 604 MB라
  `replica_cap=8`로 상한 (weight 4.8 GB — 여전히 2×L2를 크게 상회).
- **flush 방식은 검증 후 제거됨**: "replica 1개 + iteration마다 L2 flush"
  대안을 g6.xlarge에서 비교한 결과, 쓰기 기반(`zero_()`)은 dirty-line
  writeback으로 GEMM이 +30~45% 부풀려지고, 읽기 기반(randn 8×L2)은 반대로
  최대 −50% 낙관 편향 (버퍼 크기·내용과 무관 → L2가 아니라 **TLB/페이지
  지역성** 때문: replica 1개만 반복 접근하면 TLB가 hot인데, serving은
  layer마다 다른 weight를 스트리밍하므로 TLB까지 cold). flush로는 이를 제거할
  수 없어 회전 방식만 사용한다. Mixtral처럼 expert가 수백 MB면 두 방식이
  ±2%로 일치했다 (회전의 sanity check).
- **silu_and_mul**: vLLM 설치 시 동일 CUDA 커널(`torch.ops._C.silu_and_mul`)을
  쓰고, 없으면 torch.compile fusion, 최후에 eager (JSON `silu_impl`에 기록 —
  eager는 activation BW가 부풀려짐).
- **timing**: CUDA event pair로 iteration별 latency 기록, **median**이 주 통계
  (clock ramp/throttle에 강건). warmup은 (stage, M)마다 수행 — cuBLAS heuristic
  선택이 shape별 첫 호출에 일어나기 때문.
- **M=1의 silu_and_mul**은 kernel launch latency 지배적이라 BW 수치가 의미
  없음 — `min` latency를 함께 참고.
- peak 수치는 NVIDIA datasheet의 **dense** (non-sparsity) FP16/BF16 tensor-core
  TFLOPS 기준 (`profiler.py`의 `PEAK_TABLE`).

## Roofline 분석

`python analyze.py`가 아래 프레임으로 results/ 전체를 분석한 표를 출력한다.

### 표기

M = expert에 라우팅된 토큰 수, K = hidden size, I = expert intermediate size
(shared expert는 I × n_shared), e = 2 bytes (bf16/fp16).

### Stage별 FLOPs / 메모리 접근량

| stage                | FLOPs                              | bytes (HBM)                               |
| -------------------- | ---------------------------------- | ----------------------------------------- |
| up&gate (w13)        | `4·M·I·K`                     | `e·(M·K + 2·I·K + 2·M·I)`         |
| activation (silu)    | `5·M·I` (명목)                 | `e·3·M·I`                            |
| down (w2)            | `2·M·I·K`                     | `e·(M·I + I·K + M·K)`               |
| **full chain** | **`6·M·I·K + 5·M·I`** | **`e·(3·I·K + M·(2K + 6I))`** |

weight 항(`3·I·K`)은 M과 무관한 상수, activation 항은 M에 비례 — 이 구조가
아래 AI 함수의 형태를 결정한다.

### Arithmetic intensity 함수 (full chain)

$$
AI(M) = \frac{6MIK}{2\,(3IK + M(2K+6I))} = \frac{M \cdot M_c}{M + M_c}
\quad\text{[FLOP/byte]},\qquad M_c = \frac{3IK}{2K+6I}
$$

M_c는 "이 모델이 도달할 수 있는 arithmetic intensity의 상한"이다
(단위 FLOP/byte). 직관적으로:

- 작은 M: bytes가 weight(`3IK`, 상수)에 지배되므로 M을 키우는 만큼 연산이
  늘어 `AI ≈ M`으로 선형 증가 (모든 모델 공통 기울기).
- 큰 M: bytes도 activation(`M(2K+6I)`)이 지배해 M과 같이 늘어나므로 AI가
  더는 오르지 못하고 **M_c로 수렴**한다. M = M_c일 때 정확히 상한의 절반
  (`AI = M_c/2`)에 도달한다.
- 따라서 **M_c가 GPU의 Ridge Point보다 작으면, 그 모델은 그 GPU에서 어떤
  M에서도 compute-bound가 되지 못한다** (M을 무한히 키워도 AI < R).

| model (kind)            | K    | I     | **M_c** |
| ----------------------- | ---- | ----- | ------------- |
| mixtral-8x22b           | 6144 | 16384 | 2731          |
| glm-5.2 (routed=shared) | 6144 | 2048  | 1536          |
| deepseek-v2-lite shared | 2048 | 2816  | 824           |
| deepseek-v2-lite routed | 2048 | 1408  | 690           |
| qwen3-30b-a3b           | 2048 | 768   | 542           |

### GPU별 · 모델별 이론 vs 실측 (full chain)

Ridge Point = `FLOPS / BW` [FLOP/byte]. **effective 값은 latency 곡선의
무릎(knee)으로 분리해 추출**한다: latency는 M이 작을 땐 평평하고(= weight
스트리밍이 지배하는 memory-bound 바닥, M과 무관) 어느 지점부터 상승하는데
(= M 비례 항 지배), `latency > 1.5×floor`가 되는 M(로그 보간)을 knee로 잡아
- `BW_eff` = knee **아래** 점들의 implied BW(`bytes/t`) 중앙값
- `F_eff` = knee **위** 점들의 implied FLOPS(`flops/t`) 중앙값
을 취한다 (fitting 없음, 곡선에서 직접 읽는 empirical 분리).
glm-5.2는 routed/shared가 같은 차원이라 routed 값만 표기 (shared ≈ ±2%).
(2026-08-07 측정; T4는 fp16, 나머지는 bf16)

| GPU | model | FLOPS (ideal → effective, TF) | BW (ideal → effective, GB/s) | Ridge Point (ideal → effective) |
| --- | --- | --- | --- | --- |
| T4 (g4dn) | mixtral-8x22b | 65 → **21.6** (33%) | 320 → **237** (74%) | 203 → **91** |
| | glm-5.2 | 65 → **19.2** (30%) | 320 → **219** (68%) | 203 → **88** |
| | dsv2-lite shared | 65 → **21.4** (33%) | 320 → **177** (55%) | 203 → **121** |
| | dsv2-lite routed | 65 → **21.8** (34%) | 320 → **200** (63%) | 203 → **109** |
| | qwen3-30b-a3b | 65 → **22.6** (35%) | 320 → **138** (43%) | 203 → **164** |
| A10G (g5) | mixtral-8x22b | 70 → **65.8** (94%) | 600 → **451** (75%) | 117 → **146** |
| | glm-5.2 | 70 → **64.9** (93%) | 600 → **422** (70%) | 117 → **154** |
| | dsv2-lite shared | 70 → **60.0** (86%) | 600 → **364** (61%) | 117 → **165** |
| | dsv2-lite routed | 70 → **61.4** (88%) | 600 → **226** (38%) | 117 → **272** |
| | qwen3-30b-a3b | 70 → **59.6** (85%) | 600 → **134** (22%) | 117 → **445** |
| L4 (g6) | mixtral-8x22b | 121 → **56.9** (47%) | 300 → **239** (80%) | 403 → **239** |
| | glm-5.2 | 121 → **55.2** (46%) | 300 → **219** (73%) | 403 → **252** |
| | dsv2-lite shared | 121 → **49.4** (41%) | 300 → **202** (67%) | 403 → **245** |
| | dsv2-lite routed | 121 → **49.3** (41%) | 300 → **218** (73%) | 403 → **226** |
| | qwen3-30b-a3b | 121 → **48.0** (40%) | 300 → **149** (50%) | 403 → **321** |
| L40S (g6e) | mixtral-8x22b | 362 → **196** (54%) | 864 → **696** (81%) | 419 → **282** |
| | glm-5.2 | 362 → **184** (51%) | 864 → **614** (71%) | 419 → **299** |
| | dsv2-lite shared | 362 → **167** (46%) | 864 → **506** (59%) | 419 → **329** |
| | dsv2-lite routed | 362 → **165** (46%) | 864 → **287** (33%) | 419 → **574** |
| | qwen3-30b-a3b | 362 → **159** (44%) | 864 → **159** (18%) | 419 → **998**† |

† qwen3@L40S: effective Ridge Point(998)가 M_c(542)를 넘는다 — 즉 M을 아무리
키워도 **compute-bound가 영원히 될 수 없다**. knee(1333)에서 latency 상승이
시작되긴 하지만 이는 tensor core 포화가 아니라 efficiency-bound 상승이다.

### Crossover M* — 이론 → 실측 (full chain)

이론값은 `M* = R·M_c/(M_c − R)` (datasheet Ridge Point 기준). **실측값은
knee 그 자체** — latency가 바닥(1.5×floor)을 떠나는 M이며, memory-bound에서
벗어나기 시작하는 지점의 empirical 정의다. analyze.py의 `M*_calc`
(effective ridge 기반 닫힌식)와 병기해 교차 확인할 수 있다.

| model | T4 | A10G | L4 | L40S |
| --- | --- | --- | --- | --- |
| mixtral-8x22b | 219 → **72** | 122 → **167** | 473 → **184** | 495 → **309** |
| glm-5.2 | 234 → **87** | 126 → **134** | 547 → **247** | 576 → **347** |
| deepseek-v2-lite shared | 270 → **129** | 136 → **103** | 790 → **297** | 852 → **217** |
| deepseek-v2-lite routed | 288 → **75** | 141 → **245** | 972 → **259** | 1068 → **553** |
| qwen3-30b-a3b | 325 → **196** | 149 → **415** | 1575 → **531** | 1846 → **1333**† |

최신 수치는 `python analyze.py`로 재생성 (KNEE_FACTOR=1.5는 analyze.py 상단
상수).

### EP 해석

expert당 기대 토큰 수는 `M ≈ B × top_k / num_experts` (uniform routing).
따라서 GPU에 모이는 배치 B가 `M* × num_experts / top_k`를 넘어야 routed
expert가 compute-bound 영역에서 동작한다. shared expert는 M = B이므로 훨씬
작은 배치에서 이미 compute-bound가 된다.
