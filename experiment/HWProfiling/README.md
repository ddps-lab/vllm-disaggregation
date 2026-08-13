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

## Bandwidth ramp — bwprofile.py

GEMM과 무관한 순수 스트리밍(`c = a + b`, `[N, 2048]`, N 2배씩)으로 "working set
크기에 따라 DRAM 대역폭이 얼마나 열리는가"를 측정한다 (`python bwprofile.py`,
그림은 `python bwplot.py` → `results/bwprofile/bandwidth_ramp.png`).

| GPU | plateau (peak 대비) | plateau의 80% 도달 크기 | launch 바닥 (t 상수 구간) |
| --- | --- | --- | --- |
| T4 (320) | 244 GB/s (76%) | ~8 MB | ~19 µs |
| A10G (600) | 486 GB/s (81%) | **~25 MB** | ~20 µs |
| L4 (300) | 232 GB/s (77%) | **~5 MB** | ~15 µs |
| L40S (864) | 670 GB/s (78%) | ~20 MB | ~14 µs |

3국면: **launch 바닥**(~0.8MB까지, t 상수·GB/s ∝ 크기) → **ramp** → **plateau**
(peak의 76~81% — mixtral GEMM에서 뽑은 BW_eff와 일치, 커널 종류와 무관한 실질
DRAM 한계). 대역폭이 높은 GPU일수록 ramp가 길다 (A10G 25MB vs L4 5MB).

이 ramp에 expert를 대입할 때는 **커널별 접근 bytes로 따로** 봐야 한다 — chain은
3개 커널이 각자 접근하므로 총합이 아니라 커널 단위가 ramp의 x축에 해당한다.
M=1(바닥) 기준, 괄호는 합산 내역:

| model | w13 커널 | w2 커널 | silu 커널 |
| --- | --- | --- | --- |
| mixtral-8x22b | 402.8 MB (W13 402.7 MB + x 12 KB + h 66 KB) | 201.4 MB (W2 201.3 MB + a 33 KB + y 12 KB) | 98 KB (h 66 + a 33) |
| glm-5.2 | 50.4 MB (W13 50.3 MB + x 12 KB + h 8 KB) | 25.2 MB (W2 25.2 MB + a 4 KB + y 12 KB) | 12 KB |
| dsv2-lite shared | 23.1 MB (W13 23.1 MB + …) | 11.6 MB (W2 11.5 MB + …) | 17 KB |
| dsv2-lite routed | 11.5 MB (W13 11.5 MB + x 4 KB + h 6 KB) | 5.8 MB (W2 5.8 MB + a 3 KB + y 4 KB) | 8 KB |
| qwen3-30b-a3b | **6.3 MB** (W13 6.29 MB + x 4 KB + h 3 KB) | **3.2 MB** (W2 3.15 MB + a 1.5 KB + y 4 KB) | 4.6 KB |

읽는 법: mixtral의 두 GEMM(403/201MB)은 모든 GPU에서 plateau 깊숙이 있어
대역폭을 제값(76~81%)으로 받는다. qwen3의 두 GEMM(6.3/3.2MB)은 A10G·L40S
ramp의 중턱(peak의 ~45~50% 구간)에 앉아 있어 **완벽한 스트리밍 커널이었어도
대역폭 절반을 못 쓰는 크기**다 — "BW를 활용할 크기에 도달하기 전에
(activation 증가로 AI가 올라) compute 쪽으로 넘어간다"의 정량 근거.
M이 커지면 activation 항(x/h/a/y)이 커널 bytes를 키워 ramp 위로 올라가지만,
그때는 이미 AI도 함께 올라 병목의 주인이 바뀐다.

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

### 사례 분석: mixtral vs qwen3 — 같은 GPU, 전혀 다른 국면 진행

가장 대조적인 두 모델(chain, 전 측정점). GB/s·TFLOPS는 `bytes/t`, `flops/t`로
계산한 implied 값, AI = flops/bytes [FLOP/byte]. bytes 열의 괄호는 **3개 커널의
개별 접근량**(w13 + silu + w2, 단위 MB / G=GB) — chain은 한 번에 접근하는 게
아니라 커널 3개가 각자 접근하므로, bandwidth ramp에 대응시킬 때는 괄호 안의
커널 단위 수치를 봐야 한다.

**A10G (peak 600 GB/s · 70 TF) — mixtral-8x22b**

| M | bytes: total (w13 + silu + w2), MB | FLOPs | t (ms) | GB/s | TFLOPS | AI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 604.2 (402.7 + 0.1 + 201.4) | 0.6 GF | 1.364 | 443 | 0.4 | 1 |
| 2 | 604.4 (402.8 + 0.2 + 201.4) | 1.2 GF | 1.312 | 461 | 0.9 | 2 |
| 4 | 604.9 (403.0 + 0.4 + 201.5) | 2.4 GF | 1.316 | 460 | 1.8 | 4 |
| 8 | 605.7 (403.3 + 0.8 + 201.7) | 4.8 GF | 1.309 | 463 | 3.7 | 8 |
| 16 | 607.5 (403.9 + 1.6 + 202.0) | 9.7 GF | 1.323 | 459 | 7.3 | 16 |
| 32 | 611.1 (405.1 + 3.1 + 202.8) | 19.3 GF | 1.422 | 430 | 13.6 | 32 |
| 64 | 618.1 (407.6 + 6.3 + 204.2) | 38.7 GF | 1.440 | 429 | 26.8 | 63 |
| 128 | 632.3 (412.6 + 12.6 + 207.1) | 77.3 GF | 1.574 | 402 | 49.1 | 122 |
| 256 | 660.6 (422.6 + 25.2 + 212.9) | 154.6 GF | 2.795 | 236 | 55.3 | 234 |
| 512 | 717.2 (442.5 + 50.3 + 224.4) | 309.3 GF | 4.952 | 145 | 62.5 | 431 |
| 1024 | 830.5 (482.3 + 100.7 + 247.5) | 618.6 GF | 9.445 | 88 | 65.5 | 745 |
| 2048 | 1.06G (562.0 + 201.3 + 293.6) | 1.24 TF | 18.729 | 56 | 66.1 | 1170 |
| 4096 | 1.51G (721.4 + 402.7 + 385.9) | 2.47 TF | 37.390 | 40 | 66.2 | 1639 |
| 8192 | 2.42G (1.04G + 805.3 + 570.4) | 4.95 TF | 74.264 | 33 | 66.6 | 2048 |
| 16384 | 4.23G (1.68G + 1.61G + 939.5) | 9.90 TF | 146.915 | 29 | 67.4 | 2341 |
| 32768 | 7.85G (2.95G + 3.22G + 1.68G) | 19.79 TF | 292.581 | 27 | 67.7 | 2521 |
| 65536 | 15.10G (5.50G + 6.44G + 3.15G) | 39.59 TF | 636.114 | 24 | 62.2 | 2622 |
| 131072 | 29.60G (10.60G + 12.88G + 6.11G) | 79.18 TF | 1294.765 | 23 | 61.2 | 2675 |

**A10G — qwen3-30b-a3b**

| M | bytes: total (w13 + silu + w2), MB | FLOPs | t (ms) | GB/s | TFLOPS | AI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.5 (6.3 + 0.0 + 3.2) | 0.0 GF | 0.068 | 140 | 0.1 | 1 |
| 2 | 9.5 (6.3 + 0.0 + 3.2) | 0.0 GF | 0.071 | 134 | 0.3 | 2 |
| 4 | 9.5 (6.3 + 0.0 + 3.2) | 0.0 GF | 0.076 | 125 | 0.5 | 4 |
| 8 | 9.6 (6.3 + 0.0 + 3.2) | 0.1 GF | 0.075 | 128 | 1.0 | 8 |
| 16 | 9.7 (6.4 + 0.1 + 3.2) | 0.2 GF | 0.075 | 130 | 2.0 | 16 |
| 32 | 10.0 (6.5 + 0.1 + 3.3) | 0.3 GF | 0.075 | 134 | 4.0 | 30 |
| 64 | 10.6 (6.8 + 0.3 + 3.5) | 0.6 GF | 0.074 | 143 | 8.2 | 57 |
| 128 | 11.7 (7.2 + 0.6 + 3.9) | 1.2 GF | 0.068 | 173 | 17.9 | 104 |
| 256 | 13.9 (8.1 + 1.2 + 4.6) | 2.4 GF | 0.078 | 179 | 31.1 | 174 |
| 512 | 18.4 (10.0 + 2.4 + 6.0) | 4.8 GF | 0.114 | 161 | 42.5 | 263 |
| 1024 | 27.3 (13.6 + 4.7 + 8.9) | 9.7 GF | 0.211 | 129 | 45.8 | 355 |
| 2048 | 45.1 (21.0 + 9.4 + 14.7) | 19.3 GF | 0.476 | 95 | 40.6 | 429 |
| 4096 | 80.7 (35.7 + 18.9 + 26.2) | 38.7 GF | 0.795 | 102 | 48.7 | 479 |
| 8192 | 152.0 (65.0 + 37.7 + 49.3) | 77.3 GF | 1.361 | 112 | 56.8 | 509 |
| 16384 | 294.6 (123.7 + 75.5 + 95.4) | 154.7 GF | 2.613 | 113 | 59.2 | 525 |
| 32768 | 579.9 (241.2 + 151.0 + 187.7) | 309.4 GF | 5.152 | 113 | 60.1 | 534 |
| 65536 | 1.15G (476.1 + 302.0 + 372.2) | 618.7 GF | 10.119 | 114 | 61.1 | 538 |
| 131072 | 2.29G (945.8 + 604.0 + 741.3) | 1.24 TF | 20.029 | 114 | 61.8 | 540 |
| 262144 | 4.57G (1.89G + 1.21G + 1.48G) | 2.47 TF | 39.963 | 114 | 61.9 | 541 |
| 524288 | 9.14G (3.76G + 2.42G + 2.96G) | 4.95 TF | 79.854 | 114 | 62.0 | 542 |
| 1048576 | 18.26G (7.52G + 4.83G + 5.91G) | 9.90 TF | 159.523 | 114 | 62.1 | 542 |

**L40S (peak 864 GB/s · 362 TF) — mixtral-8x22b**

| M | bytes: total (w13 + silu + w2), MB | FLOPs | t (ms) | GB/s | TFLOPS | AI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 604.2 (402.7 + 0.1 + 201.4) | 0.6 GF | 0.820 | 737 | 0.7 | 1 |
| 2 | 604.4 (402.8 + 0.2 + 201.4) | 1.2 GF | 0.863 | 700 | 1.4 | 2 |
| 4 | 604.9 (403.0 + 0.4 + 201.5) | 2.4 GF | 0.863 | 701 | 2.8 | 4 |
| 8 | 605.7 (403.3 + 0.8 + 201.7) | 4.8 GF | 0.867 | 698 | 5.6 | 8 |
| 16 | 607.5 (403.9 + 1.6 + 202.0) | 9.7 GF | 0.873 | 696 | 11.1 | 16 |
| 32 | 611.1 (405.1 + 3.1 + 202.8) | 19.3 GF | 0.961 | 636 | 20.1 | 32 |
| 64 | 618.1 (407.6 + 6.3 + 204.2) | 38.7 GF | 0.972 | 636 | 39.8 | 63 |
| 128 | 632.3 (412.6 + 12.6 + 207.1) | 77.3 GF | 1.012 | 625 | 76.4 | 122 |
| 256 | 660.6 (422.6 + 25.2 + 212.9) | 154.6 GF | 1.101 | 600 | 140.5 | 234 |
| 512 | 717.2 (442.5 + 50.3 + 224.4) | 309.3 GF | 1.661 | 432 | 186.2 | 431 |
| 1024 | 830.5 (482.3 + 100.7 + 247.5) | 618.6 GF | 3.074 | 270 | 201.2 | 745 |
| 2048 | 1.06G (562.0 + 201.3 + 293.6) | 1.24 TF | 5.758 | 184 | 214.9 | 1170 |
| 4096 | 1.51G (721.4 + 402.7 + 385.9) | 2.47 TF | 12.398 | 122 | 199.6 | 1639 |
| 8192 | 2.42G (1.04G + 805.3 + 570.4) | 4.95 TF | 24.643 | 98 | 200.8 | 2048 |
| 16384 | 4.23G (1.68G + 1.61G + 939.5) | 9.90 TF | 49.841 | 85 | 198.6 | 2341 |
| 32768 | 7.85G (2.95G + 3.22G + 1.68G) | 19.79 TF | 102.090 | 77 | 193.9 | 2521 |
| 65536 | 15.10G (5.50G + 6.44G + 3.15G) | 39.59 TF | 208.488 | 72 | 189.9 | 2622 |
| 131072 | 29.60G (10.60G + 12.88G + 6.11G) | 79.18 TF | 493.191 | 60 | 160.5 | 2675 |
| 262144 | 58.59G (20.80G + 25.77G + 12.01G) | 158.35 TF | 991.049 | 59 | 159.8 | 2703 |

**L40S — qwen3-30b-a3b**

| M | bytes: total (w13 + silu + w2), MB | FLOPs | t (ms) | GB/s | TFLOPS | AI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.5 (6.3 + 0.0 + 3.2) | 0.0 GF | 0.059 | 159 | 0.2 | 1 |
| 2 | 9.5 (6.3 + 0.0 + 3.2) | 0.0 GF | 0.061 | 154 | 0.3 | 2 |
| 4 | 9.5 (6.3 + 0.0 + 3.2) | 0.0 GF | 0.062 | 153 | 0.6 | 4 |
| 8 | 9.6 (6.3 + 0.0 + 3.2) | 0.1 GF | 0.061 | 156 | 1.2 | 8 |
| 16 | 9.7 (6.4 + 0.1 + 3.2) | 0.2 GF | 0.061 | 158 | 2.5 | 16 |
| 32 | 10.0 (6.5 + 0.1 + 3.3) | 0.3 GF | 0.067 | 150 | 4.5 | 30 |
| 64 | 10.6 (6.8 + 0.3 + 3.5) | 0.6 GF | 0.062 | 170 | 9.8 | 57 |
| 128 | 11.7 (7.2 + 0.6 + 3.9) | 1.2 GF | 0.062 | 187 | 19.3 | 104 |
| 256 | 13.9 (8.1 + 1.2 + 4.6) | 2.4 GF | 0.076 | 183 | 31.9 | 174 |
| 512 | 18.4 (10.0 + 2.4 + 6.0) | 4.8 GF | 0.060 | 304 | 80.1 | 263 |
| 1024 | 27.3 (13.6 + 4.7 + 8.9) | 9.7 GF | 0.068 | 403 | 143.0 | 355 |
| 2048 | 45.1 (21.0 + 9.4 + 14.7) | 19.3 GF | 0.140 | 322 | 138.3 | 429 |
| 4096 | 80.7 (35.7 + 18.9 + 26.2) | 38.7 GF | 0.229 | 352 | 168.6 | 479 |
| 8192 | 152.0 (65.0 + 37.7 + 49.3) | 77.3 GF | 0.395 | 385 | 195.7 | 509 |
| 16384 | 294.6 (123.7 + 75.5 + 95.4) | 154.7 GF | 0.920 | 320 | 168.1 | 525 |
| 32768 | 579.9 (241.2 + 151.0 + 187.7) | 309.4 GF | 1.948 | 298 | 158.8 | 534 |
| 65536 | 1.15G (476.1 + 302.0 + 372.2) | 618.7 GF | 3.813 | 302 | 162.3 | 538 |
| 131072 | 2.29G (945.8 + 604.0 + 741.3) | 1.24 TF | 7.746 | 296 | 159.8 | 540 |
| 262144 | 4.57G (1.89G + 1.21G + 1.48G) | 2.47 TF | 16.187 | 283 | 152.9 | 541 |
| 524288 | 9.14G (3.76G + 2.42G + 2.96G) | 4.95 TF | 32.021 | 285 | 154.6 | 542 |
| 1048576 | 18.26G (7.52G + 4.83G + 5.91G) | 9.90 TF | 64.816 | 282 | 152.7 | 542 |
| 2097152 | 36.52G (15.04G + 9.66G + 11.81G) | 19.80 TF | 131.467 | 278 | 150.6 | 542 |

읽는 법 — 바닥(t가 M과 무관한 구간)의 정체 판별:

- **mixtral의 바닥은 진짜 bandwidth다**: M=1에서 604MB를 옮기는 시간이 그대로
  t이고, implied BW가 peak의 74~85%(A10G 443, L40S 737). t가 bytes에 비례.
- **qwen3의 바닥은 고정 오버헤드다**: 같은 GPU가 방금 443~737 GB/s를 증명했는데
  9.5MB에 60~78µs를 쓴다(= 140~160 GB/s로 보임). bytes +46%, FLOPs 256배가
  되어도(M=1→256) t가 그대로 — 어느 자원에도 비례하지 않는 상수 = launch/
  occupancy 바닥. **implied 값이 낮게 "보이는" 것과 그 자원이 병목인 것은
  다르다.**

국면 진행의 대조:

| 조합 | 국면 순서 | 끝 상태 |
| --- | --- | --- |
| mixtral @ A10G | bandwidth(74%) → compute | compute-bound, **94%** 도달 |
| mixtral @ L40S | bandwidth(85%) → compute | compute-bound, 55% 도달 |
| qwen3 @ A10G | overhead → **(bandwidth 국면 없음)** → compute | 88% 도달, GB/s는 62TF/542=114로 고정 |
| qwen3 @ L40S | overhead → activation streaming(~300GB/s) | **영원히 memory-bound**, TFLOPS=AI×GB/s로 종속 |

- qwen3@A10G에 bandwidth 국면이 없는 이유: 대역폭이 병목이 되려면 옮길 양이
  오버헤드를 압도해야 하는데 weight 9.5MB는 A10G에게 ~21µs 일감이라 조건이
  성립할 수 없고, activation이 커질 때쯤엔 이미 compute-bound다.
- qwen3@L40S의 두 곡선이 동형인 이유: AI가 M_c=542에 얼어붙은 뒤로는 독립
  곡선이 t(M) 하나뿐이고, GB/s와 TFLOPS는 같은 1/t에 다른 상수를 곱한 것.
- mixtral 초대형 M의 TFLOPS 하락(L40S 200→190, M=131072부터 −20%p)은 별개
  현상: 중간 텐서가 2^31 원소를 넘는 인덱싱 절벽 + (T4에서는) 지속 부하
  throttling. serving에서는 그 크기의 단일 GEMM을 던질 일이 없어 회피 가능.

### EP 해석

expert당 기대 토큰 수는 `M ≈ B × top_k / num_experts` (uniform routing).
따라서 GPU에 모이는 배치 B가 `M* × num_experts / top_k`를 넘어야 routed
expert가 compute-bound 영역에서 동작한다. shared expert는 M = B이므로 훨씬
작은 배치에서 이미 compute-bound가 된다.
