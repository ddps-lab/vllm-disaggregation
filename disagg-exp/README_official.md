# Disagg-Exp — Official benchmark variant

This branch adds two files that run the same Tier-1 grid using vLLM's
**official** `vllm bench serve` (formerly `benchmarks/benchmark_serving.py`)
as the workload driver. Nothing else in `disagg-exp/` changes — `setup.sh`,
`launch_configs.sh`, `instrumented_connector.py`, `sweep.py`, and `analyze.py`
are untouched.

## Why two drivers?

`sweep.py` measures from the client side (Poisson arrivals, 2-phase
warmup/measured, custom SSE parser). The official benchmark uses the same
arrival model but a different request engine, prompt sampler, and metric
aggregator. Running both lets us:

- **Cross-validate** our custom client against the upstream baseline.
- Get **upstream-comparable numbers** for any external citation.
- Surface any client-side bottleneck specific to `sweep.py`.

## Files

| File | Role |
|---|---|
| `sweep_official.py` | Grid wrapper around `vllm bench serve` |
| `analyze_official.py` | Reads benchmark-result JSONs → table / plots / cross-check |

## Quick start

Assume `setup.sh` has already been run on the node and a vLLM server is up.

```bash
source .venv/bin/activate
export EXP_LOG_DIR=/home/ubuntu/exp-logs

# small smoke run
SWEEP_PREFILL_LENS=512 SWEEP_DECODE_LENS=128 SWEEP_RATES=1.0 \
SWEEP_NUM_PROMPTS=20 \
python disagg-exp/sweep_official.py --config A1 --base-url http://localhost:8000

# full grid
python disagg-exp/sweep_official.py --config A1 --base-url http://localhost:8000

# table + plots
python disagg-exp/analyze_official.py --log-dir $EXP_LOG_DIR --configs A1 B C1 D --plot

# side-by-side with the custom sweep.py output for the same config
python disagg-exp/analyze_official.py --log-dir $EXP_LOG_DIR \
    --configs A1 --compare-custom $EXP_LOG_DIR
```

## How `sweep_official.py` invokes the benchmark

For each `(prefill_len, decode_len, rate)` it shells out to:

```
vllm bench serve \
  --backend openai \
  --base-url <base-url> \
  --endpoint /v1/completions \
  --model llama-3.1-8b \
  --dataset-name random \
  --random-input-len <prefill_len> \
  --random-output-len <decode_len> \
  --random-range-ratio 0.0 \
  --num-prompts <NUM_PROMPTS> \
  --request-rate <rate> \
  --burstiness 1.0 \
  --ignore-eos \
  --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --save-result --save-detailed \
  --result-dir  $EXP_LOG_DIR/<config>/ \
  --result-filename p{prefill}_d{decode}_r{rate}.json \
  --extra-body '{"min_tokens": <decode_len>}' \
  --disable-tqdm \
  --metadata config=... point_id=...
```

Notes on the chosen flags:

- `--random-range-ratio 0.0` → fixed prompt length (no jitter), so the
  prefill cost is exactly `random-input-len`.
- `--ignore-eos` keeps the model generating until `max_tokens`.
- `--extra-body '{"min_tokens": N}'` forces decode length; pass
  `--no-min-tokens` to `sweep_official.py` if the running server rejects
  the field.
- `--save-detailed` writes per-request `ttfts`/`itls`/`output_lens` arrays
  in the result JSON, which `analyze_official.py` uses to recompute
  percentiles after dropping the first N requests as warmup.

## Env knobs (mirrors `sweep.py`)

| Variable | Default | Meaning |
|---|---|---|
| `SWEEP_PREFILL_LENS` | `512,2048,8192` | comma-sep list |
| `SWEEP_DECODE_LENS` | `128,512,1024,4096` | comma-sep list |
| `SWEEP_RATES` | `1.0,2.0,4.0` | requests/sec |
| `SWEEP_FIXED_DECODE` | `512` | decode held fixed in Cross 1 |
| `SWEEP_FIXED_PREFILL` | `2048` | prefill held fixed in Cross 2 |
| `SWEEP_NUM_PROMPTS` | `350` | warmup 50 + measured 300 |
| `SWEEP_WARMUP_N` | `50` | (analyzer only — server has no notion of warmup) |
| `EXP_LOG_DIR` | `./results` | output root |
| `S3_BUCKET` | `hdjung-disaggregation-result` | empty string to disable |
| `S3_SYNC_INTERVAL` | `30` | seconds |
| `ANALYZE_WARMUP_SKIP` | `50` | passed to `analyze_official.py --skip-warmup` |

## Output layout

```
$EXP_LOG_DIR/
└── <config>/
    ├── p512_d512_r1.0.json   <- vllm bench serve result (rich)
    ├── p512_d512_r1.0.log    <- stdout/stderr of the subprocess
    ├── .done_p512_d512_r1.0  <- resume marker
    └── .failed_p..._r4.0     <- failure marker (delete to retry)
```

`analyze_official.py` reads the `*.json` files. Resume: re-run the same
command; points with a `.done_*` marker are skipped.

## Cross-check workflow

```bash
# 1) run custom sweep.py (existing flow)
python disagg-exp/sweep.py --config A1 --base-url http://localhost:8000

# 2) run official wrapper on the same server
python disagg-exp/sweep_official.py --config A1 --base-url http://localhost:8000

# 3) compare numbers side-by-side
python disagg-exp/analyze_official.py --log-dir $EXP_LOG_DIR \
    --configs A1 --compare-custom $EXP_LOG_DIR
```

The custom JSONL (`*.jsonl`) and the official JSON (`*.json`) coexist in
the same `<config>/` directory — they have different extensions so neither
overwrites the other.

## Gotchas

- **CLI presence**: `vllm` must be on `PATH`. `sweep_official.py` exits with
  code 2 if it isn't. Activate the venv from `setup.sh` first.
- **min_tokens support**: if the server build does not accept
  `min_tokens` (some upstreams drop it), pass `--no-min-tokens`. Then
  responses will stop at the model's natural EOS — even with `--ignore-eos`,
  some adapters truncate early. In that case `analyze_official.py` will
  show output_lens shorter than `decode_len`; flag it in the cross-check.
- **Warmup**: the official benchmark has `--num-warmups`, but it's a fixed
  pre-stream that runs *before* timing starts and is not surfaced in the
  result JSON. We instead set `--num-prompts 350` and let
  `analyze_official.py --skip-warmup 50` drop the leading 50 from the
  per-request arrays. To use the benchmark's own warmup instead, pass
  `--skip-warmup 0` to the analyzer.
- **`--random-range-ratio` type**: it's `str` (accepts a float or a JSON
  dict). We pass `"0.0"` for both ISL and OSL.
- **Subprocess timeout**: each point caps at 3600s. Long high-rate points
  on weak GPUs can exceed this; bump in `run_one()` if needed.
