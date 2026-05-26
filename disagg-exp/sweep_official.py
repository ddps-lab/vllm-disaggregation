#!/usr/bin/env python3
"""
Wrapper around vLLM's official `vllm bench serve` (formerly benchmark_serving.py).

Drives the same workload grid as sweep.py but delegates each point to the
official benchmark, then writes its result JSON under EXP_LOG_DIR/<config>/.

Usage:
    python disagg-exp/sweep_official.py --config A1 --base-url http://localhost:8000

Env overrides (same names as sweep.py):
    # Note: PREFILL_LENS and DECODE_LENS combinations are now restricted to
    # exactly 3 defined pairs to prevent combinatorial explosion:
    # (2048, 128), (1024, 512), (128, 2048)
    SWEEP_RATES=1.0,2.0,4.0
    SWEEP_NUM_PROMPTS=300         total prompts per point (warmup 10 + measured 290)
    EXP_LOG_DIR=./results
    S3_BUCKET=hdjung-disaggregation-result    ""=disable

Per-point output:
    $EXP_LOG_DIR/<config>/p{prefill}_d{decode}_r{rate}.json   official benchmark result
    $EXP_LOG_DIR/<config>/.done_{point_id}                     completion marker
    $EXP_LOG_DIR/<config>/.failed_{point_id}                   failure marker
"""

import argparse
import asyncio
import datetime as _dt
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import aiohttp

# ── grid ──────────────────────────────────────────────────────────────────────
def _parse_list(env_key: str, default: list[float]) -> list[float]:
    raw = os.environ.get(env_key, "")
    if raw:
        return [float(x) for x in raw.split(",")]
    return default

PD_PAIRS = [
    (2048, 128),
    (1024, 512),
    (128, 2048),
]
RATES = _parse_list("SWEEP_RATES", [0.5, 1.0, 2.0])

NUM_PROMPTS = int(os.environ.get("SWEEP_NUM_PROMPTS", "300"))
WARMUP_N    = int(os.environ.get("SWEEP_WARMUP_N",   "10"))   # not enforced server-side; analyze can skip

LOG_DIR = os.environ.get("EXP_LOG_DIR", "./results")
MODEL_NAME = "llama-3.1-8b"  # must match --served-model-name on the server


# ── S3 sync (best-effort, optional) ──────────────────────────────────────────
def start_s3_sync(bucket: str, config: str, interval: int = 30) -> threading.Event | None:
    """Background sync of LOG_DIR → s3://{bucket}/raw/official/{date}/{host}/{config}/.

    "official" prefix → sweep.py(custom) 결과와 구분.
    config 한 단계 더 → 같은 호스트에서 여러 config 결과 보관해도 안 섞임.
    Returns the stop Event, or None if s5cmd is missing or bucket is empty.
    """
    if not bucket:
        print("[s3] sync disabled (empty bucket)")
        return None
    if shutil.which("s5cmd") is None:
        print("[s3] s5cmd not found on PATH — S3 sync disabled")
        return None

    host = socket.gethostname()
    date = _dt.datetime.utcnow().strftime("%Y%m%d")
    dest = f"s3://{bucket}/raw/official/{date}/{host}/{config}/"
    stop = threading.Event()

    def _loop() -> None:
        while not stop.is_set():
            try:
                subprocess.run(
                    ["s5cmd", "sync", f"{LOG_DIR}/", dest],
                    check=False, capture_output=True, timeout=120,
                )
            except Exception as exc:
                print(f"[s3] sync error: {exc}", flush=True)
            stop.wait(interval)
        # final flush
        try:
            subprocess.run(
                ["s5cmd", "sync", f"{LOG_DIR}/", dest],
                check=False, capture_output=True, timeout=300,
            )
        except Exception:
            pass

    t = threading.Thread(target=_loop, daemon=True, name="s3-sync")
    t.start()
    print(f"[s3] syncing {LOG_DIR}/ → {dest} every {interval}s")
    return stop


# ── health check (copied behavior from sweep.py) ──────────────────────────────
async def wait_for_health(base_url: str, timeout_s: int = 600) -> None:
    """
    Wait for the server to be ready. 
    Note: The FastAPI proxy lacks a dedicated /health endpoint. 
    Thus, 404 (Not Found) or 405 (Method Not Allowed) responses 
    indicate that the server process is alive and successfully responding.
    """
    deadline = time.time() + timeout_s
    print(f"Waiting for {base_url}/health ...", flush=True)
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        while time.time() < deadline:
            try:
                async with session.get(
                    f"{base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status in (200, 404, 405):
                        print("  server ready.", flush=True)
                        return
            except Exception:
                pass
            await asyncio.sleep(5)
    raise RuntimeError(f"Server at {base_url} not healthy after {timeout_s}s")


# ── one benchmark invocation ─────────────────────────────────────────────────
def run_one(
    repo_root: Path,
    base_url: str,
    config: str,
    prefill_len: int,
    decode_len: int,
    rate: float,
    result_dir: Path,
    point_id: str,
    extra_body_min: bool,
) -> tuple[bool, dict | None]:
    """Spawn `vllm bench serve` for one (prefill, decode, rate) point.

    Returns (ok, parsed_result_dict_or_None).
    """
    result_filename = f"{point_id}.json"
    result_path = result_dir / result_filename

    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--base-url", base_url,
        "--endpoint", "/v1/completions",
        "--model", MODEL_NAME,
        "--tokenizer", "meta-llama/Llama-3.1-8B-Instruct",
        "--dataset-name", "random",
        "--random-input-len", str(prefill_len),
        "--random-output-len", str(decode_len),
        "--random-range-ratio", "0.0",       # exact fixed length
        "--num-prompts", str(NUM_PROMPTS),
        "--request-rate", str(rate),
        "--burstiness", "1.0",               # Poisson
        "--ignore-eos",
        "--seed", "0",
        # Greedy + 모델 generation_config.json (temp=0.6, top_p=0.9) override.
        # 벤치마크는 deterministic 해야 비교가 의미 있으므로 temperature=0 강제.
        # top_p=1.0 은 temperature=0 일 때 효과 없지만 메타로 박아둠.
        "--extra-body", json.dumps({"temperature": 0, "top_p": 1.0}),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--save-result",
        "--save-detailed",
        "--result-dir", str(result_dir),
        "--result-filename", result_filename,
        # record context in result JSON
        "--metadata",
        f"config={config}",
        f"prefill_len={prefill_len}",
        f"decode_len={decode_len}",
        f"rate={rate}",
        f"point_id={point_id}",
    ]

    # Force exact output length: ask the server to keep generating to decode_len.
    # vLLM's RandomDataset already sets max_tokens=output_len.
    # We rely on --ignore-eos to guarantee we hit the target decode_len.
    # (Do not pass min_tokens; the proxy overrides max_tokens=1 on the prefill side
    # causing a 400 Bad Request if min_tokens > max_tokens).
    # if extra_body_min:
    #     cmd += ["--extra-body", json.dumps({"min_tokens": decode_len})]

    log_path = result_dir / f"{point_id}.log"
    print(f"[run] {point_id} → {' '.join(cmd[:6])} ... (logs: {log_path})", flush=True)

    import shlex
    cmd_str = f"{shlex.join(cmd)} 2>&1 | tee {log_path}"
    try:
        proc = subprocess.run(
            cmd_str, shell=True, executable="/bin/bash", check=False,
            cwd=str(repo_root), timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 3600s", flush=True)
        return False, None
    except Exception as exc:
        print(f"  ERROR launching subprocess: {exc}", flush=True)
        return False, None

    if proc.returncode != 0:
        print(f"  exit code {proc.returncode} — see {log_path}", flush=True)
        return False, None

    if not result_path.exists():
        print(f"  no result JSON at {result_path}", flush=True)
        return False, None

    try:
        with open(result_path) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  failed to parse {result_path}: {exc}", flush=True)
        return False, None

    return True, data


def summarize(data: dict) -> str:
    """One-line summary of an official benchmark result."""
    keys = [
        ("completed", "completed"),
        ("request_throughput", "rps"),
        ("output_throughput", "tok/s"),
        ("p50_ttft_ms", "ttft_p50_ms"),
        ("p99_ttft_ms", "ttft_p99_ms"),
        ("p50_tpot_ms", "tpot_p50_ms"),
        ("p99_tpot_ms", "tpot_p99_ms"),
    ]
    parts = []
    for k, label in keys:
        if k in data and data[k] is not None:
            try:
                parts.append(f"{label}={float(data[k]):.2f}")
            except (TypeError, ValueError):
                parts.append(f"{label}={data[k]}")
    return "  " + " ".join(parts)


# ── main ──────────────────────────────────────────────────────────────────────
def build_grid() -> list[tuple[int, int, float]]:
    points: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int, float]] = set()

    def _add(pl: int, dl: int, r: float) -> None:
        key = (pl, dl, r)
        if key not in seen:
            seen.add(key)
            points.append(key)

    for pl, dl in PD_PAIRS:
        for r in RATES:
            _add(pl, dl, r)
    return points


async def main(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    config = args.config

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(LOG_DIR) / config
    out_dir.mkdir(parents=True, exist_ok=True)

    # `vllm` CLI is required.
    if shutil.which("vllm") is None:
        print("ERROR: `vllm` CLI not on PATH. Activate the venv first.", file=sys.stderr)
        sys.exit(2)

    await wait_for_health(base_url, timeout_s=args.health_timeout)

    # Optional S3 sync.
    bucket = args.s3_bucket if args.s3_bucket is not None else os.environ.get(
        "S3_BUCKET", "hdjung-disaggregation-result"
    )
    s3_stop = start_s3_sync(bucket, config, interval=int(os.environ.get("S3_SYNC_INTERVAL", "30")))

    points = build_grid()
    print(f"Grid: {len(points)} points × num_prompts={NUM_PROMPTS}", flush=True)

    n_done = 0
    n_skip = 0
    n_fail = 0

    try:
        for prefill_len, decode_len, rate in points:
            point_id = f"p{prefill_len}_d{decode_len}_r{rate}"
            marker_done   = out_dir / f".done_{point_id}"
            marker_failed = out_dir / f".failed_{point_id}"

            if marker_done.exists():
                n_skip += 1
                continue

            print(f"\n[{n_done + n_fail + 1}/{len(points)}] {point_id}", flush=True)
            ok, data = run_one(
                repo_root=repo_root,
                base_url=base_url,
                config=config,
                prefill_len=prefill_len,
                decode_len=decode_len,
                rate=rate,
                result_dir=out_dir,
                point_id=point_id,
                extra_body_min=not args.no_min_tokens,
            )

            if ok and data is not None:
                marker_done.touch()
                marker_failed.unlink(missing_ok=True)
                print(summarize(data), flush=True)
                n_done += 1
            else:
                marker_failed.touch()
                n_fail += 1
                if args.stop_on_failure:
                    print("--stop-on-failure set; aborting sweep.", flush=True)
                    break

    finally:
        if s3_stop is not None:
            s3_stop.set()

    print(
        f"\nDone. {n_done} ran, {n_skip} skipped (already done), {n_fail} failed.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="A1|A2|A3|B|C1|C2|D|test|...")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--health-timeout", type=int, default=600,
        help="Seconds to wait for /health before aborting sweep.",
    )
    parser.add_argument(
        "--s3-bucket", default=None,
        help="Override S3_BUCKET env. Pass an empty string to disable.",
    )
    parser.add_argument(
        "--no-min-tokens", action="store_true",
        help="Do not pass min_tokens via --extra-body (some servers may reject it).",
    )
    parser.add_argument(
        "--stop-on-failure", action="store_true",
        help="Abort the whole sweep on the first failed point.",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
