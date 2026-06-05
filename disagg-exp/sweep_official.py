#!/usr/bin/env python3
"""
Wrapper around vLLM's official `vllm bench serve` (formerly benchmark_serving.py).

Drives the same workload grid as sweep.py but delegates each point to the
official benchmark, then writes its result JSON under EXP_LOG_DIR/<config>/.

Usage:
    python disagg-exp/sweep_official.py --config A --base-url http://localhost:8000 \
        --decode-metrics-url http://<decode-host>:8200/metrics

Env overrides (same names as sweep.py):
    # Note: PREFILL_LENS and DECODE_LENS combinations are now restricted to
    # exactly 3 defined pairs to prevent combinatorial explosion:
    # (2048, 64), (512, 512), (128, 1024)
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
import csv
import datetime as _dt
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import urlopen

import aiohttp

# ── grid ──────────────────────────────────────────────────────────────────────
def _parse_list(env_key: str, default: list[float]) -> list[float]:
    raw = os.environ.get(env_key, "")
    if raw:
        return [float(x) for x in raw.split(",")]
    return default

PD_PAIRS = [
    (2048, 64),
    (512, 512),
    (128, 1024),
]
RATES = _parse_list("SWEEP_RATES", [1.0, 2.0, 4.0])

NUM_PROMPTS = int(os.environ.get("SWEEP_NUM_PROMPTS", "300"))
WARMUP_N    = int(os.environ.get("SWEEP_WARMUP_N",   "10"))   # not enforced server-side; analyze can skip

LOG_DIR = os.environ.get("EXP_LOG_DIR", "./results")

# ── model identity ────────────────────────────────────────────────────────────
# Single source of truth for the model under test. Must be kept in sync with
# launch_configs.sh on the server side. Edit this block (not env vars) when
# switching models. These values are stamped into every result JSON's
# metadata so analyze_official.py / S3 archive can identify the run later.
MODEL_NAME = "qwen3-4b"                         # = --served-model-name on server
MODEL_PATH = "Qwen/Qwen3-4B"                    # HF id, used as bench --tokenizer
MODEL_DTYPE = "half"                            # half | bfloat16 | float16

# ── run identity ─────────────────────────────────────────────────────────────
# 양쪽 노드에서 export RUN_TAG=... 으로 같은 값 주면 S3 에서 한 폴더로 합쳐짐.
# 미지정 시 YYYYMMDD-HHMM (분 단위) — 같은 분에 양쪽 띄우면 폴더 일치, 다른 분이면
# 사후에 사람이 합쳐야 함.
RUN_TAG = os.environ.get("RUN_TAG") or _dt.datetime.now().strftime("%Y%m%d-%H%M")


# ── S3 sync (best-effort, optional) ──────────────────────────────────────────
def start_s3_sync(
    bucket: str,
    run_tag: str,
    host_ip: str,
    run_dir: Path,
    config_model: str,
    interval: int = 30,
) -> threading.Event | None:
    """Background sync of run_dir → S3.

    Path 구조:
        s3://{bucket}/raw/official/{run_tag}/{host_ip}/{config_model}/

    run_dir 의 system_logs/, results/ 가 그대로 mirror 된다. launch_configs.sh
    의 daemon 과 같은 위치로 동기화되지만 s5cmd 가 unchanged 파일은 skip 하므로
    redundant 해도 무해.
    """
    if not bucket:
        print("[s3] sync disabled (empty bucket)")
        return None
    if shutil.which("s5cmd") is None:
        print("[s3] s5cmd not found on PATH — S3 sync disabled")
        return None

    dest = f"s3://{bucket}/raw/official/{run_tag}/{host_ip}/{config_model}/"
    stop = threading.Event()

    def _loop() -> None:
        while not stop.is_set():
            try:
                subprocess.run(
                    ["s5cmd", "sync", f"{run_dir}/", dest],
                    check=False, capture_output=True, timeout=120,
                )
            except Exception as exc:
                print(f"[s3] sync error: {exc}", flush=True)
            stop.wait(interval)
        # final flush
        try:
            subprocess.run(
                ["s5cmd", "sync", f"{run_dir}/", dest],
                check=False, capture_output=True, timeout=300,
            )
        except Exception:
            pass

    t = threading.Thread(target=_loop, daemon=True, name="s3-sync")
    t.start()
    print(f"[s3] syncing {run_dir}/ → {dest} every {interval}s")
    return stop


# ── /metrics scraping ─────────────────────────────────────────────────────────
# Prometheus exposition format: lines look like
#   vllm:num_requests_running{model_name="..."} 12.0
# Some metrics (e.g. request_success) have multiple label combinations and emit
# one line per combination; for those we sum across all labels to get the
# instance-wide total.
_METRIC_RE = re.compile(r"^(vllm:[a-z_]+)\{[^}]*\}\s+([0-9eE+\-.]+)\s*$")

# Gauges: instantaneous state. Stored raw per sample.
_GAUGE_KEYS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)

# Counters: monotonic cumulative. Prometheus python client appends `_total` to
# Counter metrics on export. Stored raw per sample, then differenced in
# summary() to derive per-second rates.
_COUNTER_KEYS = (
    "vllm:prompt_tokens_total",       # tokens entering as prompt
    "vllm:generation_tokens_total",   # tokens emitted as completion
    "vllm:request_success_total",     # requests completed successfully
)

_METRIC_KEYS = _GAUGE_KEYS + _COUNTER_KEYS


def _scrape_once(url: str, timeout: float = 2.0) -> dict[str, float]:
    """Pull /metrics text and parse out the vllm:* values we care about.

    Multiple lines for the same metric (different labels) are summed so the
    returned value is instance-wide. Returns empty dict on any error so the
    scrape thread tolerates transient network blips without dying.
    """
    try:
        with urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if m and m.group(1) in _METRIC_KEYS:
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
            except ValueError:
                pass
    return out


class MetricsScraper(threading.Thread):
    """Polls Prefill+Decode /metrics endpoints during one benchmark point.

    On stop() it flushes a per-point CSV (time series) and returns a summary
    dict (mean/max/min/p50/p99 across samples). Designed to be lightweight —
    if an URL is empty/None that side is skipped.
    """

    def __init__(
        self,
        prefill_url: str,
        decode_url: str,
        out_csv: Path,
        interval: float = 1.0,
    ) -> None:
        super().__init__(daemon=True, name="metrics-scraper")
        self.prefill_url = prefill_url or ""
        self.decode_url = decode_url or ""
        self.out_csv = out_csv
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples: list[dict] = []

    def run(self) -> None:
        while not self.stop_event.is_set():
            row: dict = {"t": time.time()}
            if self.prefill_url:
                m = _scrape_once(self.prefill_url)
                for k in _METRIC_KEYS:
                    row[f"prefill.{k}"] = m.get(k)
            if self.decode_url:
                m = _scrape_once(self.decode_url)
                for k in _METRIC_KEYS:
                    row[f"decode.{k}"] = m.get(k)
            self.samples.append(row)
            self.stop_event.wait(self.interval)

        if not self.samples:
            return
        fields = list(self.samples[0].keys())
        try:
            with open(self.out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for row in self.samples:
                    w.writerow(row)
        except Exception as exc:
            print(f"[metrics] failed to write {self.out_csv}: {exc}", flush=True)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=10)

    def summary(self) -> dict:
        """Aggregate per-metric stats across samples.

        Gauges (running/waiting/kv_cache_usage) → mean/max/p50/p99 over the
        active period (samples where at least one side had requests in flight).

        Counters (prompt_tokens/generation_tokens/request_success) → rate
        derived from (last_active - first_active) / active_duration. This
        gives per-side RPS / prompt-TPS / generation-TPS for the run.
        """
        def _active(row: dict) -> bool:
            p = row.get("prefill.vllm:num_requests_running") or 0
            d = row.get("decode.vllm:num_requests_running") or 0
            return (p + d) > 0

        active = [s for s in self.samples if _active(s)]
        # If nothing was ever active (very short benchmark?), fall back to all.
        sample_set = active if active else self.samples

        def _stats(key: str) -> dict | None:
            vals = [s[key] for s in sample_set if s.get(key) is not None]
            if not vals:
                return None
            sv = sorted(vals)
            n = len(sv)
            return {
                "mean": sum(sv) / n,
                "max": sv[-1],
                "min": sv[0],
                "p50": sv[n // 2],
                "p99": sv[min(n - 1, int(n * 0.99))],
                "samples": n,
            }

        def _rate(key: str) -> dict | None:
            # Counters are cumulative; rate = delta / duration over active samples.
            pts = [
                (s["t"], s[key])
                for s in sample_set
                if s.get(key) is not None and s.get("t") is not None
            ]
            if len(pts) < 2:
                return None
            t0, v0 = pts[0]
            t1, v1 = pts[-1]
            duration = t1 - t0
            if duration <= 0:
                return None
            return {
                "delta": v1 - v0,
                "duration_s": duration,
                "rate_per_s": (v1 - v0) / duration,
            }

        out: dict[str, dict] = {}
        for side, url in (("prefill", self.prefill_url), ("decode", self.decode_url)):
            if not url:
                continue
            for k in _GAUGE_KEYS:
                s = _stats(f"{side}.{k}")
                if s is not None:
                    out[f"{side}.{k}"] = s
            for k in _COUNTER_KEYS:
                r = _rate(f"{side}.{k}")
                if r is not None:
                    out[f"{side}.{k}"] = r

        # Convenience fields for human-readable inline summary.
        # Decode side request_success is the most useful RPS (proxy of completed
        # end-user requests). prompt_tokens rate on Prefill = prefill TPS.
        # generation_tokens rate on Decode = decode TPS.
        def _get_rate(key: str) -> float | None:
            r = out.get(key)
            return r.get("rate_per_s") if isinstance(r, dict) and "rate_per_s" in r else None

        out["_derived"] = {
            "prefill_rps": _get_rate("prefill.vllm:request_success_total"),
            "prefill_prompt_tps": _get_rate("prefill.vllm:prompt_tokens_total"),
            "decode_rps": _get_rate("decode.vllm:request_success_total"),
            "decode_generation_tps": _get_rate("decode.vllm:generation_tokens_total"),
        }
        out["_meta"] = {
            "total_samples": len(self.samples),
            "active_samples": len(active),
            "interval_s": self.interval,
        }
        return out


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
    prefill_metrics_url: str = "",
    decode_metrics_url: str = "",
    metrics_interval: float = 1.0,
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
        "--tokenizer", MODEL_PATH,
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
        # 실제 model path + dtype. vllm bench serve의 model_id는
        # served-model-name(MODEL_NAME) 만 박혀서 dtype 변형을 구분 못 함.
        f"model_path={MODEL_PATH}",
        f"dtype={MODEL_DTYPE}",
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

    # Start /metrics scraper if either URL is configured.
    scraper: MetricsScraper | None = None
    if prefill_metrics_url or decode_metrics_url:
        scraper = MetricsScraper(
            prefill_url=prefill_metrics_url,
            decode_url=decode_metrics_url,
            out_csv=result_dir / f"{point_id}.metrics.csv",
            interval=metrics_interval,
        )
        scraper.start()

    import shlex
    cmd_str = f"{shlex.join(cmd)} 2>&1 | tee {log_path}"
    try:
        proc = subprocess.run(
            cmd_str, shell=True, executable="/bin/bash", check=False,
            cwd=str(repo_root), timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 3600s", flush=True)
        if scraper is not None:
            scraper.stop()
        return False, None
    except Exception as exc:
        print(f"  ERROR launching subprocess: {exc}", flush=True)
        if scraper is not None:
            scraper.stop()
        return False, None

    if scraper is not None:
        scraper.stop()
        try:
            summary = scraper.summary()
            with open(result_dir / f"{point_id}.metrics.json", "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as exc:
            print(f"  [metrics] summary write failed: {exc}", flush=True)

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

    # Run-scoped directory mirroring S3 structure:
    #   $LOG_DIR/{config}-{model_name}/results/
    # launch_configs.sh 가 같은 경로 (system_logs + results) 를 미리 만들어둠.
    run_tag = args.run_tag or RUN_TAG
    config_model = f"{config}-{MODEL_NAME}"
    run_dir = Path(LOG_DIR) / config_model
    out_dir = run_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model identity is hardcoded at the top of this file (single source of
    # truth). Print it here so the operator can sanity-check it matches what
    # launch_configs.sh is actually serving.
    print(
        f"[sweep] served_name={MODEL_NAME} model_path={MODEL_PATH} dtype={MODEL_DTYPE}",
        flush=True,
    )
    print(f"[sweep] run_tag={run_tag} run_dir={run_dir}", flush=True)

    # `vllm` CLI is required.
    if shutil.which("vllm") is None:
        print("ERROR: `vllm` CLI not on PATH. Activate the venv first.", file=sys.stderr)
        sys.exit(2)

    await wait_for_health(base_url, timeout_s=args.health_timeout)

    # Optional S3 sync. New path structure:
    #   raw/official/{run_tag}/{host_ip}/{config-model}/
    # host_ip 는 VLLM_HOST_IP env (이 노드의 private IP) — launch_configs.sh 와 일치.
    bucket = args.s3_bucket if args.s3_bucket is not None else os.environ.get(
        "S3_BUCKET", "hdjung-disaggregation-result"
    )
    host_ip = os.environ.get("VLLM_HOST_IP", socket.gethostname())
    s3_stop = start_s3_sync(
        bucket=bucket,
        run_tag=run_tag,
        host_ip=host_ip,
        run_dir=run_dir,
        config_model=config_model,
        interval=int(os.environ.get("S3_SYNC_INTERVAL", "30")),
    )

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
                prefill_metrics_url=args.prefill_metrics_url,
                decode_metrics_url=args.decode_metrics_url,
                metrics_interval=args.metrics_interval,
            )

            if ok and data is not None:
                marker_done.touch()
                marker_failed.unlink(missing_ok=True)
                print(summarize(data), flush=True)
                metrics_path = out_dir / f"{point_id}.metrics.json"
                if metrics_path.exists():
                    try:
                        with open(metrics_path) as f:
                            mdata = json.load(f)
                        # Line 1: per-side throughput (RPS + TPS).
                        d = mdata.get("_derived") or {}
                        def _fmt(v: float | None, unit: str) -> str:
                            return f"{v:.2f}{unit}" if isinstance(v, (int, float)) else "n/a"
                        tp_parts = []
                        if d.get("prefill_rps") is not None or d.get("prefill_prompt_tps") is not None:
                            tp_parts.append(
                                f"prefill rps={_fmt(d.get('prefill_rps'), '')} "
                                f"prompt_tps={_fmt(d.get('prefill_prompt_tps'), '')}"
                            )
                        if d.get("decode_rps") is not None or d.get("decode_generation_tps") is not None:
                            tp_parts.append(
                                f"decode rps={_fmt(d.get('decode_rps'), '')} "
                                f"gen_tps={_fmt(d.get('decode_generation_tps'), '')}"
                            )
                        if tp_parts:
                            print("  thru  — " + " | ".join(tp_parts), flush=True)
                        # Line 2: per-side batch size (running requests gauge).
                        batch_parts = []
                        for side in ("prefill", "decode"):
                            r = mdata.get(f"{side}.vllm:num_requests_running")
                            if r is not None:
                                batch_parts.append(
                                    f"{side} mean={r['mean']:.1f} "
                                    f"p99={r['p99']:.1f} max={r['max']:.0f}"
                                )
                        if batch_parts:
                            print("  batch — " + " | ".join(batch_parts), flush=True)
                    except Exception:
                        pass
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
    parser.add_argument("--config", required=True, help="A|test|... (결과 폴더 라벨; 현재 실험은 A = g6.xlarge P1D1, 첫 실험)")
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
    parser.add_argument(
        "--run-tag", default=os.environ.get("RUN_TAG", ""),
        help="실험 식별자 (S3 경로의 한 레벨). 양쪽 노드에서 같은 값을 export 하면 "
             "S3 한 폴더로 합쳐짐. 미지정 시 YYYYMMDD-HHMM (분 단위) 자동.",
    )
    parser.add_argument(
        "--prefill-metrics-url",
        default=os.environ.get("PREFILL_METRICS_URL", "http://127.0.0.1:8100/metrics"),
        help="Prometheus /metrics endpoint of the prefill vLLM instance. "
             "Empty string disables scraping for that side.",
    )
    parser.add_argument(
        "--decode-metrics-url",
        default=os.environ.get("DECODE_METRICS_URL", ""),
        help="Prometheus /metrics endpoint of the decode vLLM instance. "
             "For configA (cross-node PD), set this to http://<decode-host>:8200/metrics.",
    )
    parser.add_argument(
        "--metrics-interval", type=float,
        default=float(os.environ.get("METRICS_INTERVAL", "1.0")),
        help="Seconds between /metrics polls during each benchmark point.",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
