#!/usr/bin/env python3
"""
Client workload driver for disagg-exp tier-1.

Usage:
    python sweep.py --config A --base-url http://localhost:8000

Env overrides for the grid:
    SWEEP_PREFILL_LENS=512,2048,8192
    SWEEP_DECODE_LENS=128,512,1024,4096
    SWEEP_RATES=1.0,2.0,4.0

S3 sync (embedded — runs as a background thread while sweep is active):
    S3_BUCKET=hdjung-disaggregation-result   # default if --s3-bucket omitted
    S3_SYNC_INTERVAL=30                       # seconds between syncs
    --s3-bucket ""                            # disable

Each completed sweep point writes a JSONL file:
    $EXP_LOG_DIR/<config>/<point_id>.jsonl
"""

import argparse
import asyncio
import atexit
import datetime as _dt
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp

# ── grid (실험 조건표) ────────────────────────────────────────────────────────
# 벤치마크에서 테스트할 조건들을 정의합니다.
# 환경변수로 오버라이드 가능하며, 기본값은 아래와 같습니다.
# 이 조합들의 교차곱(Cross Product)이 실험의 전체 Grid를 구성합니다.

# [헬퍼] 환경변수에서 콤마 구분 텍스트를 파이썬 리스트로 변환 (예: "1.0,4.0" -> [1.0, 4.0])
def _parse_list(env_key: str, default: list[float]) -> list[float]:
    raw = os.environ.get(env_key, "")
    if raw:
        return [float(x) for x in raw.split(",")]
    return default

PREFILL_LENS = [int(x) for x in _parse_list("SWEEP_PREFILL_LENS", [512, 2048, 8192])]  # 질문 길이 (토큰 수)
DECODE_LENS  = [int(x) for x in _parse_list("SWEEP_DECODE_LENS",  [128, 512, 1024, 4096])]  # 답변 길이
RATES        = _parse_list("SWEEP_RATES", [1.0, 2.0, 4.0])  # 초당 요청 수 (QPS)

WARMUP_N   = int(os.environ.get("SWEEP_WARMUP_N",   "50"))   # 준비운동 요청 수
MEASURED_N = int(os.environ.get("SWEEP_MEASURED_N", "300"))  # 실전 측정 요청 수

# 웜업 단계에서 서버가 감당 못하면 실전을 스킵하는 기준
ABORT_FAIL_RATE  = float(os.environ.get("SWEEP_ABORT_FAIL_RATE",  "0.30"))  # 에러율 30% 초과 시
ABORT_TTFT_P99_S = float(os.environ.get("SWEEP_ABORT_TTFT_P99_S", "180.0"))  # TTFT p99가 3분 초과 시

LOG_DIR = os.environ.get("EXP_LOG_DIR", "./results")  # 모든 결과 파일의 저장 경로

MODEL_NAME = "llama-3.1-8b"  # launch_configs.sh의 --served-model-name과 반드시 일치해야 함


# ── request (요청 1개의 측정 결과를 담는 데이터 구조) ─────────────────────────
# JSONL 파일에 저장될 때 이 필드들이 그대로 JSON 키가 됩니다.
@dataclass
class Result:
    req_id: str
    phase: str          # "warmup"(준비운동) | "measured"(실전 데이터)
    prefill_len: int    # 요청한 질문 길이
    decode_len: int     # 요청한 답변 길이
    rate: float         # 목표 QPS
    send_ts: float      # 요청을 보낸 정확한 시각 (처리량 계산용)
    ttft_s: float | None       # 첫 토큰 도착 시간 (초) — 핵심 메트릭
    e2e_s: float | None        # 전체 응답 완료 시간 (초)
    prompt_tokens: int | None  # 서버가 실제 처리한 질문 토큰 수 (검증용)
    completion_tokens: int | None  # 서버가 실제 생성한 답변 토큰 수
    status: str         # "success" | "error" | "timeout"
    error: str | None


# ── 클라이언트 초시계 (API 요청 1개의 TTFT/E2E 측정) ────────────────────────
# vLLM 내부 메트릭은 PD 분리 시 전체 파이프라인(Prefill→KV전송→Decode)을 모르므로,
# 여기서(클라이언트) SSE 스트리밍으로 직접 측정합니다.
# 원리: stream=True로 요청하면 서버가 토큰을 하나 생성할 때마다 한 줄씩 보내줌.
#       → "첫 줄 도착 시점" = TTFT, "마지막 줄 도착 시점" = E2E
async def _do_request(
    session: aiohttp.ClientSession,
    base_url: str,
    req_id: str,
    phase: str,
    prefill_len: int,
    decode_len: int,
    rate: float,
) -> Result:
    # 토큰 ID를 직접 넣어서 정확히 prefill_len개의 입력을 보장 (문장 토크나이징 불확실성 제거)
    # 1부터 시작: 0번은 특수 토큰(BOS/PAD)이라 예상치 못한 동작 방지
    prompt_ids = list(range(1, prefill_len + 1))

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_ids,        # 토큰 ID 리스트로 직접 전달 (문자열 아님)
        "max_tokens": decode_len,    # 최대 생성 길이
        "min_tokens": decode_len,    # 최소 생성 길이 (max=min → 정확히 고정)
        "temperature": 0,            # 결정론적 생성 (재현성)
        "top_p": 1.0,
        "ignore_eos": True,          # EOS 토큰이 나와도 멈추지 않고 끝까지 생성
        "stream": True,              # SSE 스트리밍 활성화 (TTFT 측정의 핵심)
        "stream_options": {"include_usage": True},  # 마지막 청크에 토큰 수 포함
    }

    send_ts = time.time()  # ① 스톱워치 시작
    ttft_s: float | None = None
    e2e_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    try:
        # ② 서버에 HTTP POST 요청 전송 (스트리밍 연결 열기)
        # 타임아웃: 300초(5분) 안에 전체 응답이 완료되지 않으면 timeout 처리
        async with session.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return Result(req_id, phase, prefill_len, decode_len, rate,
                               send_ts, None, None, None, None,
                               "error", f"http_{resp.status}: {body[:200]}")

            # ③ SSE 스트리밍: 서버가 토큰을 생성할 때마다 한 줄씩 실시간으로 도착
            got_first = False
            async for raw_line in resp.content:  # 한 줄 올 때마다 깨어남
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":  # 서버가 "끝!" 신호를 보냄
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # ④ 첫 토큰 도착 → TTFT 측정 (딱 한 번만 실행됨)
                if not got_first:
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("text", ""):
                        ttft_s = time.time() - send_ts
                        got_first = True

                # 마지막 청크에 들어있는 토큰 사용량 정보 수집
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")

            e2e_s = time.time() - send_ts  # ⑤ 마지막 토큰까지 도착 → E2E 측정

    except asyncio.TimeoutError:
        return Result(req_id, phase, prefill_len, decode_len, rate,
                       send_ts, None, None, None, None, "timeout", "timeout")
    except Exception as exc:
        return Result(req_id, phase, prefill_len, decode_len, rate,
                       send_ts, None, None, None, None, "error", str(exc)[:200])

    return Result(req_id, phase, prefill_len, decode_len, rate,
                   send_ts, ttft_s, e2e_s, prompt_tokens, completion_tokens,
                   "success", None)


# ── single sweep point (Grid 조건 1개 테스트) ─────────────────────────────────
# 하나의 실험 조건(예: 질문512 답변128 QPS4)에 대해:
# 1) 웜업 50발 → 서버 상태 확인 (에러율, TTFT 체크)
# 2) 통과하면 실전 200발 → 결과를 JSONL 파일로 저장
async def run_point(
    base_url: str,
    config: str,
    prefill_len: int,
    decode_len: int,
    rate: float,
    out_path: Path,
) -> bool:
    """Returns True if measured phase was completed (not aborted)."""

    # limit=0: 동시 연결 수 제한 없음 (수백 개 요청이 동시에 날아감)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:

        # [내부 함수] n개의 요청을 포아송 분포(실제 유저들의 불규칙한 접속 패턴)에 맞춰 비동기로 발사
        # random.expovariate(rate): QPS=4면 평균 0.25초 간격이지만, 실제로는 랜덤하게 몰리거나 빔
        async def fire_phase(phase: str, n: int) -> list[Result]:
            results: list[Result] = []
            tasks: list[asyncio.Task] = []
            for i in range(n):
                req_id = f"{config}_{prefill_len}_{decode_len}_{rate}_{phase}_{i}"
                delay = random.expovariate(rate)  # 포아송 분포 기반 랜덤 대기
                await asyncio.sleep(delay)
                t = asyncio.create_task(         # 비동기로 요청 발사 (안 기다리고 다음으로)
                    _do_request(session, base_url, req_id, phase,
                                prefill_len, decode_len, rate)
                )
                tasks.append(t)
            for t in tasks:
                results.append(await t)  # 모든 요청이 끝날 때까지 대기
            return results

        # ── 1단계: 준비운동 (웜업) ──
        warmup_results = await fire_phase("warmup", WARMUP_N)

        # ── 2단계: 웜업 건강 검진 ──
        # 에러율이 너무 높거나 TTFT가 비정상이면 → 실전 스킵 (서버 과부하 방지)
        ok = [r for r in warmup_results if r.status == "success"]
        fail_rate = 1.0 - len(ok) / max(len(warmup_results), 1)
        ttfts = sorted(r.ttft_s for r in ok if r.ttft_s is not None)
        ttft_p99 = ttfts[int(len(ttfts) * 0.99)] if ttfts else 0.0

        aborted = fail_rate > ABORT_FAIL_RATE or ttft_p99 > ABORT_TTFT_P99_S

        # ── 3단계: 실전 측정 (또는 스킵) ──
        measured_results: list[Result] = []
        if not aborted:
            measured_results = await fire_phase("measured", MEASURED_N)
        else:
            print(
                f"  ABORT: fail_rate={fail_rate:.2f} ttft_p99={ttft_p99:.1f}s"
                f" — skipping measured phase",
                flush=True,
            )

        all_results = warmup_results + measured_results

    # ── 4단계: 결과를 JSONL 파일로 저장 (한 줄에 요청 1개의 전체 측정값) ──
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(asdict(r)) + "\n")

    return not aborted


# ── S3 sync (백그라운드 자동 백업) ─────────────────────────────────────────────
# 실험 도중 서버가 꺼져도 데이터가 살아남도록, 30초마다 ./results/ → S3로 복사합니다.
# s5cmd(빠름)이 있으면 쓰고, 없으면 aws cli로 폴백합니다.
# --s3-bucket "" 로 비활성화 가능합니다.
class S3Syncer:

    SYNC_TIMEOUT_S = 300  # 1회 sync 최대 대기 시간

    def __init__(self, bucket: str, log_dir: str, config: str, interval: int = 30):
        self.bucket = bucket
        self.log_dir = log_dir
        self.interval = interval
        # S3 저장 경로: s3://버킷/raw/custom/20260518/hostname/configA1/
        # "custom" prefix → sweep_official.py(공식 벤치) 결과와 구분.
        # config 한 단계 더 → 같은 호스트에서 여러 config 결과 보관해도 안 섞임.
        date = _dt.datetime.utcnow().strftime("%Y%m%d")
        host = socket.gethostname()
        self.dest = f"s3://{bucket}/raw/custom/{date}/{host}/{config}/"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._final_done = False
        self._lock = threading.Lock()
        self._cmd = self._pick_cmd()  # s5cmd 또는 aws cli 자동 탐색
        self._log_path = Path(log_dir) / "s3_sync.log"
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pick_cmd() -> list[str] | None:
        if shutil.which("s5cmd"):
            return ["s5cmd", "sync"]
        if shutil.which("aws"):
            return ["aws", "s3", "sync"]
        return None

    def _log(self, msg: str) -> None:
        ts = _dt.datetime.utcnow().isoformat(timespec="seconds")
        line = f"[{ts}] {msg}\n"
        try:
            with open(self._log_path, "a") as f:
                f.write(line)
        except Exception:
            pass
        print(f"[s3_sync] {msg}", flush=True)

    def _sync_once(self) -> None:
        """로컬 폴더 전체를 S3로 1회 복사 (이동이 아님 — 로컬 원본은 유지)"""
        if not self._cmd:
            return
        argv = self._cmd + [f"{self.log_dir}/", self.dest]
        try:
            with open(self._log_path, "a") as logf:
                subprocess.run(
                    argv,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=self.SYNC_TIMEOUT_S,
                )
        except subprocess.TimeoutExpired:
            self._log(f"sync timed out after {self.SYNC_TIMEOUT_S}s")
        except Exception as exc:
            self._log(f"sync error: {exc}")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._sync_once()
            # Interruptible sleep: stop_event triggers immediate wakeup.
            self._stop_event.wait(timeout=self.interval)

    def start(self) -> None:
        if not self.bucket:
            print("[s3_sync] disabled (no bucket)", flush=True)
            return
        if not self._cmd:
            self._log("WARN: neither s5cmd nor aws cli found — S3 sync DISABLED")
            return
        self._log(f"starting (cmd={self._cmd[0]}, every {self.interval}s) → {self.dest}")
        self._thread = threading.Thread(target=self._loop, daemon=True, name="s3-sync")
        self._thread.start()
        atexit.register(self.stop)

    def stop(self) -> None:
        # Guard against double-invocation (manual stop + atexit).
        with self._lock:
            if self._final_done:
                return
            self._final_done = True

        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        # Final sync regardless — captures whatever the loop missed.
        self._log("final sync ...")
        self._sync_once()
        self._log("stopped")


# ── health check ──────────────────────────────────────────────────────────────
# [헬스 체크 함수] 서버가 완전히 켜져서 트래픽을 받을 준비가 될 때까지 기다립니다.
async def wait_for_health(base_url: str, timeout_s: int = 300) -> None:
    import aiohttp
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
                    if resp.status == 200:
                        print("  server ready.", flush=True)
                        return
            except Exception:
                pass
            await asyncio.sleep(5)
    raise RuntimeError(f"Server at {base_url} not healthy after {timeout_s}s")


# ── main (전체 실험 오케스트레이터) ────────────────────────────────────────────
# 실행 순서: 영수증 생성 → S3 백업 시작 → 서버 대기 → Grid 조건표 생성 → 순차 실행
async def main(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    config = args.config
    out_dir = Path(LOG_DIR) / config  # 예: ./results/D/
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 실험 영수증 생성 (나중에 "이 폴더가 뭐였지?" 할 때 보는 파일) ──
    meta_path = out_dir / "metadata.json"
    if not meta_path.exists():
        # launch_configs.sh 실제 설정에 맞춘 매핑
        # C/C1/C2: same-node PD over shm,  D: cross-node PD over TCP
        tp_map = {"A1": 2, "A2": 4, "A3": 1, "B": 1, "C": 2, "C1": 2, "C2": 1, "D": 1}
        pp_map = {"A1": 2, "A2": 1, "A3": 4, "B": 1, "C": 1, "C1": 1, "C2": 2, "D": 1}
        tp_size = tp_map.get(config, 1)
        pp_size = pp_map.get(config, 1)
        if config in ("C", "C1", "C2"):
            network = "shm"
        elif config == "D":
            network = "TCP"
        else:
            network = "None"
        meta = {
            "config": config,
            "tp_size": tp_size,
            "pp_size": pp_size,
            "network": network,
            "start_time": _dt.datetime.utcnow().isoformat(),
            "description": f"Disaggregated serving benchmark for {config}"
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ── S3 백업 시작 (실험 도중 서버가 죽어도 데이터 보존) ──
    syncer = S3Syncer(
        bucket=args.s3_bucket or "",
        log_dir=LOG_DIR,
        config=config,
        interval=int(os.environ.get("S3_SYNC_INTERVAL", "30")),
    )
    syncer.start()

    # ── 서버가 켜질 때까지 대기 ──
    await wait_for_health(base_url)

    # ── Grid 조건표 생성 ──
    # cross1: 질문길이(3) × QPS(3), 답변=512 고정 → 9개
    # cross2: 답변길이(4) × QPS(3), 질문=2048 고정 → 12개 (중복 제거)
    # 합계: 약 18~21개 조건
    points: list[tuple[int, int, float]] = []
    fixed_decode = 512
    fixed_prefill = 2048
    for pl in PREFILL_LENS:
        for r in RATES:
            points.append((pl, fixed_decode, r))
    for dl in DECODE_LENS:
        for r in RATES:
            if (fixed_prefill, dl, r) not in points:
                points.append((fixed_prefill, dl, r))

    print(f"Grid: {len(points)} points × (warmup={WARMUP_N} + measured={MEASURED_N})", flush=True)

    done = 0
    skipped = 0
    for prefill_len, decode_len, rate in points:
        point_id = f"p{prefill_len}_d{decode_len}_r{rate}"  # 파일명 = 실험 조건
        out_path = out_dir / f"{point_id}.jsonl"             # 결과 저장 파일
        marker_done   = out_dir / f".done_{point_id}"        # 완료 도장 (재실행 시 스킵)
        marker_failed = out_dir / f".failed_{point_id}"      # 실패 도장

        # 이미 완료된 조건은 건너뜀 → 중간에 끊겨도 이어서 실행 가능
        if marker_done.exists():
            skipped += 1
            continue

        print(f"[{done+1}/{len(points)}] prefill={prefill_len} decode={decode_len} rate={rate} ...", flush=True)

        try:
            ok = await run_point(base_url, config, prefill_len, decode_len, rate, out_path)
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            marker_failed.touch()
            continue

        if ok:
            marker_done.touch()
            marker_failed.unlink(missing_ok=True)
        else:
            marker_failed.touch()

        done += 1

    print(f"\nDone. {done} run, {skipped} skipped (already done).", flush=True)

    # Explicit stop so the final sync runs before process exit messages clear
    # the terminal. atexit would also call it, but the guard prevents double-runs.
    syncer.stop()


# ── 진입점: python sweep.py --config D --base-url http://localhost:8000 ──────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="A|B|C|D")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--s3-bucket",
        default=os.environ.get("S3_BUCKET", "hdjung-disaggregation-result"),
        help='S3 bucket for background sync. Pass "" to disable.',
    )
    args = parser.parse_args()
    asyncio.run(main(args))  # 비동기 이벤트 루프 시작 → main() 실행
