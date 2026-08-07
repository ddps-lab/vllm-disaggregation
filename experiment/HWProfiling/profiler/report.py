import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger("hwprofiling")

CSV_FIELDS = [
    "expert_kind", "M", "stage",
    "median_ms", "mean_ms", "min_ms", "p95_ms",
    "achieved_tflops", "achieved_gbps", "compute_util", "mem_bw_util",
    "replicas_used", "rotation_ok",
]


def setup_console_logging(verbose: bool) -> None:
    root = logging.getLogger("hwprofiling")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(h)


def setup_run_dir(results_base: Path, model_key: str, instance_label: str) -> Path:
    run_dir = results_base / model_key / instance_label
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def attach_file_log(run_dir: Path, timestamp: str) -> logging.Handler:
    handler = logging.FileHandler(run_dir / f"run_{timestamp}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger("hwprofiling").addHandler(handler)
    return handler


def detach_file_log(handler: logging.Handler) -> None:
    logging.getLogger("hwprofiling").removeHandler(handler)
    handler.close()


def write_results(run_dir: Path, timestamp: str, payload: dict) -> None:
    json_path = run_dir / f"results_{timestamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", json_path)

    csv_path = run_dir / f"results_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for m in payload["measurements"]:
            row = {k: m.get(k) for k in CSV_FIELDS if k in m}
            lat = m["latency_ms"]
            row.update(
                median_ms=round(lat["median"], 6),
                mean_ms=round(lat["mean"], 6),
                min_ms=round(lat["min"], 6),
                p95_ms=round(lat["p95"], 6),
            )
            for k in ("achieved_tflops", "achieved_gbps", "compute_util", "mem_bw_util"):
                if m.get(k) is not None:
                    row[k] = round(m[k], 4)
            writer.writerow(row)
    logger.info("wrote %s", csv_path)
