from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sounddet.config import load_config, resolve_app_path
from sounddet.engine import DetectionEngine
from sounddet.event_store import EventStore


def memory_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    except Exception:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeated online replay sessions to check long-run stability.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "test.yaml"))
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--minutes-per-run", type=float, default=0.25)
    parser.add_argument("--streams-per-dataset", type=int, default=1)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "stability"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.default_model = args.model
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    store = EventStore(cfg)
    started = time.time()
    records = []
    run_idx = 0
    while time.time() - started < args.duration_sec:
        run_idx += 1
        before = memory_mb()
        engine = DetectionEngine(cfg, model_key=cfg.default_model, store=store)
        status = "complete"
        error = ""
        try:
            summary = engine.run_streaming_trial(
                minutes=args.minutes_per_run,
                streams_per_dataset=args.streams_per_dataset,
                trial_mode=cfg.trial_mode,
                out_dir=out_root / "sessions",
            )
        except Exception as exc:
            summary = {}
            status = "failed"
            error = str(exc)
        after = memory_mb()
        records.append(
            {
                "run": run_idx,
                "status": status,
                "error": error,
                "elapsed_sec": time.time() - started,
                "memory_before_mb": before,
                "memory_after_mb": after,
                "memory_delta_mb": after - before,
                **summary,
            }
        )
    report = {
        "duration_sec": time.time() - started,
        "runs": len(records),
        "failed_runs": sum(1 for r in records if r["status"] != "complete"),
        "max_memory_delta_mb": max([r["memory_delta_mb"] for r in records] or [0.0]),
        "records": records,
    }
    (out_root / "stability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    return 1 if report["failed_runs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


