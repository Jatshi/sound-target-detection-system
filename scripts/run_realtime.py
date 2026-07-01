from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sounddet.audio_stream import list_audio_devices
from sounddet.config import load_config
from sounddet.engine import DetectionEngine
from sounddet.event_store import EventStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Realtime input utility.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    if args.list_devices:
        for dev in list_audio_devices():
            print(dev)
        return 0
    if args.duration_sec is None:
        parser.error("pass --duration-sec for microphone detection, or --list-devices")
    cfg = load_config()
    if args.model:
        cfg.default_model = args.model
    engine = DetectionEngine(cfg, model_key=cfg.default_model, store=EventStore(cfg))
    summary = engine.run_microphone(duration_sec=args.duration_sec, device=args.device)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
