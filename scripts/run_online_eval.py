from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from sounddet.config import load_config
from sounddet.evaluator import evaluate_online_replay
from sounddet.model_adapter import ModelAdapter
from sounddet.model_registry import registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Run online replay sound target detection evaluation.")
    parser.add_argument("--config", default=str(APP_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--quick", action="store_true", help="Run 1 stream per OOD with 60 s streams.")
    parser.add_argument("--minutes", type=float, default=None)
    parser.add_argument("--streams-per-dataset", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--class-thresholds", default=None, help="Comma-separated thresholds for Gunshot,Glass,Babycry.")
    parser.add_argument("--confirm-frames", type=int, default=None)
    parser.add_argument("--ema-alpha", type=float, default=None)
    parser.add_argument("--merge-gap-sec", type=float, default=None)
    parser.add_argument("--trial-mode", choices=["stream", "aligned"], default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.threshold is not None:
        cfg.target_threshold = args.threshold
    if args.class_thresholds:
        cfg.class_thresholds = tuple(float(x.strip()) for x in args.class_thresholds.split(",") if x.strip())[:3]
    if args.confirm_frames is not None:
        cfg.confirm_frames = args.confirm_frames
    if args.ema_alpha is not None:
        cfg.ema_alpha = args.ema_alpha
    if args.merge_gap_sec is not None:
        cfg.merge_gap_sec = args.merge_gap_sec
    if args.trial_mode is not None:
        cfg.trial_mode = args.trial_mode
    model_key = args.model or cfg.default_model
    spec = registry()[model_key]
    if not spec.available:
        print(f"Model is unavailable: {model_key} -> {spec.checkpoint}")
        return 2
    out_dir = APP_ROOT / (args.out or cfg.output_dir)
    minutes = args.minutes
    streams = args.streams_per_dataset
    if args.quick:
        minutes = 1.0
        streams = 1
        if args.out is None:
            out_dir = APP_ROOT / "outputs" / "online_eval_quick"
    adapter = ModelAdapter(spec.checkpoint, mode=spec.mode)
    summary = evaluate_online_replay(cfg, adapter, out_dir, minutes=minutes, streams_per_dataset=streams, trial_mode=cfg.trial_mode)
    print("Online replay evaluation complete")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  output_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
