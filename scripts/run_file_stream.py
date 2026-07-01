from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sounddet.config import load_config
from sounddet.evaluator import evaluate_audio_files
from sounddet.model_adapter import ModelAdapter
from sounddet.model_registry import registry


def collect_audio(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = []
    for pattern in ("*.wav", "*.flac", "*.ogg"):
        files.extend(path.rglob(pattern))
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sliding-window detection on a WAV file or audio folder.")
    parser.add_argument("input", type=Path, help="WAV file or folder containing audio files.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--model", default="neurocap_full")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--confirm-frames", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "file_stream")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.threshold is not None:
        cfg.target_threshold = args.threshold
    if args.confirm_frames is not None:
        cfg.confirm_frames = args.confirm_frames
    specs = registry()
    if args.model not in specs:
        raise SystemExit(f"Unknown model: {args.model}")
    spec = specs[args.model]
    if not spec.available:
        raise SystemExit(f"Checkpoint unavailable: {spec.checkpoint}")
    paths = collect_audio(args.input)
    if not paths:
        raise SystemExit(f"No audio files found: {args.input}")
    adapter = ModelAdapter(spec.checkpoint, mode=spec.mode)
    summary = evaluate_audio_files(cfg, adapter, paths, args.out)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"output_dir: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
