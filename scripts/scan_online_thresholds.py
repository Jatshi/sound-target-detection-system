from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from sounddet.config import load_config
from sounddet.evaluator import evaluate_online_replay
from sounddet.model_adapter import ModelAdapter
from sounddet.model_registry import registry


def parse_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan global and classwise thresholds on online replay trials.")
    parser.add_argument("--config", default=str(APP_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--model", default="neurocap_full")
    parser.add_argument("--trial-mode", choices=["stream", "aligned"], default="aligned")
    parser.add_argument("--minutes", type=float, default=1.0)
    parser.add_argument("--streams-per-dataset", type=int, default=1)
    parser.add_argument("--global-grid", default="0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--class-grid", default="")
    parser.add_argument("--confirm-grid", default="1,2")
    parser.add_argument("--ema-grid", default="0.4")
    parser.add_argument("--out", type=Path, default=APP_ROOT / "outputs" / "threshold_scan")
    args = parser.parse_args()

    spec = registry()[args.model]
    if not spec.available:
        raise SystemExit(f"Checkpoint unavailable: {spec.checkpoint}")
    adapter = ModelAdapter(spec.checkpoint, mode=spec.mode)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    first = True
    global_grid = parse_grid(args.global_grid)
    confirm_grid = [int(x) for x in parse_grid(args.confirm_grid)]
    ema_grid = parse_grid(args.ema_grid)
    candidates: list[tuple[float | None, tuple[float, float, float] | None]] = [(thr, None) for thr in global_grid]
    if args.class_grid:
        values = parse_grid(args.class_grid)
        candidates.extend((None, combo) for combo in itertools.product(values, repeat=3))

    for ema_alpha in ema_grid:
      for confirm in confirm_grid:
        for global_thr, class_thr in candidates:
            cfg = load_config(args.config)
            cfg.trial_mode = args.trial_mode
            cfg.ema_alpha = ema_alpha
            cfg.confirm_frames = confirm
            if class_thr is None:
                cfg.target_threshold = float(global_thr)
                tag = f"global_{global_thr:.2f}"
            else:
                cfg.class_thresholds = class_thr
                tag = "class_" + "_".join(f"{x:.2f}" for x in class_thr)
            run_dir = args.out / args.model / args.trial_mode / f"ema{ema_alpha:.2f}_confirm{confirm}_{tag}"
            summary = evaluate_online_replay(
                cfg,
                adapter,
                run_dir,
                minutes=args.minutes,
                streams_per_dataset=args.streams_per_dataset,
                rebuild=first,
                trial_mode=args.trial_mode,
            )
            first = False
            rows.append({
                "model": args.model,
                "trial_mode": args.trial_mode,
                "ema_alpha": ema_alpha,
                "confirm_frames": confirm,
                "global_threshold": global_thr,
                "class_thresholds": "" if class_thr is None else ",".join(f"{x:.2f}" for x in class_thr),
                **summary,
            })
            pd.DataFrame(rows).to_csv(args.out / f"{args.model}_{args.trial_mode}_scan.csv", index=False)
            print(rows[-1])

    df = pd.DataFrame(rows)
    df["score"] = df["event_f1"] - 0.002 * df["false_alarms_per_hour"]
    df.sort_values(["score", "event_f1", "event_precision"], ascending=False).to_csv(args.out / f"{args.model}_{args.trial_mode}_ranked.csv", index=False)
    print("Best:")
    print(df.sort_values(["score", "event_f1", "event_precision"], ascending=False).head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


