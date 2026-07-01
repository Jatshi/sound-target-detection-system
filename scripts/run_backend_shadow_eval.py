from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from sounddet.config import load_config
from sounddet.edge_backend import EdgeInferenceBackend
from sounddet.online_trial import build_streams
from sounddet.sliding_window import iter_windows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run backend shadow comparison on online replay streams.")
    parser.add_argument("--backend", default="onnxruntime-cuda", choices=["onnxruntime-cuda", "onnxruntime-cpu", "tensorrt"])
    parser.add_argument("--model", default="neurocap_full")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--deformable-policy", choices=["native", "static-conv"], default="static-conv")
    parser.add_argument("--minutes", type=float, default=1.0)
    parser.add_argument("--streams-per-dataset", type=int, default=1)
    parser.add_argument("--out", default=str(APP_ROOT / "outputs" / "backend_shadow"))
    args = parser.parse_args()

    cfg = load_config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    streams, _ = build_streams(cfg, out, minutes=args.minutes, streams_per_dataset=args.streams_per_dataset, write_audio=True, trial_mode=cfg.trial_mode)
    main = EdgeInferenceBackend(args.model, "pytorch", deformable_policy=args.deformable_policy)
    shadow = EdgeInferenceBackend(args.model, args.backend, model_path=args.model_path)
    rows = []
    for stream in streams:
        wav, sr = sf.read(str(stream.wav_path), dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        windows = [buf for _t, buf in iter_windows(np.asarray(wav, dtype=np.float32), sr, cfg.window_sec, cfg.hop_sec)]
        ref = main.predict_batch(windows)
        cand = shadow.predict_batch(windows)
        for i, (r, c) in enumerate(zip(ref, cand)):
            rows.append(
                {
                    "stream_id": stream.stream_id,
                    "dataset": stream.dataset,
                    "window_index": i,
                    "pytorch_pred": r.pred,
                    "shadow_pred": c.pred,
                    "pytorch_confidence": r.confidence,
                    "shadow_confidence": c.confidence,
                    "agree": r.pred == c.pred,
                    "max_prob_error": float(np.max(np.abs(np.asarray(r.probs) - np.asarray(c.probs)))),
                }
            )
    df = pd.DataFrame(rows)
    summary = {
        "backend": args.backend,
        "model": args.model,
        "n_windows": int(len(df)),
        "prediction_agreement": float(df["agree"].mean()) if not df.empty else 0.0,
        "max_probability_error": float(df["max_prob_error"].max()) if not df.empty else 0.0,
        "mean_probability_error": float(df["max_prob_error"].mean()) if not df.empty else 0.0,
    }
    df.to_csv(out / "backend_shadow_windows.csv", index=False)
    (out / "backend_drift_report.md").write_text(
        "\n".join(
            [
                "# Backend Drift Report",
                "",
                f"- backend: `{summary['backend']}`",
                f"- model: `{summary['model']}`",
                f"- n_windows: `{summary['n_windows']}`",
                f"- prediction_agreement: `{summary['prediction_agreement']:.6f}`",
                f"- max_probability_error: `{summary['max_probability_error']:.6f}`",
                f"- mean_probability_error: `{summary['mean_probability_error']:.6f}`",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["prediction_agreement"] >= 0.995 else 1


if __name__ == "__main__":
    raise SystemExit(main())


