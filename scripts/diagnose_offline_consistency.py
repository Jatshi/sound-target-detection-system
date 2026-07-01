from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sounddet.model_adapter import ModelAdapter
from sounddet.model_registry import registry
from sounddet.features import load_audio_mono


PAPER_PREDICTIONS = {
    "OOD-A": ROOT / "reference" / "offline_predictions" / "ood_a_predictions.csv",
    "OOD-B": ROOT / "reference" / "offline_predictions" / "ood_b_predictions.csv",
    "OOD-C": ROOT / "reference" / "offline_predictions" / "ood_c_predictions.csv",
    "OOD-D": ROOT / "reference" / "offline_predictions" / "ood_d_predictions.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare app inference with stored offline prediction CSVs.")
    parser.add_argument("--dataset", default="OOD-A", choices=sorted(PAPER_PREDICTIONS))
    parser.add_argument("--model", default="neurocap_full")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "diagnostics" / "offline_consistency")
    args = parser.parse_args()

    csv_path = PAPER_PREDICTIONS[args.dataset]
    df = pd.read_csv(csv_path).head(args.n).copy()
    spec = registry()[args.model]
    adapter = ModelAdapter(spec.checkpoint, mode=spec.mode)
    windows = [load_audio_mono(p, samples=44100).squeeze(0).numpy() for p in df["file_path"]]
    preds = adapter.predict_batch(windows)
    df["app_pred"] = [p.pred for p in preds]
    df["app_confidence"] = [p.confidence for p in preds]
    paper_col = "pred_calibrated" if args.model == "neurocap_full" and "pred_calibrated" in df.columns else "pred_sound"
    if args.model != "neurocap_full" and args.model != "neurocap_sound_only":
        paper_col = None
    label = df["label"].astype(int)
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "n": len(df),
        "app_accuracy": accuracy_score(label, df["app_pred"]),
        "paper_accuracy": accuracy_score(label, df[paper_col].astype(int)) if paper_col else None,
        "prediction_agreement_with_paper": (df["app_pred"].astype(int) == df[paper_col].astype(int)).mean() if paper_col else None,
    }
    rep = classification_report(label, df["app_pred"], labels=list(range(5)), output_dict=True, zero_division=0)
    summary["app_macro_f1"] = rep["macro avg"]["f1-score"]
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"{args.dataset}_{args.model}_predictions.csv", index=False)
    pd.DataFrame([summary]).to_csv(args.out / f"{args.dataset}_{args.model}_summary.csv", index=False)
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
