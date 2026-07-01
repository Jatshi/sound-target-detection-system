from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from sounddet.edge_backend import EdgeInferenceBackend
from sounddet.features import load_audio_mono


REFERENCE = {
    "OOD-A": APP_ROOT / "reference" / "offline_predictions" / "ood_a_predictions.csv",
    "OOD-B": APP_ROOT / "reference" / "offline_predictions" / "ood_b_predictions.csv",
    "OOD-C": APP_ROOT / "reference" / "offline_predictions" / "ood_c_predictions.csv",
    "OOD-D": APP_ROOT / "reference" / "offline_predictions" / "ood_d_predictions.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PyTorch deploy wrapper with an edge backend.")
    parser.add_argument("--backend", default="onnxruntime-cuda", choices=["onnxruntime-cuda", "onnxruntime-cpu", "onnxruntime-tensorrt", "pytorch"])
    parser.add_argument("--model", default="neurocap_full")
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--dataset", default="OOD-B", choices=sorted(REFERENCE))
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--agreement-threshold", type=float, default=0.995)
    parser.add_argument("--max-prob-error", type=float, default=1e-3)
    parser.add_argument("--deformable-policy", choices=["native", "static-conv"], default="static-conv")
    parser.add_argument("--out", default=str(APP_ROOT / "outputs" / "edge_validation"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REFERENCE[args.dataset]).head(args.n).copy()
    windows = [load_audio_mono(p, samples=44100).squeeze(0).numpy() for p in df["file_path"]]
    baseline = EdgeInferenceBackend(args.model, "pytorch", deformable_policy=args.deformable_policy)
    candidate = EdgeInferenceBackend(args.model, args.backend, model_path=args.onnx)
    p_ref = baseline.predict_batch(windows)
    p_edge = candidate.predict_batch(windows)
    ref_probs = np.asarray([p.probs for p in p_ref])
    edge_probs = np.asarray([p.probs for p in p_edge])
    ref_pred = ref_probs.argmax(axis=1)
    edge_pred = edge_probs.argmax(axis=1)
    agreement = float((ref_pred == edge_pred).mean())
    max_err = float(np.max(np.abs(ref_probs - edge_probs)))
    mean_err = float(np.mean(np.abs(ref_probs - edge_probs)))
    result = {
        "model": args.model,
        "backend": args.backend,
        "dataset": args.dataset,
        "n": len(df),
        "top1_agreement": agreement,
        "max_probability_error": max_err,
        "mean_probability_error": mean_err,
        "deformable_policy": args.deformable_policy,
        "passed": agreement >= args.agreement_threshold and max_err <= args.max_prob_error,
    }
    df["pytorch_deploy_pred"] = ref_pred
    df["edge_pred"] = edge_pred
    df["agreement"] = ref_pred == edge_pred
    df.to_csv(out / f"{args.dataset}_{args.model}_{args.backend}_validation.csv", index=False)
    (out / f"{args.dataset}_{args.model}_{args.backend}_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
