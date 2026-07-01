from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET_CLASSES = {0, 1, 2}


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                vals.append(format(value, floatfmt))
            else:
                vals.append(str(value))
        rows.append(vals)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(vals) + " |" for vals in rows)
    return "\n".join(out)


def event_by_dataset(run_dir: Path) -> pd.DataFrame:
    pred = pd.read_csv(run_dir / "events_pred.csv")
    gt = pd.read_csv(run_dir / "events_gt.csv")
    rows = []
    for ds in sorted(gt["dataset"].dropna().unique()):
        p = pred[pred["dataset"] == ds].copy()
        g = gt[(gt["dataset"] == ds) & (gt["label"].isin(TARGET_CLASSES))].copy()
        p = p[p["label"].isin(TARGET_CLASSES)].copy()
        tp = int(g["matched"].sum()) if "matched" in g else 0
        fn = int((~g["matched"]).sum()) if "matched" in g else len(g)
        fp = int((~p["matched"]).sum()) if "matched" in p else len(p)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "dataset": ds,
            "target_event_precision": precision,
            "target_event_recall": recall,
            "target_event_f1": f1,
            "target_gt_events": len(g),
            "target_pred_events": len(p),
        })
    return pd.DataFrame(rows)


def snr_bucket(value) -> str:
    try:
        x = float(value)
    except Exception:
        return "unknown"
    if x <= -5:
        return "<=-5"
    if x <= 0:
        return "(-5,0]"
    if x <= 5:
        return "(0,5]"
    return ">5"


def target_recall_by_snr(run_dir: Path) -> pd.DataFrame:
    gt = pd.read_csv(run_dir / "events_gt.csv")
    gt = gt[(gt["dataset"] == "OOD-A") & (gt["label"].isin(TARGET_CLASSES))].copy()
    if gt.empty or "snr_db" not in gt.columns:
        return pd.DataFrame(columns=["snr_bucket", "target_event_recall", "target_gt_events"])
    gt["snr_bucket"] = gt["snr_db"].map(snr_bucket)
    rows = []
    for bucket, sub in gt.groupby("snr_bucket", sort=False):
        rows.append({
            "snr_bucket": bucket,
            "target_event_recall": float(sub["matched"].mean()) if "matched" in sub else 0.0,
            "target_gt_events": int(len(sub)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize online replay model comparison outputs.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "online_model_comparison")
    args = parser.parse_args()
    runs = {
        "NeuroCAP full": "online_full_neurocap_final",
        "NeuroCAP sound-only": "online_full_sound_final",
        "Baseline ResNet10": "online_full_resnet10_final",
        "Baseline Dilated": "online_full_dilated_final",
        "Baseline Deformable": "online_full_deformable_final",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    dataset_rows = []
    event_dataset_rows = []
    snr_rows = []
    for model, folder in runs.items():
        run_dir = args.root / folder
        s = pd.read_csv(run_dir / "summary.csv").iloc[0].to_dict()
        s["model"] = model
        s["run_dir"] = str(run_dir)
        summary_rows.append(s)
        by_ds = pd.read_csv(run_dir / "by_dataset.csv")
        by_ds.insert(0, "model", model)
        dataset_rows.append(by_ds)
        ev_ds = event_by_dataset(run_dir)
        ev_ds.insert(0, "model", model)
        event_dataset_rows.append(ev_ds)
        snr = target_recall_by_snr(run_dir)
        snr.insert(0, "model", model)
        snr_rows.append(snr)
    summary = pd.DataFrame(summary_rows)
    ordered_cols = ["model"] + [c for c in summary.columns if c not in {"model", "run_dir"}] + ["run_dir"]
    summary = summary[ordered_cols]
    by_dataset = pd.concat(dataset_rows, ignore_index=True)
    event_by_ds = pd.concat(event_dataset_rows, ignore_index=True)
    by_snr = pd.concat(snr_rows, ignore_index=True)
    summary.to_csv(args.out / "summary.csv", index=False)
    by_dataset.to_csv(args.out / "by_dataset.csv", index=False)
    event_by_ds.to_csv(args.out / "event_by_dataset.csv", index=False)
    by_snr.to_csv(args.out / "ood_a_target_recall_by_snr.csv", index=False)
    lines = ["# Online Model Comparison", ""]
    cols = ["model", "target_event_precision", "target_event_recall", "target_event_f1", "false_alarms_per_hour", "window_accuracy", "window_macro_f1", "latency_p95_ms"]
    lines.append(markdown_table(summary[cols].sort_values("target_event_f1", ascending=False)))
    lines.append("")
    lines.append("## By Dataset")
    lines.append(markdown_table(by_dataset))
    lines.append("")
    lines.append("## Target Events By Dataset")
    lines.append(markdown_table(event_by_ds))
    lines.append("")
    lines.append("## OOD-A Target Recall By SNR")
    lines.append(markdown_table(by_snr))
    (args.out / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary[cols].sort_values("target_event_f1", ascending=False).to_string(index=False))
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

