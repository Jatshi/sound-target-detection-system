from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from . import CLASS_NAMES, TARGET_CLASSES
from .postprocess import EventRecord, WindowRecord


def window_label_at(t_start: float, t_end: float, gt: pd.DataFrame) -> int:
    if gt.empty:
        return 4
    overlaps = gt[(gt["start"] < t_end) & (gt["end"] > t_start)]
    if overlaps.empty:
        return 4
    targets = overlaps[overlaps["label"].isin(TARGET_CLASSES)]
    if not targets.empty:
        durations = (np.minimum(targets["end"], t_end) - np.maximum(targets["start"], t_start)).to_numpy()
        return int(targets.iloc[int(np.argmax(durations))]["label"])
    durations = (np.minimum(overlaps["end"], t_end) - np.maximum(overlaps["start"], t_start)).to_numpy()
    return int(overlaps.iloc[int(np.argmax(durations))]["label"])


def attach_window_labels(windows: list[WindowRecord], gt_df: pd.DataFrame) -> list[WindowRecord]:
    by_stream = {k: v for k, v in gt_df.groupby("stream_id")} if not gt_df.empty else {}
    for w in windows:
        w.label = window_label_at(w.t_start, w.t_end, by_stream.get(w.stream_id, pd.DataFrame()))
    return windows


def match_events(pred: pd.DataFrame, gt: pd.DataFrame, tolerance_sec: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pred.copy()
    gt = gt.copy()
    pred["matched"] = False
    pred["match_gt_index"] = -1
    pred["timing_error_sec"] = np.nan
    gt["matched"] = False
    for gi, g in gt.iterrows():
        if int(g["label"]) in TARGET_CLASSES:
            label_mask = pred["label"].isin(TARGET_CLASSES)
        else:
            label_mask = pred["label"] == g["label"]
        candidates = pred[
            (pred["stream_id"] == g["stream_id"])
            & label_mask
            & (
                ((pred["start"] < g["end"]) & (pred["end"] > g["start"]))
                | (((pred["start"] + pred["end"]) / 2 - (g["start"] + g["end"]) / 2).abs() <= tolerance_sec)
            )
        ]
        if candidates.empty:
            continue
        center_g = (g["start"] + g["end"]) / 2
        centers = (candidates["start"] + candidates["end"]) / 2
        pi = (centers - center_g).abs().idxmin()
        pred.loc[pi, "matched"] = True
        if int(pred.loc[pi, "match_gt_index"]) < 0:
            pred.loc[pi, "match_gt_index"] = int(gi)
            pred.loc[pi, "timing_error_sec"] = float(((pred.loc[pi, "start"] + pred.loc[pi, "end"]) / 2) - center_g)
        gt.loc[gi, "matched"] = True
    return pred, gt


def compute_reports(
    windows: list[WindowRecord],
    events_pred: list[EventRecord],
    events_gt: pd.DataFrame,
    out_dir: str | Path,
    tolerance_sec: float = 0.5,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    win_df = pd.DataFrame([asdict(w) for w in windows])
    if not win_df.empty:
        prob_cols = [f"prob_{name}" for name in CLASS_NAMES]
        probs = pd.DataFrame(win_df.pop("probs").tolist(), columns=prob_cols)
        win_df = pd.concat([win_df, probs], axis=1)
    pred_df = pd.DataFrame([asdict(e) for e in events_pred])
    if pred_df.empty:
        pred_df = pd.DataFrame(columns=["stream_id", "dataset", "start", "end", "label", "confidence", "source"])
    gt_df = events_gt.copy()
    if gt_df.empty:
        gt_df = pd.DataFrame(columns=["stream_id", "dataset", "start", "end", "label"])

    pred_matched, gt_matched = match_events(pred_df, gt_df, tolerance_sec=tolerance_sec)
    duration_hours = max(1e-9, win_df.groupby("stream_id")["t_end"].max().sum() / 3600.0) if not win_df.empty else 1e-9
    tp = int(gt_matched["matched"].sum()) if "matched" in gt_matched else 0
    fn = int((~gt_matched["matched"]).sum()) if "matched" in gt_matched else 0
    fp = int((~pred_matched["matched"]).sum()) if "matched" in pred_matched else 0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    target_gt = gt_matched[gt_matched["label"].isin(TARGET_CLASSES)] if "label" in gt_matched else pd.DataFrame()
    target_pred = pred_matched[pred_matched["label"].isin(TARGET_CLASSES)] if "label" in pred_matched else pd.DataFrame()
    t_tp = int(target_gt["matched"].sum()) if "matched" in target_gt else 0
    t_fn = int((~target_gt["matched"]).sum()) if "matched" in target_gt else 0
    t_fp = int((~target_pred["matched"]).sum()) if "matched" in target_pred else 0
    target_event_precision = t_tp / (t_tp + t_fp) if t_tp + t_fp else 0.0
    target_event_recall = t_tp / (t_tp + t_fn) if t_tp + t_fn else 0.0
    target_event_f1 = 2 * target_event_precision * target_event_recall / (target_event_precision + target_event_recall) if target_event_precision + target_event_recall else 0.0

    y_true = win_df["label"].astype(int).to_numpy() if "label" in win_df else np.array([], dtype=int)
    y_pred = win_df["pred"].astype(int).to_numpy() if "pred" in win_df else np.array([], dtype=int)
    acc = accuracy_score(y_true, y_pred) if y_true.size else 0.0
    rep = classification_report(y_true, y_pred, labels=list(range(5)), target_names=CLASS_NAMES, output_dict=True, zero_division=0) if y_true.size else {}
    macro_f1 = rep.get("macro avg", {}).get("f1-score", 0.0)
    if y_true.size:
        y_true_target = np.isin(y_true, list(TARGET_CLASSES)).astype(int)
        y_pred_target = np.isin(y_pred, list(TARGET_CLASSES)).astype(int)
        target_pr, target_rc, target_f1, _ = precision_recall_fscore_support(
            y_true_target, y_pred_target, labels=[1], average="binary", zero_division=0
        )
    else:
        target_pr = target_rc = target_f1 = 0.0
    latency = win_df["latency_ms"].to_numpy() if "latency_ms" in win_df else np.array([])

    summary = {
        "window_accuracy": acc,
        "window_macro_f1": macro_f1,
        "target_window_precision": float(target_pr),
        "target_window_recall": float(target_rc),
        "target_window_f1": float(target_f1),
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "target_event_precision": target_event_precision,
        "target_event_recall": target_event_recall,
        "target_event_f1": target_event_f1,
        "false_alarms_per_hour": fp / duration_hours,
        "mean_abs_timing_error_sec": float(np.nanmean(np.abs(pred_matched["timing_error_sec"]))) if "timing_error_sec" in pred_matched and pred_matched["timing_error_sec"].notna().any() else np.nan,
        "latency_p50_ms": float(np.percentile(latency, 50)) if latency.size else np.nan,
        "latency_p95_ms": float(np.percentile(latency, 95)) if latency.size else np.nan,
        "n_windows": int(len(win_df)),
        "n_gt_events": int(len(gt_df)),
        "n_pred_events": int(len(pred_df)),
    }

    by_dataset = []
    if not win_df.empty:
        for ds, sub in win_df.groupby("dataset"):
            yt = sub["label"].astype(int).to_numpy()
            yp = sub["pred"].astype(int).to_numpy()
            r = classification_report(yt, yp, labels=list(range(5)), output_dict=True, zero_division=0)
            by_dataset.append({"dataset": ds, "n_windows": len(sub), "accuracy": accuracy_score(yt, yp), "macro_f1": r["macro avg"]["f1-score"]})
    by_class = []
    if y_true.size:
        pr, rc, f1v, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(5)), zero_division=0)
        for i, name in enumerate(CLASS_NAMES):
            by_class.append({"class_id": i, "class_name": name, "precision": pr[i], "recall": rc[i], "f1": f1v[i], "support": int(sup[i])})

    win_df.to_csv(out_dir / "window_predictions.csv", index=False)
    pred_matched.to_csv(out_dir / "events_pred.csv", index=False)
    gt_matched.to_csv(out_dir / "events_gt.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(by_dataset).to_csv(out_dir / "by_dataset.csv", index=False)
    pd.DataFrame(by_class).to_csv(out_dir / "by_class.csv", index=False)
    _write_figures(win_df, pred_matched, gt_matched, out_dir)
    _write_markdown(summary, by_dataset, by_class, out_dir)
    return summary


def _write_figures(win_df: pd.DataFrame, pred: pd.DataFrame, gt: pd.DataFrame, out_dir: Path) -> None:
    if not win_df.empty and "label" in win_df:
        cm = confusion_matrix(win_df["label"].astype(int), win_df["pred"].astype(int), labels=list(range(5)))
        fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(5), CLASS_NAMES, rotation=35, ha="right")
        ax.set_yticks(range(5), CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Window label")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=160)
    for _, row in gt.head(80).iterrows():
        ax.broken_barh([(row["start"], row["end"] - row["start"])], (row["label"] - 0.35, 0.3), facecolors="tab:green")
    for _, row in pred.head(80).iterrows():
        ax.broken_barh([(row["start"], row["end"] - row["start"])], (row["label"] + 0.05, 0.3), facecolors="tab:red", alpha=0.75)
    ax.set_yticks(range(5), CLASS_NAMES)
    ax.set_xlabel("Time in stream (s), first 80 events")
    ax.set_title("Ground truth (green) and detected events (red)")
    fig.tight_layout()
    fig.savefig(out_dir / "timeline.png")
    plt.close(fig)


def _write_markdown(summary: dict, by_dataset: list[dict], by_class: list[dict], out_dir: Path) -> None:
    lines = ["# Online Streaming Reliability Report", ""]
    lines.append("## Summary")
    for k, v in summary.items():
        lines.append(f"- `{k}`: {v:.6f}" if isinstance(v, float) else f"- `{k}`: {v}")
    lines.append("")
    lines.append("## By Dataset")
    for row in by_dataset:
        lines.append(f"- {row['dataset']}: Acc={row['accuracy']:.4f}, Macro-F1={row['macro_f1']:.4f}, n={row['n_windows']}")
    lines.append("")
    lines.append("## By Class")
    for row in by_class:
        lines.append(f"- {row['class_name']}: P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f}, n={row['support']}")
    (out_dir / "reliability_report.md").write_text("\n".join(lines), encoding="utf-8")
