from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from sounddet import CLASS_NAMES, TARGET_CLASSES
from sounddet.dynamic_threshold import DynamicThresholdCalibrator
from sounddet.event_dedupe import dedupe_events_by_edit_distance
from sounddet.metrics import match_events
from sounddet.postprocess import EventPostProcessor, WindowRecord


def default_run_dir() -> Path:
    candidates = [
        APP_ROOT / "outputs" / "online_full_neurocap_tuned",
        APP_ROOT / "outputs" / "online_full_neurocap_final",
        APP_ROOT / "outputs" / "online_full_resnet10_tuned",
        APP_ROOT / "outputs" / "online_full_resnet10_final",
    ]
    for path in candidates:
        if (path / "window_predictions.csv").exists() and (path / "events_gt.csv").exists():
            return path
    matches = sorted((APP_ROOT / "outputs").glob("**/window_predictions.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for win in matches:
        if (win.parent / "events_gt.csv").exists():
            return win.parent
    raise FileNotFoundError("No online replay run with window_predictions.csv and events_gt.csv was found.")


def load_windows(path: Path) -> list[WindowRecord]:
    df = pd.read_csv(path)
    prob_cols = [f"prob_{name}" for name in CLASS_NAMES]
    probs = df[prob_cols].to_numpy(dtype=float)
    raw_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    windows = []
    for i, row in df.iterrows():
        windows.append(
            WindowRecord(
                stream_id=str(row["stream_id"]),
                dataset=str(row["dataset"]),
                t_start=float(row["t_start"]),
                t_end=float(row["t_end"]),
                pred=int(raw_pred[i]),
                confidence=float(conf[i]),
                probs=probs[i].tolist(),
                latency_ms=float(row.get("latency_ms", 0.0)),
                label=int(row["label"]) if "label" in row and pd.notna(row["label"]) else None,
            )
        )
    return windows


def group_windows(windows: list[WindowRecord]) -> dict[str, list[WindowRecord]]:
    grouped: dict[str, list[WindowRecord]] = {}
    for window in windows:
        grouped.setdefault(window.stream_id, []).append(window)
    return grouped


def apply_target_thresholds(windows: list[WindowRecord], thresholds: list[float]) -> list[WindowRecord]:
    out = []
    for w in windows:
        pred = int(np.argmax(w.probs))
        conf = float(w.probs[pred])
        if pred in TARGET_CLASSES and conf < thresholds[pred]:
            pred = 4
        out.append(WindowRecord(w.stream_id, w.dataset, w.t_start, w.t_end, pred, conf, w.probs, w.latency_ms, w.label))
    return out


def run_strategy(
    name: str,
    windows: list[WindowRecord],
    gt: pd.DataFrame,
    thresholds: list[float],
    ema_alpha: float,
    confirm_frames: int,
    merge_gap_sec: float,
    dynamic: bool,
    dedupe: bool,
) -> tuple[dict, pd.DataFrame]:
    thresholded = apply_target_thresholds(windows, thresholds)
    events = []
    for stream_windows in group_windows(thresholded).values():
        calibrator = None
        if dynamic:
            calibrator = DynamicThresholdCalibrator(thresholds, hop_sec=0.5, window_sec=60.0, std_scale=3.0)
        processor = EventPostProcessor(
            thresholds,
            ema_alpha=ema_alpha,
            confirm_frames=confirm_frames,
            merge_gap_sec=merge_gap_sec,
            dynamic_calibrator=calibrator,
            edit_distance_dedupe=False,
        )
        stream_events = processor.process(stream_windows)
        if dedupe:
            stream_events = dedupe_events_by_edit_distance(stream_events, gap_sec=merge_gap_sec)
        events.extend(stream_events)
    pred = pd.DataFrame([asdict(e) for e in events])
    if pred.empty:
        pred = pd.DataFrame(columns=["stream_id", "dataset", "start", "end", "label", "confidence", "source"])
    pred_m, gt_m = match_events(pred, gt, tolerance_sec=0.5)
    duration_hours = max(1e-9, sum(max(w.t_end for w in ws) for ws in group_windows(windows).values()) / 3600.0)
    target_gt = gt_m[gt_m["label"].isin(TARGET_CLASSES)] if "label" in gt_m else pd.DataFrame()
    target_pred = pred_m[pred_m["label"].isin(TARGET_CLASSES)] if "label" in pred_m else pd.DataFrame()
    tp = int(target_gt["matched"].sum()) if "matched" in target_gt else 0
    fn = int((~target_gt["matched"]).sum()) if "matched" in target_gt else 0
    fp = int((~target_pred["matched"]).sum()) if "matched" in target_pred else 0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "strategy": name,
        "dynamic_threshold": dynamic,
        "edit_distance_dedupe": dedupe,
        "target_event_precision": precision,
        "target_event_recall": recall,
        "target_event_f1": f1,
        "false_alarms_per_hour": fp / duration_hours,
        "n_pred_events": int(len(pred)),
        "target_tp": tp,
        "target_fp": fp,
        "target_fn": fn,
    }
    return summary, pred_m


def classwise_rows(strategy: str, pred_m: pd.DataFrame, gt: pd.DataFrame) -> list[dict]:
    rows = []
    for label, name in enumerate(CLASS_NAMES):
        if label not in TARGET_CLASSES:
            continue
        gt_l = gt[gt["label"] == label]
        pred_l = pred_m[pred_m["label"] == label] if "label" in pred_m else pd.DataFrame()
        matched = match_events(pred_l.copy(), gt_l.copy(), tolerance_sec=0.5) if not gt_l.empty or not pred_l.empty else (pred_l, gt_l)
        pred_l_m, gt_l_m = matched
        tp = int(gt_l_m["matched"].sum()) if "matched" in gt_l_m else 0
        fn = int((~gt_l_m["matched"]).sum()) if "matched" in gt_l_m else 0
        fp = int((~pred_l_m["matched"]).sum()) if "matched" in pred_l_m else 0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"strategy": strategy, "class_id": label, "class_name": name, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    return rows


def write_plot(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), dpi=170)
    metrics = [
        ("target_event_f1", "Target F1"),
        ("target_event_recall", "Recall"),
        ("target_event_precision", "Precision"),
        ("false_alarms_per_hour", "FAR/hour"),
    ]
    colors = ["#7bb7ff", "#69d49a", "#f0b35a", "#f06f6c"]
    for ax, (col, title) in zip(axes, metrics):
        vals = summary[col].to_numpy(dtype=float)
        ax.bar(range(len(vals)), vals, color=colors, edgecolor="#223343", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(vals)), summary["strategy"], rotation=35, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "strategy_ablation_metrics.png")
    plt.close(fig)


def write_report(summary: pd.DataFrame, by_class: pd.DataFrame, run_dir: Path, out: Path) -> None:
    base = summary.iloc[0]
    lines = ["# Online Strategy Ablation Report", ""]
    lines.append(f"- Source run: `{run_dir}`")
    lines.append("- Default deployment policy remains unchanged unless a strategy is clearly positive.")
    lines.append("")
    lines.append("## Summary")
    for _, row in summary.iterrows():
        delta_far = float(row["false_alarms_per_hour"] - base["false_alarms_per_hour"])
        delta_f1 = float(row["target_event_f1"] - base["target_event_f1"])
        positive = delta_far < 0 and delta_f1 >= -0.01 and float(row["target_event_recall"]) >= float(base["target_event_recall"]) - 0.02
        verdict = "positive support" if positive else "diagnostic only"
        lines.append(
            f"- {row['strategy']}: F1={row['target_event_f1']:.4f}, Recall={row['target_event_recall']:.4f}, "
            f"Precision={row['target_event_precision']:.4f}, FAR/hour={row['false_alarms_per_hour']:.2f}, "
            f"events={int(row['n_pred_events'])}, verdict={verdict}"
        )
    lines.append("")
    lines.append("## Interpretation")
    best = summary.sort_values(["target_event_f1", "target_event_recall"], ascending=False).iloc[0]
    safest = summary.sort_values(["false_alarms_per_hour", "target_event_f1"], ascending=[True, False]).iloc[0]
    lines.append(f"- Highest target-event F1: `{best['strategy']}`.")
    lines.append(f"- Lowest FAR/hour: `{safest['strategy']}`.")
    lines.append("- If a strategy reduces FAR but also lowers recall/F1 materially, keep it as an optional deployment preset rather than enabling it by default.")
    lines.append("")
    lines.append("## Classwise Target Results")
    for _, row in by_class.iterrows():
        lines.append(f"- {row['strategy']} / {row['class_name']}: P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f}")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate online decision strategies from saved window probabilities.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=APP_ROOT / "outputs" / "strategy_ablation")
    parser.add_argument("--class-thresholds", default="0.25,0.25,0.35")
    parser.add_argument("--ema-alpha", type=float, default=0.60)
    parser.add_argument("--confirm-frames", type=int, default=1)
    parser.add_argument("--merge-gap-sec", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = args.run_dir or default_run_dir()
    args.out.mkdir(parents=True, exist_ok=True)
    thresholds3 = [float(x.strip()) for x in args.class_thresholds.split(",") if x.strip()]
    thresholds = [0.5, 0.5, 0.5, 1.0, 1.0]
    for i, value in enumerate(thresholds3[:3]):
        thresholds[i] = value
    windows = load_windows(run_dir / "window_predictions.csv")
    gt = pd.read_csv(run_dir / "events_gt.csv")
    strategies = [
        ("baseline", False, False),
        ("dynamic_threshold", True, False),
        ("edit_distance_dedupe", False, True),
        ("dynamic_plus_dedupe", True, True),
    ]
    summary_rows = []
    class_rows = []
    for name, dynamic, dedupe in strategies:
        summary, pred_m = run_strategy(name, windows, gt, thresholds, args.ema_alpha, args.confirm_frames, args.merge_gap_sec, dynamic, dedupe)
        summary_rows.append(summary)
        class_rows.extend(classwise_rows(name, pred_m, gt))
        pred_m.to_csv(args.out / f"{name}_events_pred.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    by_class_df = pd.DataFrame(class_rows)
    summary_df.to_csv(args.out / "summary.csv", index=False)
    by_class_df.to_csv(args.out / "by_class.csv", index=False)
    (args.out / "metadata.json").write_text(json.dumps({"run_dir": str(run_dir), "thresholds": thresholds, "ema_alpha": args.ema_alpha, "confirm_frames": args.confirm_frames, "merge_gap_sec": args.merge_gap_sec}, indent=2), encoding="utf-8")
    write_plot(summary_df, args.out)
    write_report(summary_df, by_class_df, run_dir, args.out)
    print(summary_df.to_string(index=False))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
