from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .event_store import EventStore


def export_session_report(store: EventStore, session_id: str, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sessions = store.query("SELECT * FROM sessions WHERE session_id=?", (session_id,))
    events = pd.DataFrame(store.query("SELECT * FROM events WHERE session_id=? ORDER BY start", (session_id,)))
    windows = pd.DataFrame(store.query("SELECT * FROM window_predictions WHERE session_id=? ORDER BY t_start", (session_id,)))
    if sessions:
        pd.DataFrame(sessions).to_csv(out / "session.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    windows.to_csv(out / "window_predictions.csv", index=False)
    _write_markdown(sessions[0] if sessions else {}, events, windows, out / "report.md")
    _write_event_timeline(events, out / "event_timeline.png")
    _write_pdf(out / "report.md", out / "report.pdf")
    package = out / f"{session_id}_report.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out.glob("*"):
            if p != package and p.is_file():
                zf.write(p, p.name)
    return package


def _write_markdown(session: dict, events: pd.DataFrame, windows: pd.DataFrame, path: Path) -> None:
    lines = ["# Sound Target Detection Session Report", ""]
    lines.append(f"- Session: `{session.get('session_id', 'unknown')}`")
    lines.append(f"- Model: `{session.get('model_key', 'unknown')}`")
    lines.append(f"- Input: `{session.get('input_source', 'unknown')}`")
    lines.append(f"- Status: `{session.get('status', 'unknown')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Events: {len(events)}")
    lines.append(f"- Windows: {len(windows)}")
    if not windows.empty and "latency_ms" in windows:
        lines.append(f"- Latency p50 ms: {windows['latency_ms'].median():.4f}")
        lines.append(f"- Latency p95 ms: {windows['latency_ms'].quantile(0.95):.4f}")
    lines.append("")
    lines.append("## Events")
    if events.empty:
        lines.append("No events.")
    else:
        show = events[["id", "start", "end", "label_name", "confidence", "review_status"]].head(100)
        lines.append(show.to_csv(index=False))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_event_timeline(events: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3), dpi=160)
    if not events.empty:
        for _, row in events.iterrows():
            ax.broken_barh([(row["start"], row["end"] - row["start"])], (row["label"] - 0.35, 0.7), alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Class")
    ax.set_title("Detected events")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_pdf(markdown_path: Path, pdf_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    fig = plt.figure(figsize=(8.27, 11.69), dpi=160)
    fig.text(0.05, 0.95, text[:5000], va="top", family="monospace", fontsize=8)
    fig.savefig(pdf_path)
    plt.close(fig)
