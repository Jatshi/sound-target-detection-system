from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import CLASS_NAMES
from .config import AppConfig
from .metrics import attach_window_labels, compute_reports
from .model_adapter import ModelAdapter
from .postprocess import EventPostProcessor, WindowRecord
from .online_trial import build_streams
from .sliding_window import iter_windows


def evaluate_online_replay(
    cfg: AppConfig,
    adapter: ModelAdapter,
    out_dir: str | Path,
    minutes: float | None = None,
    streams_per_dataset: int | None = None,
    batch_size: int = 64,
    rebuild: bool = True,
    trial_mode: str | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rebuild or not (out_dir / "streams.csv").exists() or not (out_dir / "events_gt.csv").exists():
        streams, _ = build_streams(cfg, out_dir, minutes=minutes, streams_per_dataset=streams_per_dataset, write_audio=True, trial_mode=trial_mode)
    else:
        streams_df = pd.read_csv(out_dir / "streams.csv")
        streams = [type("Stream", (), row.to_dict()) for _, row in streams_df.iterrows()]
    gt_df = pd.read_csv(out_dir / "events_gt.csv") if (out_dir / "events_gt.csv").exists() else pd.DataFrame()

    windows_all: list[WindowRecord] = []
    threshold = cfg.class_thresholds if cfg.class_thresholds is not None else cfg.target_threshold
    processor = EventPostProcessor(threshold, cfg.ema_alpha, cfg.confirm_frames, cfg.merge_gap_sec)
    events_pred = []
    def threshold_for(label: int) -> float:
        if cfg.class_thresholds is not None and label < len(cfg.class_thresholds):
            return float(cfg.class_thresholds[label])
        return float(cfg.target_threshold)

    for stream in streams:
        wav, sr = sf.read(str(stream.wav_path), dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        if sr != cfg.sample_rate:
            raise RuntimeError(f"Unexpected sample rate for {stream.wav_path}: {sr}")
        starts, buffers = [], []
        for t_start, buf in iter_windows(np.asarray(wav, dtype=np.float32), cfg.sample_rate, cfg.window_sec, cfg.hop_sec):
            starts.append(t_start)
            buffers.append(buf)
        preds = []
        for i in range(0, len(buffers), batch_size):
            preds.extend(adapter.predict_batch(buffers[i : i + batch_size]))
        stream_windows = []
        for t_start, pred in zip(starts, preds):
            decision = pred.pred
            if decision in {0, 1, 2} and pred.confidence < threshold_for(decision):
                decision = 4
            row = WindowRecord(
                stream_id=stream.stream_id,
                dataset=stream.dataset,
                t_start=t_start,
                t_end=t_start + cfg.window_sec,
                pred=decision,
                confidence=pred.confidence,
                probs=pred.probs,
                latency_ms=pred.latency_ms,
            )
            stream_windows.append(row)
        windows_all.extend(stream_windows)
        events_pred.extend(processor.process(stream_windows))

    windows_all = attach_window_labels(windows_all, gt_df)
    return compute_reports(windows_all, events_pred, gt_df, out_dir, tolerance_sec=cfg.match_tolerance_sec)


def evaluate_audio_files(
    cfg: AppConfig,
    adapter: ModelAdapter,
    audio_paths: list[str | Path],
    out_dir: str | Path,
    batch_size: int = 64,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = cfg.class_thresholds if cfg.class_thresholds is not None else cfg.target_threshold
    processor = EventPostProcessor(threshold, cfg.ema_alpha, cfg.confirm_frames, cfg.merge_gap_sec)
    windows_all: list[WindowRecord] = []
    events_all = []
    def threshold_for(label: int) -> float:
        if cfg.class_thresholds is not None and label < len(cfg.class_thresholds):
            return float(cfg.class_thresholds[label])
        return float(cfg.target_threshold)

    for path in audio_paths:
        path = Path(path)
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        if sr != cfg.sample_rate:
            raise RuntimeError(f"Unexpected sample rate for {path}: {sr}; expected {cfg.sample_rate}")
        starts, buffers = [], []
        for t_start, buf in iter_windows(np.asarray(wav, dtype=np.float32), cfg.sample_rate, cfg.window_sec, cfg.hop_sec):
            starts.append(t_start)
            buffers.append(buf)
        preds = []
        for i in range(0, len(buffers), batch_size):
            preds.extend(adapter.predict_batch(buffers[i : i + batch_size]))
        stream_id = path.stem
        stream_windows = []
        for t_start, pred in zip(starts, preds):
            decision = pred.pred
            if decision in {0, 1, 2} and pred.confidence < threshold_for(decision):
                decision = 4
            stream_windows.append(
                WindowRecord(
                    stream_id=stream_id,
                    dataset="user_audio",
                    t_start=t_start,
                    t_end=t_start + cfg.window_sec,
                    pred=decision,
                    confidence=pred.confidence,
                    probs=pred.probs,
                    latency_ms=pred.latency_ms,
                )
            )
        windows_all.extend(stream_windows)
        events_all.extend(processor.process(stream_windows))

    window_rows = []
    for w in windows_all:
        row = {
            "stream_id": w.stream_id,
            "dataset": w.dataset,
            "t_start": w.t_start,
            "t_end": w.t_end,
            "pred": w.pred,
            "pred_name": CLASS_NAMES[w.pred],
            "confidence": w.confidence,
            "latency_ms": w.latency_ms,
        }
        for i, p in enumerate(w.probs):
            row[f"prob_{i}"] = p
        window_rows.append(row)
    event_rows = [
        {
            "stream_id": e.stream_id,
            "dataset": e.dataset,
            "start": e.start,
            "end": e.end,
            "label": e.label,
            "label_name": CLASS_NAMES[e.label],
            "confidence": e.confidence,
        }
        for e in events_all
    ]
    pd.DataFrame(window_rows).to_csv(out_dir / "window_predictions.csv", index=False)
    pd.DataFrame(event_rows).to_csv(out_dir / "events_pred.csv", index=False)
    summary = {
        "n_files": len(audio_paths),
        "n_windows": len(windows_all),
        "n_events": len(events_all),
        "latency_p50_ms": float(np.median([w.latency_ms for w in windows_all])) if windows_all else 0.0,
        "latency_p95_ms": float(np.percentile([w.latency_ms for w in windows_all], 95)) if windows_all else 0.0,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    return summary
