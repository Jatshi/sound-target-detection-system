from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import soundfile as sf

from .config import AppConfig, resolve_app_path
from .event_store import EventStore
from .health import InputHealthMonitor
from .metrics import attach_window_labels, compute_reports
from .model_adapter import ModelAdapter
from .model_registry import registry
from .postprocess import EventPostProcessor, EventRecord, WindowRecord
from .online_trial import build_streams
from .sliding_window import iter_windows
from .audio_stream import MicrophoneStream
from .alerting import AlertManager
from .system_monitor import read_runtime_metrics
from .source_localization import DirectionEstimator


EventCallback = Callable[[EventRecord], None]


class DetectionEngine:
    def __init__(self, cfg: AppConfig, model_key: str | None = None, store: EventStore | None = None):
        self.cfg = cfg
        self.model_key = model_key or cfg.default_model
        self.store = store or EventStore(cfg)
        self.session_id = ""
        self.adapter: ModelAdapter | None = None
        self.event_callbacks: list[EventCallback] = []
        self.stop_requested = False

    def add_event_callback(self, cb: EventCallback) -> None:
        self.event_callbacks.append(cb)

    def load_model(self) -> ModelAdapter:
        spec = registry()[self.model_key]
        if not spec.available:
            raise FileNotFoundError(f"Model checkpoint unavailable: {spec.checkpoint}")
        self.adapter = ModelAdapter(spec.checkpoint, mode=spec.mode)
        self.store.upsert_model(spec.key, spec.label, str(spec.checkpoint), spec.mode, spec.available, spec.sha256())
        return self.adapter

    def start_session(self, input_source: str, out_dir: str | Path | None = None) -> Path:
        self.session_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        output = (Path(out_dir) / self.session_id) if out_dir is not None else resolve_app_path(self.cfg, "outputs") / "sessions" / self.session_id
        output.mkdir(parents=True, exist_ok=True)
        self.store.start_session(self.session_id, self.model_key, input_source, asdict(self.cfg), str(output))
        self._write_runtime_metrics("session_start")
        return output

    def stop(self) -> None:
        self.stop_requested = True

    def run_streaming_trial(self, minutes: float | None = None, streams_per_dataset: int | None = None, trial_mode: str | None = None, out_dir: str | Path | None = None) -> dict:
        adapter = self.adapter or self.load_model()
        out = self.start_session(f"online_replay:{trial_mode or self.cfg.trial_mode}", out_dir)
        streams, _ = build_streams(self.cfg, out, minutes=minutes, streams_per_dataset=streams_per_dataset, write_audio=True, trial_mode=trial_mode)
        gt_df = pd.read_csv(out / "events_gt.csv")
        summary = self._run_stream_files(adapter, streams, gt_df, out)
        self.store.end_session(self.session_id, "complete")
        return summary

    def run_audio_files(self, audio_paths: list[str | Path], out_dir: str | Path | None = None, realtime: bool = False) -> dict:
        adapter = self.adapter or self.load_model()
        out = self.start_session("audio_files", out_dir)
        streams = [type("Stream", (), {"stream_id": Path(p).stem, "dataset": "user_audio", "wav_path": str(p)}) for p in audio_paths]
        summary = self._run_stream_files(adapter, streams, pd.DataFrame(), out, realtime=realtime)
        self.store.end_session(self.session_id, "complete")
        return summary

    def run_microphone(self, duration_sec: float | None = None, device: int | None = None, out_dir: str | Path | None = None, channels: int = 1) -> dict:
        adapter = self.adapter or self.load_model()
        out = self.start_session(f"microphone:device={device}:channels={channels}", out_dir)
        health = InputHealthMonitor(self.cfg)
        threshold = self.cfg.class_thresholds if self.cfg.class_thresholds is not None else self.cfg.target_threshold
        processor = EventPostProcessor(threshold, self.cfg.ema_alpha, self.cfg.confirm_frames, self.cfg.merge_gap_sec)
        stream = MicrophoneStream(self.cfg.sample_rate, self.cfg.hop_samples, device=device, channels=channels)
        direction = DirectionEstimator(self.cfg.sample_rate) if channels >= 2 else None
        direction_rows = []
        recorded: list[np.ndarray] = []
        windows: list[WindowRecord] = []
        rolling = np.zeros(0, dtype=np.float32)
        started = time.time()
        t_start = 0.0
        for chunk in stream:
            if self.stop_requested:
                break
            chunk = np.asarray(chunk, dtype=np.float32)
            recorded.append(chunk)
            mono = chunk.mean(axis=1) if chunk.ndim == 2 else chunk.reshape(-1)
            est = direction.estimate(chunk) if direction is not None else None
            if est is not None:
                direction_rows.append({"t_start": t_start, "azimuth_deg": est.azimuth_deg, "confidence": est.confidence, "lag_samples": est.lag_samples})
            rolling = np.concatenate([rolling, mono])[-self.cfg.window_samples :]
            if len(rolling) >= self.cfg.window_samples:
                health.update(rolling, self.cfg.window_samples)
                pred = adapter.predict_one(rolling)
                rms = float(np.sqrt(np.mean(rolling * rolling)))
                decision = pred.pred
                probs = pred.probs
                confidence = pred.confidence
                if rms < self.cfg.live_background_rms_threshold:
                    decision = 4
                    probs = [0.0, 0.0, 0.0, 0.0, 1.0]
                    confidence = 1.0
                elif decision in {0, 1, 2} and pred.confidence < self._threshold_for(decision):
                    decision = 4
                windows.append(WindowRecord("microphone", "live", t_start, t_start + self.cfg.window_sec, decision, confidence, probs, pred.latency_ms))
                t_start += self.cfg.hop_sec
            if duration_sec is not None and time.time() - started >= duration_sec:
                break
        if recorded and recorded[0].ndim == 2:
            wav = np.concatenate(recorded, axis=0)
        else:
            wav = np.concatenate([np.asarray(x).reshape(-1) for x in recorded]) if recorded else np.zeros(0, dtype=np.float32)
        sf.write(str(out / "microphone_recording.wav"), wav, self.cfg.sample_rate)
        windows = attach_window_labels(windows, pd.DataFrame())
        events = processor.process(windows)
        clip_paths = self._save_event_clips(events, wav, out / "clips")
        self.store.insert_events(self.session_id, events, clip_paths)
        self._send_alerts(events, out / "alerts")
        self.store.insert_windows(self.session_id, windows)
        health_payload = health.health.to_dict()
        health_payload["live_background_rms_threshold"] = self.cfg.live_background_rms_threshold
        (out / "input_health.json").write_text(json.dumps(health_payload, indent=2), encoding="utf-8")
        if direction_rows:
            pd.DataFrame(direction_rows).to_csv(out / "direction_estimates.csv", index=False)
        summary = compute_reports(windows, events, pd.DataFrame(), out, tolerance_sec=self.cfg.match_tolerance_sec)
        if direction_rows:
            ddf = pd.DataFrame(direction_rows)
            summary["direction_azimuth_median_deg"] = float(ddf["azimuth_deg"].median())
            summary["direction_confidence_mean"] = float(ddf["confidence"].mean())
        self._write_runtime_metrics("session_end")
        self.store.end_session(self.session_id, "complete")
        return summary

    def _run_stream_files(self, adapter: ModelAdapter, streams, gt_df: pd.DataFrame, out: Path, realtime: bool = False) -> dict:
        all_windows: list[WindowRecord] = []
        all_events: list[EventRecord] = []
        health = InputHealthMonitor(self.cfg)
        threshold = self.cfg.class_thresholds if self.cfg.class_thresholds is not None else self.cfg.target_threshold
        processor = EventPostProcessor(threshold, self.cfg.ema_alpha, self.cfg.confirm_frames, self.cfg.merge_gap_sec)
        for stream in streams:
            if self.stop_requested:
                break
            wav, sr = sf.read(str(stream.wav_path), dtype="float32", always_2d=False)
            if getattr(wav, "ndim", 1) > 1:
                wav = wav.mean(axis=1)
            if sr != self.cfg.sample_rate:
                raise RuntimeError(f"Unexpected sample rate for {stream.wav_path}: {sr}; expected {self.cfg.sample_rate}")
            stream_windows: list[WindowRecord] = []
            for t_start, buf in iter_windows(np.asarray(wav, dtype=np.float32), self.cfg.sample_rate, self.cfg.window_sec, self.cfg.hop_sec):
                if self.stop_requested:
                    break
                health.update(buf, self.cfg.window_samples)
                pred = adapter.predict_one(buf)
                decision = pred.pred
                if decision in {0, 1, 2} and pred.confidence < self._threshold_for(decision):
                    decision = 4
                w = WindowRecord(stream.stream_id, stream.dataset, t_start, t_start + self.cfg.window_sec, decision, pred.confidence, pred.probs, pred.latency_ms)
                stream_windows.append(w)
                all_windows.append(w)
                if realtime:
                    time.sleep(max(0.0, self.cfg.hop_sec))
            events = processor.process(stream_windows)
            clip_paths = self._save_event_clips(events, wav, out / "clips")
            self.store.insert_events(self.session_id, events, clip_paths)
            self._send_alerts(events, out / "alerts")
            for ev in events:
                for cb in self.event_callbacks:
                    cb(ev)
            all_events.extend(events)
        all_windows = attach_window_labels(all_windows, gt_df)
        self.store.insert_windows(self.session_id, all_windows)
        (out / "input_health.json").write_text(json.dumps(health.health.to_dict(), indent=2), encoding="utf-8")
        summary = compute_reports(all_windows, all_events, gt_df, out, tolerance_sec=self.cfg.match_tolerance_sec)
        self._write_runtime_metrics("session_end")
        return summary

    def _threshold_for(self, label: int) -> float:
        if self.cfg.class_thresholds is not None and label < len(self.cfg.class_thresholds):
            return float(self.cfg.class_thresholds[label])
        return float(self.cfg.target_threshold)

    def _save_event_clips(self, events: list[EventRecord], wav: np.ndarray, clip_dir: Path) -> dict[int, str]:
        clip_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[int, str] = {}
        for i, ev in enumerate(events):
            start = max(0, int((ev.start - self.cfg.clip_context_sec) * self.cfg.sample_rate))
            end = min(len(wav), int((ev.end + self.cfg.clip_context_sec) * self.cfg.sample_rate))
            clip = wav[start:end]
            name = f"{self.session_id}_{i:04d}_{ev.label}_{ev.confidence:.3f}.wav"
            path = clip_dir / name
            sf.write(str(path), clip, self.cfg.sample_rate)
            paths[i] = str(path)
        return paths

    def _send_alerts(self, events: list[EventRecord], out_dir: Path) -> None:
        manager = AlertManager(out_dir, webhook_url=self.cfg.alert_webhook_url, cooldown_sec=self.cfg.alert_cooldown_sec)
        for ev in events:
            if ev.label in {0, 1, 2}:
                manager.handle_event(self.session_id, ev)

    def _write_runtime_metrics(self, stage: str) -> None:
        health_dir = resolve_app_path(self.cfg, self.cfg.runtime_health_dir) / self.session_id
        health_dir.mkdir(parents=True, exist_ok=True)
        row = read_runtime_metrics(self.cfg.app_root).to_dict()
        row["stage"] = stage
        path = health_dir / "runtime_timeseries.csv"
        header = not path.exists()
        with path.open("a", encoding="utf-8") as f:
            if header:
                f.write(",".join(row.keys()) + "\n")
            f.write(",".join("" if v is None else str(v) for v in row.values()) + "\n")
