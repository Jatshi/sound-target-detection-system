from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import CLASS_NAMES, TARGET_CLASSES
from .dynamic_threshold import DynamicThresholdCalibrator
from .event_dedupe import dedupe_events_by_edit_distance


@dataclass
class WindowRecord:
    stream_id: str
    dataset: str
    t_start: float
    t_end: float
    pred: int
    confidence: float
    probs: list[float]
    latency_ms: float
    label: int | None = None


@dataclass
class EventRecord:
    stream_id: str
    dataset: str
    start: float
    end: float
    label: int
    confidence: float
    source: str = "pred"


class EventPostProcessor:
    def __init__(
        self,
        threshold: float | list[float] | tuple[float, ...] = 0.5,
        ema_alpha: float = 0.4,
        confirm_frames: int = 2,
        merge_gap_sec: float = 0.8,
        dynamic_calibrator: DynamicThresholdCalibrator | None = None,
        edit_distance_dedupe: bool = False,
    ):
        self.threshold = threshold
        self.ema_alpha = ema_alpha
        self.confirm_frames = confirm_frames
        self.merge_gap_sec = merge_gap_sec
        self.dynamic_calibrator = dynamic_calibrator
        self.edit_distance_dedupe = edit_distance_dedupe

    def process(self, windows: list[WindowRecord]) -> list[EventRecord]:
        events: list[EventRecord] = []
        smooth = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        active_label: int | None = None
        active_start = 0.0
        active_conf = 0.0
        streak_label: int | None = None
        streak = 0
        for w in windows:
            smooth = self.ema_alpha * np.asarray(w.probs, dtype=np.float32) + (1.0 - self.ema_alpha) * smooth
            if self.dynamic_calibrator is not None:
                self.dynamic_calibrator.update(smooth)
            label = int(np.argmax(smooth))
            conf = float(smooth[label])
            is_target = label in TARGET_CLASSES and conf >= self._threshold_for(label)
            if is_target:
                if streak_label == label:
                    streak += 1
                else:
                    streak_label = label
                    streak = 1
                if active_label is None and streak >= self.confirm_frames:
                    active_label = label
                    active_start = max(0.0, w.t_start - (self.confirm_frames - 1) * (w.t_end - w.t_start))
                    active_conf = conf
                elif active_label == label:
                    active_conf = max(active_conf, conf)
                elif active_label is not None:
                    events.append(EventRecord(w.stream_id, w.dataset, active_start, w.t_start, active_label, active_conf))
                    active_label = label
                    active_start = w.t_start
                    active_conf = conf
            else:
                streak_label = None
                streak = 0
                if active_label is not None:
                    events.append(EventRecord(w.stream_id, w.dataset, active_start, w.t_end, active_label, active_conf))
                    active_label = None
                    active_conf = 0.0
        if active_label is not None and windows:
            events.append(EventRecord(windows[-1].stream_id, windows[-1].dataset, active_start, windows[-1].t_end, active_label, active_conf))
        merged = self._merge(events)
        if self.edit_distance_dedupe:
            merged = dedupe_events_by_edit_distance(merged, gap_sec=self.merge_gap_sec)
        return merged

    def _threshold_for(self, label: int) -> float:
        if self.dynamic_calibrator is not None:
            thresholds = self.dynamic_calibrator.current_thresholds()
            if label < len(thresholds):
                return float(thresholds[label])
        if isinstance(self.threshold, (list, tuple)) and label < len(self.threshold):
            return float(self.threshold[label])
        return float(self.threshold)

    def _merge(self, events: list[EventRecord]) -> list[EventRecord]:
        out: list[EventRecord] = []
        for ev in sorted(events, key=lambda e: (e.stream_id, e.start)):
            if out and out[-1].stream_id == ev.stream_id and out[-1].label == ev.label and ev.start - out[-1].end <= self.merge_gap_sec:
                out[-1].end = max(out[-1].end, ev.end)
                out[-1].confidence = max(out[-1].confidence, ev.confidence)
            else:
                out.append(ev)
        return out
