from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import TARGET_CLASSES


@dataclass
class DynamicThresholdCalibrator:
    """Rolling background-aware threshold controller.

    The controller tracks target-class confidence during windows currently
    classified as background/non-target and raises thresholds when the recent
    background floor becomes noisy. It is deliberately conservative and never
    lowers a threshold below the configured base value.
    """

    base_thresholds: list[float]
    hop_sec: float = 0.5
    window_sec: float = 60.0
    std_scale: float = 3.0
    max_raise: float = 0.35
    history: deque[np.ndarray] = field(default_factory=deque)

    def update(self, probs: list[float] | np.ndarray) -> list[float]:
        arr = np.asarray(probs, dtype=np.float32)
        pred = int(arr.argmax())
        if pred not in TARGET_CLASSES:
            self.history.append(arr[list(TARGET_CLASSES)].copy())
            max_len = max(1, int(round(self.window_sec / self.hop_sec)))
            while len(self.history) > max_len:
                self.history.popleft()
        return self.current_thresholds()

    def current_thresholds(self) -> list[float]:
        thresholds = list(self.base_thresholds)
        if not self.history:
            return thresholds
        hist = np.stack(list(self.history), axis=0)
        floor = hist.mean(axis=0) + self.std_scale * hist.std(axis=0)
        for i, label in enumerate(TARGET_CLASSES):
            raised = min(float(self.base_thresholds[label]) + self.max_raise, float(floor[i]))
            thresholds[label] = max(float(self.base_thresholds[label]), raised)
        return thresholds

    def to_dict(self) -> dict:
        return {
            "base_thresholds": list(self.base_thresholds),
            "current_thresholds": self.current_thresholds(),
            "history_windows": len(self.history),
            "window_sec": self.window_sec,
            "std_scale": self.std_scale,
        }
