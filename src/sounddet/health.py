from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import AppConfig


@dataclass
class InputHealth:
    n_chunks: int = 0
    n_nan_chunks: int = 0
    n_silent_chunks: int = 0
    n_clipped_chunks: int = 0
    n_short_chunks: int = 0
    max_peak: float = 0.0
    mean_rms: float = 0.0
    sample_rate_ok: bool = True
    notes: str = ""

    def to_dict(self):
        return asdict(self)


class InputHealthMonitor:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.health = InputHealth()
        self._rms_sum = 0.0

    def update(self, chunk: np.ndarray, expected_len: int | None = None) -> None:
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self.health.n_chunks += 1
        if expected_len is not None and len(x) < expected_len:
            self.health.n_short_chunks += 1
        if not np.isfinite(x).all():
            self.health.n_nan_chunks += 1
            x = np.nan_to_num(x)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        rms = float(np.sqrt(np.mean(x * x) + 1e-12)) if x.size else 0.0
        self.health.max_peak = max(self.health.max_peak, peak)
        self._rms_sum += rms
        self.health.mean_rms = self._rms_sum / max(1, self.health.n_chunks)
        if rms < self.cfg.health_silence_rms:
            self.health.n_silent_chunks += 1
        if peak >= self.cfg.health_clip_peak:
            self.health.n_clipped_chunks += 1
