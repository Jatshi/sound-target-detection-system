from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DirectionEstimate:
    azimuth_deg: float
    confidence: float
    lag_samples: int


class DirectionEstimator:
    """Lightweight two-channel direction cue for microphone arrays.

    The estimate is intentionally conservative: without a calibrated array
    geometry we report a left/right azimuth proxy from inter-channel delay and
    energy balance. It is useful for UI situational awareness, not as a
    calibrated acoustic localization claim.
    """

    def __init__(self, sample_rate: int, mic_spacing_m: float = 0.08, sound_speed_mps: float = 343.0):
        self.sample_rate = sample_rate
        self.max_lag = max(1, int(round(sample_rate * mic_spacing_m / sound_speed_mps)))

    def estimate(self, chunk: np.ndarray) -> DirectionEstimate | None:
        data = np.asarray(chunk, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 16:
            return None
        left = data[:, 0] - float(np.mean(data[:, 0]))
        right = data[:, 1] - float(np.mean(data[:, 1]))
        if float(np.sqrt(np.mean(left * left) + np.mean(right * right))) < 1e-6:
            return None
        corr = np.correlate(left, right, mode="full")
        center = len(right) - 1
        lo = max(0, center - self.max_lag)
        hi = min(len(corr), center + self.max_lag + 1)
        window = corr[lo:hi]
        best = int(np.argmax(np.abs(window)))
        lag = (lo + best) - center
        delay_ratio = float(np.clip(lag / max(self.max_lag, 1), -1.0, 1.0))
        energy_ratio = float(
            np.clip((np.sqrt(np.mean(left * left)) - np.sqrt(np.mean(right * right))) / (np.sqrt(np.mean(left * left)) + np.sqrt(np.mean(right * right)) + 1e-9), -1.0, 1.0)
        )
        azimuth = float(np.clip(65.0 * delay_ratio + 25.0 * energy_ratio, -90.0, 90.0))
        confidence = float(np.clip(np.max(np.abs(window)) / (np.sum(np.abs(window)) + 1e-9) * len(window), 0.0, 1.0))
        return DirectionEstimate(azimuth_deg=azimuth, confidence=confidence, lag_samples=lag)
