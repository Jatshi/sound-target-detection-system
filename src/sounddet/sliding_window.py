from __future__ import annotations

import numpy as np


class SlidingWindowManager:
    def __init__(self, window_size: int, hop_size: int):
        self.window_size = int(window_size)
        self.hop_size = int(hop_size)
        self.buffer = np.zeros(self.window_size, dtype=np.float32)
        self.samples_seen = 0

    def feed(self, chunk: np.ndarray) -> tuple[bool, np.ndarray, float]:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size != self.hop_size:
            if chunk.size < self.hop_size:
                chunk = np.pad(chunk, (0, self.hop_size - chunk.size))
            else:
                chunk = chunk[: self.hop_size]
        self.buffer[:-self.hop_size] = self.buffer[self.hop_size :]
        self.buffer[-self.hop_size :] = chunk
        self.samples_seen += self.hop_size
        ready = self.samples_seen >= self.window_size
        window_start_sample = max(0, self.samples_seen - self.window_size)
        return ready, self.buffer.copy(), float(window_start_sample)


def iter_windows(wav: np.ndarray, sample_rate: int, window_sec: float = 1.0, hop_sec: float = 0.5):
    window = int(round(sample_rate * window_sec))
    hop = int(round(sample_rate * hop_sec))
    mgr = SlidingWindowManager(window, hop)
    for pos in range(0, len(wav), hop):
        chunk = wav[pos : pos + hop]
        if len(chunk) < hop:
            chunk = np.pad(chunk, (0, hop - len(chunk)))
        ready, buf, start_sample = mgr.feed(chunk)
        if ready:
            yield start_sample / sample_rate, buf
