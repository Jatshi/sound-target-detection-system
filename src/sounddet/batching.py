from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .model_adapter import Prediction


@dataclass
class InferenceRequest:
    stream_id: str
    timestamp: float
    window: np.ndarray
    future: Future
    enqueue_time: float


@dataclass
class BatchStats:
    batches: int = 0
    windows: int = 0
    dropped: int = 0
    max_queue_depth: int = 0
    total_queue_wait_ms: float = 0.0

    @property
    def mean_batch_size(self) -> float:
        return self.windows / self.batches if self.batches else 0.0

    @property
    def mean_queue_wait_ms(self) -> float:
        return self.total_queue_wait_ms / self.windows if self.windows else 0.0


class ContinuousBatchingScheduler:
    """Small GPU-serving style batcher for multiple audio streams.

    Each stream can enqueue one window at a time. The scheduler flushes when it
    reaches `max_batch_size` or the oldest request has waited `max_wait_ms`.
    """

    def __init__(
        self,
        infer_fn: Callable[[list[np.ndarray]], list[Prediction]],
        max_batch_size: int = 8,
        max_wait_ms: float = 8.0,
        max_queue_size: int = 512,
    ):
        self.infer_fn = infer_fn
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_ms = max(0.0, float(max_wait_ms))
        self.queue: queue.Queue[InferenceRequest] = queue.Queue(maxsize=max_queue_size)
        self.stats = BatchStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, drain: bool = True) -> None:
        if drain:
            while not self.queue.empty():
                time.sleep(0.001)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def submit(self, stream_id: str, timestamp: float, window: np.ndarray) -> Future:
        future: Future = Future()
        req = InferenceRequest(stream_id, timestamp, window, future, time.perf_counter())
        try:
            self.queue.put_nowait(req)
            self.stats.max_queue_depth = max(self.stats.max_queue_depth, self.queue.qsize())
        except queue.Full:
            self.stats.dropped += 1
            future.set_exception(RuntimeError("continuous batching queue is full"))
        return future

    def _loop(self) -> None:
        pending: list[InferenceRequest] = []
        oldest = 0.0
        while not self._stop.is_set() or pending or not self.queue.empty():
            timeout = 0.001
            if pending and self.max_wait_ms > 0:
                elapsed_ms = (time.perf_counter() - oldest) * 1000.0
                timeout = max(0.0, (self.max_wait_ms - elapsed_ms) / 1000.0)
            try:
                req = self.queue.get(timeout=timeout)
                if not pending:
                    oldest = req.enqueue_time
                pending.append(req)
            except queue.Empty:
                pass
            if not pending:
                continue
            waited_ms = (time.perf_counter() - oldest) * 1000.0
            should_flush = (
                len(pending) >= self.max_batch_size
                or (waited_ms >= self.max_wait_ms and self.queue.empty())
                or (self._stop.is_set() and self.queue.empty())
            )
            if should_flush:
                self._flush(pending)
                pending = []
                oldest = 0.0

    def _flush(self, reqs: list[InferenceRequest]) -> None:
        now = time.perf_counter()
        self.stats.batches += 1
        self.stats.windows += len(reqs)
        self.stats.total_queue_wait_ms += sum((now - r.enqueue_time) * 1000.0 for r in reqs)
        try:
            preds = self.infer_fn([r.window for r in reqs])
            for req, pred in zip(reqs, preds):
                req.future.set_result(pred)
        except Exception as exc:
            for req in reqs:
                req.future.set_exception(exc)
