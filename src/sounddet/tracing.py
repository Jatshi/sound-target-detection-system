from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class TraceLogger:
    """JSONL span logger compatible with later OpenTelemetry ingestion."""

    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[str]:
        trace_id = str(attrs.pop("trace_id", "") or uuid.uuid4())
        start = time.perf_counter()
        try:
            yield trace_id
            status = "ok"
            error = ""
        except Exception as exc:
            status = "error"
            error = str(exc)
            raise
        finally:
            if self.enabled:
                row = {
                    "trace_id": trace_id,
                    "span": name,
                    "status": status,
                    "error": error,
                    "start_unix": time.time(),
                    "duration_ms": (time.perf_counter() - start) * 1000.0,
                    **attrs,
                }
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def event(self, name: str, **attrs) -> None:
        if not self.enabled:
            return
        row = {"trace_id": str(attrs.pop("trace_id", "") or uuid.uuid4()), "span": name, "status": "event", "start_unix": time.time(), **attrs}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
