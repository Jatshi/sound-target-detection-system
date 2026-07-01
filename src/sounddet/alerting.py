from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from . import CLASS_NAMES
from .postprocess import EventRecord


SEVERITY = {0: "critical", 1: "warning", 2: "info", 3: "ignore", 4: "ignore"}


class AlertManager:
    def __init__(self, out_dir: str | Path, webhook_url: str = "", cooldown_sec: float = 2.0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.webhook_url = webhook_url
        self.cooldown_sec = cooldown_sec
        self.last_sent: dict[int, float] = {}

    def handle_event(self, session_id: str, event: EventRecord) -> dict:
        now = time.time()
        if now - self.last_sent.get(event.label, 0.0) < self.cooldown_sec:
            status = "cooldown"
        else:
            payload = {
                "session_id": session_id,
                "class_id": event.label,
                "class_name": CLASS_NAMES[event.label],
                "severity": SEVERITY.get(event.label, "info"),
                "confidence": event.confidence,
                "start": event.start,
                "end": event.end,
                "event": asdict(event),
            }
            self._write_json(payload)
            self._send_webhook(payload)
            self.last_sent[event.label] = now
            status = "sent"
        return {"status": status, "class_name": CLASS_NAMES[event.label], "confidence": event.confidence}

    def _write_json(self, payload: dict) -> None:
        path = self.out_dir / "alerts.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _send_webhook(self, payload: dict) -> None:
        if not self.webhook_url:
            return
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.webhook_url, data=data, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=2).read()
        except Exception:
            pass

