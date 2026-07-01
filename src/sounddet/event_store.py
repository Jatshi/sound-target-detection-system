from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from . import CLASS_NAMES
from .config import AppConfig, resolve_app_path


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  started_at REAL NOT NULL,
  ended_at REAL,
  model_key TEXT NOT NULL,
  input_source TEXT NOT NULL,
  config_json TEXT NOT NULL,
  output_dir TEXT NOT NULL,
  status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  stream_id TEXT,
  dataset TEXT,
  start REAL NOT NULL,
  end REAL NOT NULL,
  label INTEGER NOT NULL,
  label_name TEXT NOT NULL,
  confidence REAL NOT NULL,
  clip_path TEXT,
  matched INTEGER,
  review_status TEXT DEFAULT 'unreviewed',
  review_note TEXT DEFAULT '',
  created_at REAL NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE TABLE IF NOT EXISTS window_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  stream_id TEXT,
  dataset TEXT,
  t_start REAL NOT NULL,
  t_end REAL NOT NULL,
  pred INTEGER NOT NULL,
  pred_name TEXT NOT NULL,
  confidence REAL NOT NULL,
  probs_json TEXT NOT NULL,
  latency_ms REAL NOT NULL,
  label INTEGER,
  created_at REAL NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE TABLE IF NOT EXISTS audio_clips (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  event_id INTEGER,
  path TEXT NOT NULL,
  start REAL,
  end REAL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS model_registry (
  model_key TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  checkpoint TEXT NOT NULL,
  mode TEXT NOT NULL,
  available INTEGER NOT NULL,
  sha256 TEXT,
  checked_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  action TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_windows_session ON window_predictions(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_logs(session_id);
"""


class EventStore:
    def __init__(self, cfg_or_output: AppConfig | str | Path):
        if isinstance(cfg_or_output, AppConfig):
            self.cfg = cfg_or_output
            self.output_dir = resolve_app_path(cfg_or_output, cfg_or_output.output_dir)
            self.db_path = resolve_app_path(cfg_or_output, cfg_or_output.db_path)
        else:
            self.cfg = None
            self.output_dir = Path(cfg_or_output)
            self.db_path = self.output_dir / "sounddet.db"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def write_csv(self, name: str, rows) -> Path:
        path = self.output_dir / name
        data = [asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in rows]
        pd.DataFrame(data).to_csv(path, index=False)
        return path

    @staticmethod
    def _value(obj, key: str, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def start_session(self, session_id: str, model_key: str, input_source: str, config: dict[str, Any], output_dir: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?, ?, NULL, ?, ?, ?, ?, ?)",
                (session_id, time.time(), model_key, input_source, json.dumps(config, ensure_ascii=False), output_dir, "running"),
            )
        self.audit(session_id, "session_started", {"model_key": model_key, "input_source": input_source})

    def end_session(self, session_id: str, status: str = "complete") -> None:
        with self.connect() as conn:
            conn.execute("UPDATE sessions SET ended_at=?, status=? WHERE session_id=?", (time.time(), status, session_id))
        self.audit(session_id, "session_ended", {"status": status})

    def insert_events(self, session_id: str, events, clip_paths: dict[int, str] | None = None) -> None:
        clip_paths = clip_paths or {}
        rows = []
        for i, e in enumerate(events):
            label = int(self._value(e, "label"))
            rows.append(
                (
                    session_id,
                    self._value(e, "stream_id", ""),
                    self._value(e, "dataset", ""),
                    float(self._value(e, "start")),
                    float(self._value(e, "end")),
                    label,
                    CLASS_NAMES[label],
                    float(self._value(e, "confidence", 0.0)),
                    clip_paths.get(i),
                    None,
                    time.time(),
                )
            )
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO events(session_id, stream_id, dataset, start, end, label, label_name, confidence, clip_path, matched, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            event_ids = [row["id"] for row in conn.execute("SELECT id FROM events WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, len(rows))).fetchall()]
            event_ids = list(reversed(event_ids))
            clip_rows = []
            for event_id, row in zip(event_ids, rows):
                clip_path = row[8]
                if clip_path:
                    clip_rows.append((session_id, event_id, clip_path, row[3], row[4], time.time()))
            if clip_rows:
                conn.executemany(
                    "INSERT INTO audio_clips(session_id, event_id, path, start, end, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    clip_rows,
                )

    def insert_windows(self, session_id: str, windows) -> None:
        rows = []
        for w in windows:
            pred = int(self._value(w, "pred"))
            rows.append(
                (
                    session_id,
                    self._value(w, "stream_id", ""),
                    self._value(w, "dataset", ""),
                    float(self._value(w, "t_start")),
                    float(self._value(w, "t_end")),
                    pred,
                    CLASS_NAMES[pred],
                    float(self._value(w, "confidence", 0.0)),
                    json.dumps(self._value(w, "probs", [])),
                    float(self._value(w, "latency_ms", 0.0)),
                    self._value(w, "label", None),
                    time.time(),
                )
            )
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO window_predictions(session_id, stream_id, dataset, t_start, t_end, pred, pred_name, confidence, probs_json, latency_ms, label, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def upsert_model(self, model_key: str, label: str, checkpoint: str, mode: str, available: bool, sha256: str | None) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO model_registry VALUES (?, ?, ?, ?, ?, ?, ?)",
                (model_key, label, checkpoint, mode, int(available), sha256, time.time()),
            )

    def review_event(self, event_id: int, status: str, note: str = "") -> None:
        with self.connect() as conn:
            conn.execute("UPDATE events SET review_status=?, review_note=? WHERE id=?", (status, note, event_id))
        self.audit(None, "event_reviewed", {"event_id": event_id, "status": status, "note": note})

    def audit(self, session_id: str | None, action: str, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO audit_logs(session_id, action, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (session_id, action, json.dumps(payload, ensure_ascii=False), time.time()),
            )

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]
