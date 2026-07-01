from __future__ import annotations

import asyncio
import json
import math
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import AppConfig, load_config, resolve_app_path
from .engine import DetectionEngine
from .event_store import EventStore
from .model_registry import registry_status
from .reporting import export_session_report
from .system_monitor import prometheus_lines, read_runtime_metrics
from . import CLASS_NAMES


class RuntimeState:
    def __init__(self):
        self.cfg: AppConfig = load_config(os.environ.get("SOUNDDET_CONFIG"))
        self.store = EventStore(self.cfg)
        self.engine: DetectionEngine | None = None
        self.thread: threading.Thread | None = None
        self.model_key = self.cfg.default_model
        self.last_summary: dict[str, Any] = {}
        self.websockets: list[WebSocket] = []
        self.config_history: list[dict[str, Any]] = []


state = RuntimeState()
app = FastAPI(title="Sound Target Detection API", version="1.0.0")
WEB_DIR = resolve_app_path(state.cfg, "web")
if WEB_DIR.exists():
    app.mount("/console", StaticFiles(directory=str(WEB_DIR), html=True), name="console")


def require_token(x_api_token: str | None = Header(default=None)):
    token = state.cfg.api_token
    if token and x_api_token != token:
        raise HTTPException(status_code=401, detail="Invalid API token")


@app.get("/health")
def health():
    return {"ok": True, "running": state.thread is not None and state.thread.is_alive(), "model": state.model_key}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    events = state.store.query("SELECT COUNT(*) AS n FROM events")
    windows = state.store.query("SELECT COUNT(*) AS n FROM window_predictions")
    lat_rows = state.store.query("SELECT latency_ms FROM window_predictions ORDER BY id DESC LIMIT 2000")
    latencies = sorted(float(r["latency_ms"]) for r in lat_rows)
    p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
    p99 = latencies[int(0.99 * (len(latencies) - 1))] if latencies else 0.0
    class_rows = state.store.query("SELECT label_name, COUNT(*) AS n, AVG(confidence) AS c FROM events GROUP BY label_name")
    class_metrics = []
    for row in class_rows:
        label = str(row["label_name"]).replace(" ", "_").lower()
        class_metrics.append(f'sounddet_events_by_class_total{{class="{label}"}} {int(row["n"])}')
        class_metrics.append(f'sounddet_event_confidence_mean{{class="{label}"}} {float(row["c"] or 0.0):.6f}')
    runtime = read_runtime_metrics(state.cfg.app_root)
    return "\n".join(
        [
            f"sounddet_events_total {events[0]['n'] if events else 0}",
            f"sounddet_windows_total {windows[0]['n'] if windows else 0}",
            f"sounddet_running {1 if state.thread is not None and state.thread.is_alive() else 0}",
            f"sounddet_inference_latency_p95_ms {p95:.6f}",
            f"sounddet_inference_latency_p99_ms {p99:.6f}",
            f"sounddet_inference_queue_depth 0",
            f"sounddet_window_dropped_total{{reason=\"backpressure\"}} 0",
            *class_metrics,
            *prometheus_lines(runtime),
        ]
    )


@app.get("/dashboard/overview")
def dashboard_overview():
    runtime = read_runtime_metrics(state.cfg.app_root)
    sessions = state.store.query("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 1")
    events = state.store.query("SELECT * FROM events ORDER BY created_at DESC LIMIT 50")
    windows = state.store.query("SELECT * FROM window_predictions ORDER BY id DESC LIMIT 200")
    class_rows = state.store.query("SELECT label_name, COUNT(*) AS n FROM events GROUP BY label_name")
    latency_rows = list(reversed(windows[-80:])) if windows else []
    latest_summary = _read_latest_summary(sessions[0]["output_dir"]) if sessions else {}
    class_counts = {name: 0 for name in CLASS_NAMES}
    for row in class_rows:
        if row["label_name"] in class_counts:
            class_counts[row["label_name"]] = int(row["n"])
    latest_confidence = float(events[0]["confidence"]) if events else 0.0
    return _json_clean({
        "health": health(),
        "runtime": runtime.to_dict(),
        "config": {
            "sample_rate": state.cfg.sample_rate,
            "window_sec": state.cfg.window_sec,
            "hop_sec": state.cfg.hop_sec,
            "target_threshold": state.cfg.target_threshold,
            "class_thresholds": state.cfg.class_thresholds,
            "ema_alpha": state.cfg.ema_alpha,
            "confirm_frames": state.cfg.confirm_frames,
            "merge_gap_sec": state.cfg.merge_gap_sec,
            "trial_mode": state.cfg.trial_mode,
        },
        "latest_session": sessions[0] if sessions else None,
        "summary": latest_summary,
        "counts": {
            "events": int(state.store.query("SELECT COUNT(*) AS n FROM events")[0]["n"]),
            "windows": int(state.store.query("SELECT COUNT(*) AS n FROM window_predictions")[0]["n"]),
            "latest_confidence": latest_confidence,
            "class_counts": class_counts,
        },
        "recent_events": events,
        "latency_series": [
            {
                "t": float(row["t_start"]),
                "latency_ms": float(row["latency_ms"]),
                "pred_name": row["pred_name"],
                "confidence": float(row["confidence"]),
            }
            for row in latency_rows
        ],
        "models": registry_status(),
    })


@app.get("/")
def root():
    return {"service": "Sound Target Detection API", "docs": "/docs", "console": "/console"}


@app.get("/models")
def models():
    return registry_status()


@app.post("/models/select", dependencies=[Depends(require_token)])
def select_model(payload: dict[str, str]):
    key = payload.get("model_key")
    if not key:
        raise HTTPException(status_code=400, detail="model_key is required")
    known = {m["key"] for m in registry_status()}
    if key not in known:
        raise HTTPException(status_code=404, detail="Unknown model")
    state.model_key = key
    state.store.audit(None, "model_selected", {"model_key": key})
    return {"model_key": key}


@app.get("/config")
def get_config():
    return state.cfg.__dict__


@app.post("/config", dependencies=[Depends(require_token)])
def update_config(payload: dict[str, Any]):
    state.config_history.append(asdict(state.cfg))
    for key, value in payload.items():
        if hasattr(state.cfg, key):
            setattr(state.cfg, key, value)
    state.store.audit(None, "config_updated", payload)
    return state.cfg.__dict__


@app.post("/config/rollback", dependencies=[Depends(require_token)])
def rollback_config():
    if not state.config_history:
        raise HTTPException(status_code=404, detail="No previous config")
    previous = state.config_history.pop()
    for key, value in previous.items():
        if hasattr(state.cfg, key):
            setattr(state.cfg, key, value)
    state.store.audit(None, "config_rollback", previous)
    return state.cfg.__dict__


@app.post("/sessions/start", dependencies=[Depends(require_token)])
def start_session(payload: dict[str, Any] | None = None):
    if state.thread is not None and state.thread.is_alive():
        raise HTTPException(status_code=409, detail="Session already running")
    payload = payload or {}
    model_key = payload.get("model_key", state.model_key)
    trial_mode = payload.get("trial_mode", state.cfg.trial_mode)
    minutes = float(payload.get("minutes", 1.0))
    streams = int(payload.get("streams_per_dataset", 1))
    engine = DetectionEngine(state.cfg, model_key=model_key, store=state.store)
    engine.add_event_callback(lambda ev: asyncio.run(_broadcast({"type": "event", "event": asdict(ev)})))
    state.engine = engine

    def run():
        try:
            state.last_summary = engine.run_streaming_trial(minutes=minutes, streams_per_dataset=streams, trial_mode=trial_mode)
            asyncio.run(_broadcast({"type": "session_complete", "summary": state.last_summary}))
        except Exception as exc:
            state.last_summary = {"error": str(exc)}
            asyncio.run(_broadcast({"type": "session_failed", "error": str(exc)}))

    state.thread = threading.Thread(target=run, daemon=True)
    state.thread.start()
    return {"started": True, "model_key": model_key, "trial_mode": trial_mode}


@app.post("/sessions/stop", dependencies=[Depends(require_token)])
def stop_session():
    if state.engine is not None:
        state.engine.stop()
    return {"stop_requested": True}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    rows = state.store.query("SELECT * FROM sessions WHERE session_id=?", (session_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="Session not found")
    return rows[0]


@app.get("/events")
def get_events(limit: int = 100, offset: int = 0, session_id: str | None = None, label: int | None = None, review_status: str | None = None):
    where = []
    params: list[Any] = []
    if session_id:
        where.append("session_id=?")
        params.append(session_id)
    if label is not None:
        where.append("label=?")
        params.append(label)
    if review_status:
        where.append("review_status=?")
        params.append(review_status)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    params.extend([limit, offset])
    return state.store.query(f"SELECT * FROM events{clause} ORDER BY created_at DESC LIMIT ? OFFSET ?", tuple(params))


@app.post("/events/{event_id}/review", dependencies=[Depends(require_token)])
def review_event(event_id: int, payload: dict[str, str]):
    state.store.review_event(event_id, payload.get("status", "reviewed"), payload.get("note", ""))
    return {"ok": True}


@app.get("/reports/{session_id}")
def report(session_id: str):
    out_dir = resolve_app_path(state.cfg, state.cfg.report_dir) / session_id
    package = export_session_report(state.store, session_id, out_dir)
    return FileResponse(str(package), filename=package.name)


@app.websocket("/stream/events")
async def stream_events(ws: WebSocket):
    await ws.accept()
    state.websockets.append(ws)
    try:
        await ws.send_text(json.dumps({"type": "connected"}))
        while True:
            await asyncio.sleep(2)
            await ws.send_text(json.dumps({"type": "heartbeat", "running": state.thread is not None and state.thread.is_alive()}))
    except WebSocketDisconnect:
        state.websockets.remove(ws)


async def _broadcast(payload: dict[str, Any]) -> None:
    dead = []
    for ws in state.websockets:
        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in state.websockets:
            state.websockets.remove(ws)


def _read_latest_summary(output_dir: str | None) -> dict[str, Any]:
    if not output_dir:
        return {}
    path = Path(output_dir) / "summary.csv"
    if not path.exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_csv(path)
        if df.empty:
            return {}
        row = df.iloc[0].to_dict()
        return {str(k): (float(v) if isinstance(v, (int, float)) else v) for k, v in row.items()}
    except Exception:
        return {}


def _json_clean(value):
    if isinstance(value, dict):
        return {k: _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
