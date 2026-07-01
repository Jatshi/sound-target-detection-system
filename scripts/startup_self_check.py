from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import torch

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from sounddet.audio_stream import list_audio_devices
from sounddet.config import load_config, resolve_app_path
from sounddet.model_registry import registry_status
from sounddet.system_monitor import read_runtime_metrics


def main() -> int:
    cfg = load_config()
    checks = {
        "timestamp": time.time(),
        "app_root": cfg.app_root,
        "cuda_available": torch.cuda.is_available(),
        "audio_devices": list_audio_devices(),
        "models": registry_status(),
        "runtime": read_runtime_metrics(cfg.app_root).to_dict(),
        "database_ok": False,
        "output_ok": False,
    }
    try:
        db_path = resolve_app_path(cfg, cfg.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("SELECT 1")
        checks["database_ok"] = True
    except Exception as exc:
        checks["database_error"] = str(exc)
    try:
        out = resolve_app_path(cfg, "outputs")
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks["output_ok"] = True
    except Exception as exc:
        checks["output_error"] = str(exc)
    checks["ok"] = checks["database_ok"] and checks["output_ok"] and all(m["available"] for m in checks["models"] if m["key"] in {"neurocap_full", "neurocap_sound_only"})
    out_file = APP_ROOT / "outputs" / "startup_self_check.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": checks["ok"], "report": str(out_file)}, indent=2))
    return 0 if checks["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

