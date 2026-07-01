from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import CLASS_NAMES


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_model_package(path: str | Path, model_key: str, backend: str, artifact: str | Path, input_shape: list[int], metrics: dict | None = None) -> Path:
    artifact_path = Path(artifact)
    package = {
        "model_key": model_key,
        "backend": backend,
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path) if artifact_path.exists() else None,
        "input_shape": input_shape,
        "classes": CLASS_NAMES,
        "created_at": time.time(),
        "metrics": metrics or {},
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(package, indent=2), encoding="utf-8")
    return out


def verify_model_package(path: str | Path) -> dict:
    package = json.loads(Path(path).read_text(encoding="utf-8"))
    artifact = Path(package["artifact"])
    package["verified"] = artifact.exists() and sha256(artifact) == package.get("artifact_sha256")
    package["class_order_ok"] = package.get("classes") == CLASS_NAMES
    return package

