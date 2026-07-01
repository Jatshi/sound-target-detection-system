from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest = []
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        rel_text = str(rel)
        parts = set(rel.parts)
        if parts.intersection({"__pycache__", "outputs", "logs", "data", "build", "dist", "deliverables"}):
            continue
        if rel_text == "MANIFEST.json":
            continue
        if (
            p.is_file()
        ):
            manifest.append({"path": rel_text, "bytes": p.stat().st_size, "sha256": sha256(p)})
    out = ROOT / "MANIFEST.json"
    out.write_text(json.dumps({"files": manifest}, indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
