from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Verify MANIFEST.json hashes.")
    parser.add_argument("--manifest", default=str(ROOT / "MANIFEST.json"))
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in data.get("files", []):
        path = ROOT / item["path"]
        if not path.exists():
            failures.append((item["path"], "missing"))
            continue
        current = sha256(path)
        if current != item["sha256"]:
            failures.append((item["path"], "hash mismatch"))
    if failures:
        for path, reason in failures:
            print(f"{path}: {reason}")
        return 1
    print(f"manifest ok: {len(data.get('files', []))} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

