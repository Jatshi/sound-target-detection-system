from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Triton model repository for the ONNX classifier graph.")
    parser.add_argument("--onnx", default=str(APP_ROOT / "models" / "edge" / "neurocap_resnet10_opt.onnx"))
    parser.add_argument("--fallback", default=str(APP_ROOT / "models" / "edge" / "neurocap_resnet10.onnx"))
    parser.add_argument("--repo", default=str(APP_ROOT / "deploy" / "triton" / "model_repository"))
    args = parser.parse_args()

    src = Path(args.onnx)
    if not src.exists():
        src = Path(args.fallback)
    if not src.exists():
        raise FileNotFoundError(f"No ONNX model found: {args.onnx} or {args.fallback}")
    target_dir = Path(args.repo) / "neurocap_resnet10_onnx" / "1"
    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / "model.onnx"
    shutil.copy2(src, dst)
    meta = {"source": str(src), "target": str(dst), "sha256": sha256(dst), "bytes": dst.stat().st_size}
    (target_dir.parent / "model_package.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
