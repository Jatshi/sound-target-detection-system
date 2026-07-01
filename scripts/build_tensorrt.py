from __future__ import annotations

import argparse
import json
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a TensorRT build command for an exported ONNX model.")
    parser.add_argument("--onnx", default=str(APP_ROOT / "models" / "edge" / "neurocap_full.onnx"))
    parser.add_argument("--engine", default=str(APP_ROOT / "models" / "edge" / "neurocap_full.trt"))
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    flags = ["trtexec", f"--onnx={args.onnx}", f"--saveEngine={args.engine}", "--explicitBatch"]
    if args.fp16:
        flags.append("--fp16")
    report = {
        "status": "manual_build_required",
        "reason": "TensorRT Python/runtime is device-specific and is not installed in this environment.",
        "command": " ".join(flags),
    }
    out = Path(args.engine).with_suffix(".build.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

