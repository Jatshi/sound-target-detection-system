from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from sounddet.edge_backend import backend_summary, build_deploy_module
from sounddet.model_package import write_model_package


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a deployable ONNX classification graph.")
    parser.add_argument("--model", default="neurocap_full")
    parser.add_argument("--mode", choices=["spec"], default="spec", help="Only spec input is production-supported in this release.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--deformable-policy", choices=["native", "static-conv"], default="static-conv")
    parser.add_argument("--dynamic-batch", action="store_true", help="Try dynamic batch export. Default is fully static batch=1 for edge reliability.")
    args = parser.parse_args()

    out = Path(args.out) if args.out else APP_ROOT / "models" / "edge" / f"{args.model}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path = out.parent / "export_report.json"
    started = time.time()
    report = {
        "model": args.model,
        "mode": args.mode,
        "onnx_path": str(out),
        "opset": args.opset,
        "deformable_policy": args.deformable_policy,
        "status": "started",
        "started_at": started,
    }
    try:
        model, wav2spec, info = build_deploy_module(args.model, args.device, deformable_policy=args.deformable_policy)
        device = next(model.parameters()).device
        dummy_wav = torch.zeros(args.batch_size, 1, 44100, device=device)
        with torch.no_grad():
            dummy_spec = wav2spec(dummy_wav)
            logits = model(dummy_spec)
        torch.onnx.export(
            model,
            dummy_spec,
            str(out),
            input_names=["spec"],
            output_names=["logits"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes={"spec": {0: "batch"}, "logits": {0: "batch"}} if args.dynamic_batch else None,
        )
        report.update(
            {
                "status": "success",
                "input_shape": list(dummy_spec.shape),
                "output_shape": list(logits.shape),
                "backend": backend_summary(type("B", (), {"info": info})()),
                "elapsed_sec": time.time() - started,
            }
        )
        package_path = out.parent / f"{args.model}_model_package.json"
        write_model_package(package_path, args.model, "onnxruntime", out, list(dummy_spec.shape), metrics={"export_elapsed_sec": report["elapsed_sec"]})
        report["model_package"] = str(package_path)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_sec": time.time() - started,
            }
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, indent=2))
        return 1
    finally:
        if report.get("status") == "success":
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
