from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))


DEFAULT_PASSES = [
    "eliminate_identity",
    "eliminate_nop_dropout",
    "eliminate_deadend",
    "eliminate_unused_initializer",
    "fuse_bn_into_conv",
    "fuse_add_bias_into_conv",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize an ONNX graph and write a reproducible report.")
    parser.add_argument("--input", default=str(APP_ROOT / "models" / "edge" / "neurocap_resnet10.onnx"))
    parser.add_argument("--output", default=str(APP_ROOT / "models" / "edge" / "neurocap_resnet10_opt.onnx"))
    parser.add_argument("--passes", default=",".join(DEFAULT_PASSES))
    parser.add_argument("--report", default=str(APP_ROOT / "models" / "edge" / "onnx_optimization_report.json"))
    args = parser.parse_args()

    import onnx

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(src))
    onnx.checker.check_model(model)
    before_nodes = len(model.graph.node)
    before_initializers = len(model.graph.initializer)
    requested = [p.strip() for p in args.passes.split(",") if p.strip()]
    applied: list[str] = []
    skipped: list[str] = []
    optimizer_available = False
    try:
        import onnxoptimizer

        optimizer_available = True
        available = set(onnxoptimizer.get_available_passes())
        applied = [p for p in requested if p in available]
        skipped = [p for p in requested if p not in available]
        if applied:
            model = onnxoptimizer.optimize(model, applied)
    except Exception as exc:
        skipped = requested
        optimizer_error = str(exc)
    else:
        optimizer_error = ""

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    after_nodes = len(model.graph.node)
    after_initializers = len(model.graph.initializer)
    report = {
        "input": str(src),
        "output": str(dst),
        "optimizer_available": optimizer_available,
        "optimizer_error": optimizer_error,
        "requested_passes": requested,
        "applied_passes": applied,
        "skipped_passes": skipped,
        "nodes_before": before_nodes,
        "nodes_after": after_nodes,
        "initializers_before": before_initializers,
        "initializers_after": after_initializers,
        "status": "optimized" if applied else "checked_shape_inferred_only",
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
