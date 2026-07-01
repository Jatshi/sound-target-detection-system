from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ONNX graph latency with precomputed spectrogram input.")
    parser.add_argument("--onnx", default=str(APP_ROOT / "models" / "edge" / "neurocap_resnet10.onnx"))
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out", default=str(APP_ROOT / "models" / "edge" / "benchmark_resnet10_onnx_model_only.csv"))
    args = parser.parse_args()
    providers = [args.provider, "CPUExecutionProvider"] if args.provider != "CPUExecutionProvider" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    spec = np.random.default_rng(2026).normal(size=(1, 1, 64, 87)).astype("float32")
    for _ in range(args.warmup):
        sess.run([output_name], {input_name: spec})
    latencies = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run([output_name], {input_name: spec})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    row = {
        "onnx": args.onnx,
        "requested_provider": args.provider,
        "actual_providers": ";".join(sess.get_providers()),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "iters": args.iters,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

