from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from sounddet.edge_backend import EdgeInferenceBackend


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values), p))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch/ONNX/TensorRT edge inference backends.")
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "onnxruntime-cuda", "onnxruntime-cpu", "onnxruntime-tensorrt", "tensorrt"])
    parser.add_argument("--model", default="neurocap_full")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--batch-sizes", default="1,4,16,64")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--deformable-policy", choices=["native", "static-conv"], default="native")
    parser.add_argument("--out", default=str(APP_ROOT / "models" / "edge" / "backend_benchmark.csv"))
    args = parser.parse_args()

    backend = EdgeInferenceBackend(args.model, args.backend, model_path=args.model_path, deformable_policy=args.deformable_policy)
    rows = []
    rng = np.random.default_rng(2026)
    for batch_size in [int(x) for x in args.batch_sizes.split(",") if x.strip()]:
        windows = [rng.normal(0, 0.05, 44100).astype("float32") for _ in range(batch_size)]
        for _ in range(args.warmup):
            backend.predict_batch(windows)
        latencies = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            backend.predict_batch(windows)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        rows.append(
            {
                "backend": args.backend,
                "model": args.model,
                "batch_size": batch_size,
                "iters": args.iters,
                "latency_p50_ms": percentile(latencies, 50),
                "latency_p95_ms": percentile(latencies, 95),
                "latency_p99_ms": percentile(latencies, 99),
                "throughput_windows_per_sec": batch_size / (statistics.mean(latencies) / 1000.0),
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
