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

from sounddet.batching import ContinuousBatchingScheduler
from sounddet.edge_backend import EdgeInferenceBackend


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else float("nan")


def run_direct(backend: EdgeInferenceBackend, streams: int, windows_per_stream: int, rng: np.random.Generator) -> dict:
    lat = []
    t0 = time.perf_counter()
    for _ in range(windows_per_stream):
        for _sid in range(streams):
            window = rng.normal(0, 0.05, 44100).astype("float32")
            a = time.perf_counter()
            backend.predict_batch([window])
            lat.append((time.perf_counter() - a) * 1000.0)
    elapsed = time.perf_counter() - t0
    return {
        "mode": "direct_single_window",
        "streams": streams,
        "windows": streams * windows_per_stream,
        "elapsed_sec": elapsed,
        "latency_p50_ms": pct(lat, 50),
        "latency_p95_ms": pct(lat, 95),
        "latency_p99_ms": pct(lat, 99),
        "throughput_windows_per_sec": len(lat) / elapsed,
        "mean_batch_size": 1.0,
        "queue_wait_ms": 0.0,
        "dropped": 0,
    }


def run_batched(backend: EdgeInferenceBackend, streams: int, windows_per_stream: int, max_batch_size: int, max_wait_ms: float, rng: np.random.Generator) -> dict:
    items = []
    for step in range(windows_per_stream):
        for sid in range(streams):
            items.append((sid, step, rng.normal(0, 0.05, 44100).astype("float32")))
    scheduler = ContinuousBatchingScheduler(backend.predict_batch, max_batch_size=max_batch_size, max_wait_ms=max_wait_ms, max_queue_size=max(4096, streams * windows_per_stream * 2))
    scheduler.start()
    lat = []
    t0 = time.perf_counter()
    futures = []
    submitted = []
    for sid, step, window in items:
        submitted_at = time.perf_counter()
        futures.append(scheduler.submit(f"stream_{sid:02d}", step * 0.5, window))
        submitted.append(submitted_at)
    for fut, submitted_at in zip(futures, submitted):
        fut.result(timeout=30.0)
        lat.append((time.perf_counter() - submitted_at) * 1000.0)
    scheduler.stop()
    elapsed = time.perf_counter() - t0
    return {
        "mode": "continuous_batching",
        "streams": streams,
        "windows": streams * windows_per_stream,
        "elapsed_sec": elapsed,
        "latency_p50_ms": pct(lat, 50),
        "latency_p95_ms": pct(lat, 95),
        "latency_p99_ms": pct(lat, 99),
        "throughput_windows_per_sec": len(lat) / elapsed,
        "mean_batch_size": scheduler.stats.mean_batch_size,
        "queue_wait_ms": scheduler.stats.mean_queue_wait_ms,
        "dropped": scheduler.stats.dropped,
        "max_queue_depth": scheduler.stats.max_queue_depth,
        "max_batch_size": max_batch_size,
        "max_wait_ms": max_wait_ms,
    }


def run_hop_synchronous(backend: EdgeInferenceBackend, streams: int, windows_per_stream: int, rng: np.random.Generator) -> dict:
    lat = []
    t0 = time.perf_counter()
    total = 0
    for _step in range(windows_per_stream):
        windows = [rng.normal(0, 0.05, 44100).astype("float32") for _ in range(streams)]
        a = time.perf_counter()
        backend.predict_batch(windows)
        elapsed_ms = (time.perf_counter() - a) * 1000.0
        lat.extend([elapsed_ms] * streams)
        total += streams
    elapsed = time.perf_counter() - t0
    return {
        "mode": "hop_synchronous_batch",
        "streams": streams,
        "windows": total,
        "elapsed_sec": elapsed,
        "latency_p50_ms": pct(lat, 50),
        "latency_p95_ms": pct(lat, 95),
        "latency_p99_ms": pct(lat, 99),
        "throughput_windows_per_sec": total / elapsed,
        "mean_batch_size": float(streams),
        "queue_wait_ms": 0.0,
        "dropped": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare direct single-window inference with cross-stream continuous batching.")
    parser.add_argument("--backend", default="onnxruntime-cuda", choices=["pytorch", "onnxruntime-cuda", "onnxruntime-cpu", "onnxruntime-tensorrt"])
    parser.add_argument("--model", default="neurocap_resnet10")
    parser.add_argument("--model-path", default=str(APP_ROOT / "models" / "edge" / "neurocap_resnet10_opt.onnx"))
    parser.add_argument("--streams", default="1,4,8,16,32")
    parser.add_argument("--windows-per-stream", type=int, default=20)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-wait-ms", type=float, default=8.0)
    parser.add_argument("--out", default=str(APP_ROOT / "outputs" / "infra_benchmarks" / "continuous_batching_summary.csv"))
    args = parser.parse_args()

    model_path = args.model_path if args.backend.startswith("onnxruntime") else None
    backend = EdgeInferenceBackend(args.model, args.backend, model_path=model_path)
    rows = []
    for streams in [int(x) for x in args.streams.split(",") if x.strip()]:
        rng = np.random.default_rng(2026 + streams)
        rows.append(run_direct(backend, streams, args.windows_per_stream, rng))
        rng = np.random.default_rng(2026 + streams)
        rows.append(run_hop_synchronous(backend, streams, args.windows_per_stream, rng))
        rng = np.random.default_rng(2026 + streams)
        rows.append(run_batched(backend, streams, args.windows_per_stream, args.max_batch_size, args.max_wait_ms, rng))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "backend", args.backend)
    df.insert(1, "model", args.model)
    df.to_csv(out, index=False)
    report = out.with_suffix(".md")
    best = df.sort_values("throughput_windows_per_sec", ascending=False).iloc[0].to_dict()
    direct_32 = df[(df["mode"] == "direct_single_window") & (df["streams"] == max(df["streams"]))].iloc[0]
    batch_32 = df[(df["mode"] == "continuous_batching") & (df["streams"] == max(df["streams"]))].iloc[0]
    speedup = float(batch_32["throughput_windows_per_sec"] / max(1e-9, direct_32["throughput_windows_per_sec"]))
    report.write_text(
        "\n".join(
            [
                "# Continuous Batching Benchmark",
                "",
                f"- Backend: `{args.backend}`",
                f"- Model: `{args.model}`",
                f"- Max batch size: `{args.max_batch_size}`",
                f"- Max wait: `{args.max_wait_ms} ms`",
                f"- Best throughput: `{best['throughput_windows_per_sec']:.2f}` windows/s",
                f"- Highest-stream throughput speedup: `{speedup:.2f}x`",
                "",
                "Detailed CSV: `" + str(out) + "`",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
