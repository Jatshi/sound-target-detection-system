from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import psutil


@dataclass
class RuntimeMetrics:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_free_gb: float
    gpu_util_percent: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    gpu_temperature_c: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def read_runtime_metrics(root: str | Path) -> RuntimeMetrics:
    root_path = Path(root)
    disk = shutil.disk_usage(root_path if root_path.exists() else Path.cwd())
    gpu = _read_nvidia_smi()
    return RuntimeMetrics(
        timestamp=time.time(),
        cpu_percent=float(psutil.cpu_percent(interval=0.0)),
        memory_percent=float(psutil.virtual_memory().percent),
        disk_free_gb=float(disk.free / 1024**3),
        **gpu,
    )


def prometheus_lines(metrics: RuntimeMetrics) -> list[str]:
    data = metrics.to_dict()
    lines = []
    for key, value in data.items():
        if key == "timestamp" or value is None:
            continue
        lines.append(f"sounddet_{key} {value}")
    return lines


def _read_nvidia_smi() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        first = result.stdout.strip().splitlines()[0]
        util, mem_used, mem_total, temp = [float(x.strip()) for x in first.split(",")]
        return {
            "gpu_util_percent": util,
            "gpu_memory_used_mb": mem_used,
            "gpu_memory_total_mb": mem_total,
            "gpu_temperature_c": temp,
        }
    except Exception:
        return {}

