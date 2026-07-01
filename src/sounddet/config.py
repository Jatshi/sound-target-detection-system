from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path


@dataclass
class AppConfig:
    sample_rate: int = 44100
    window_sec: float = 1.0
    hop_sec: float = 0.5
    target_threshold: float = 0.50
    class_thresholds: tuple[float, float, float] | None = None
    ema_alpha: float = 0.40
    confirm_frames: int = 2
    merge_gap_sec: float = 0.80
    match_tolerance_sec: float = 0.50
    default_model: str = "neurocap_full"
    stream_minutes: float = 5.0
    streams_per_dataset: int = 5
    trial_mode: str = "stream"
    seed: int = 2026
    dataset_root: str = r"datasets/ood_final_datasets"
    output_dir: str = r"outputs\online_eval"
    app_root: str = ""
    db_path: str = r"data\sounddet.db"
    log_dir: str = "logs"
    clip_dir: str = r"outputs\audio_clips"
    report_dir: str = r"outputs\reports"
    service_host: str = "127.0.0.1"
    service_port: int = 8765
    api_token: str = ""
    clip_context_sec: float = 3.0
    health_silence_rms: float = 1e-4
    health_clip_peak: float = 0.99
    live_background_rms_threshold: float = 0.04
    alert_webhook_url: str = ""
    alert_cooldown_sec: float = 2.0
    runtime_health_dir: str = r"outputs\health"
    inference_backend: str = "pytorch"
    continuous_batch_max_size: int = 8
    continuous_batch_max_wait_ms: float = 8.0
    dynamic_threshold_enabled: bool = False
    dynamic_threshold_window_sec: float = 60.0
    dynamic_threshold_std_scale: float = 3.0
    trace_enabled: bool = False
    trace_dir: str = r"outputs\traces"

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.window_sec))

    @property
    def hop_samples(self) -> int:
        return int(round(self.sample_rate * self.hop_sec))


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg = AppConfig()
    cfg.app_root = str(Path(__file__).resolve().parents[2])
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    path = Path(path)
    if not path.exists():
        return cfg
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in values.items():
        _set_value(cfg, key, value)
    for key in list(asdict(cfg).keys()):
        env = os.environ.get(f"SOUNDDET_{key.upper()}")
        if env is not None:
            _set_value(cfg, key, env)
    return cfg


def _set_value(cfg: AppConfig, key: str, value: str) -> None:
    if not hasattr(cfg, key):
        return
    old = getattr(cfg, key)
    if isinstance(old, bool):
        setattr(cfg, key, str(value).lower() in {"1", "true", "yes", "on"})
    elif isinstance(old, int):
        setattr(cfg, key, int(float(value)))
    elif isinstance(old, float):
        setattr(cfg, key, float(value))
    elif key == "class_thresholds":
        parts = [x.strip() for x in str(value).split(",") if x.strip()]
        setattr(cfg, key, tuple(float(x) for x in parts[:3]) if parts else None)
    else:
        setattr(cfg, key, str(value))


def resolve_app_path(cfg: AppConfig, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return Path(cfg.app_root) / p
