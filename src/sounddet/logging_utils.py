from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import AppConfig, resolve_app_path


def setup_logging(cfg: AppConfig, name: str = "sounddet") -> logging.Logger:
    log_dir = resolve_app_path(cfg, cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    info_file = RotatingFileHandler(log_dir / "sounddet.log", maxBytes=10_000_000, backupCount=30, encoding="utf-8")
    info_file.setFormatter(fmt)
    err_file = RotatingFileHandler(log_dir / "errors.log", maxBytes=5_000_000, backupCount=30, encoding="utf-8")
    err_file.setLevel(logging.ERROR)
    err_file.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(info_file)
    logger.addHandler(err_file)
    return logger
