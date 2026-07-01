from __future__ import annotations

import time
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import CLASS_NAMES
from .features import trim_or_pad
from .light_baselines import LightBaselineWrapper
from .neurocap_model import TARGET_SAMPLES, WaveToSpec, AudioBaselineWrapper, DeformableNet, DilatedNet, NeuroCAPQARN, ResNet10_TimeAware


@dataclass
class Prediction:
    pred: int
    confidence: float
    probs: list[float]
    latency_ms: float
    model_mode: str


class ModelAdapter:
    def __init__(self, checkpoint: str | Path, mode: str = "neurocap_full", device: str | None = None, batch_size: int = 64):
        self.checkpoint = Path(checkpoint)
        self.mode = mode
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._build_model(mode).to(self.device)
        payload = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.wav2spec = WaveToSpec().to(self.device).eval()
        self.calibration = self._load_calibration()

    def _load_calibration(self) -> dict:
        cfg_path = Path(__file__).resolve().parents[2] / "reference" / "calibration" / "selected_calibration.json"
        if not cfg_path.exists():
            return {"alpha": 1.0, "uncertainty_min": 0.0, "gate_min": 0.0}
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        while "base_global_calibration" in cfg:
            cfg = cfg["base_global_calibration"]
        return cfg

    def _build_model(self, mode: str):
        if mode in {"neurocap_full", "neurocap_sound_only"}:
            return NeuroCAPQARN(num_classes=len(CLASS_NAMES), cond_dim=527)
        if mode == "neurocap_resnet10":
            return NeuroCAPQARN(num_classes=len(CLASS_NAMES), cond_dim=527, backbone="resnet10")
        if mode == "baseline_deformable":
            return AudioBaselineWrapper(DeformableNet(num_classes=len(CLASS_NAMES), out_dim=527))
        if mode == "baseline_resnet10":
            return AudioBaselineWrapper(ResNet10_TimeAware(num_classes=len(CLASS_NAMES), out_dim=527))
        if mode == "baseline_dilated":
            return AudioBaselineWrapper(DilatedNet(num_classes=len(CLASS_NAMES), out_dim=527))
        if mode == "baseline_paulnet":
            return LightBaselineWrapper("paulnet", num_classes=len(CLASS_NAMES))
        if mode == "baseline_gtcnn":
            return LightBaselineWrapper("gtcnn", num_classes=len(CLASS_NAMES))
        if mode == "baseline_dualbranch":
            return LightBaselineWrapper("dualbranch", num_classes=len(CLASS_NAMES))
        raise ValueError(f"Unknown model mode: {mode}")

    def _logits_from_spec(self, spec: torch.Tensor) -> torch.Tensor:
        if self.mode == "neurocap_sound_only":
            _, sound_logits, *_ = self.model(spec, return_gate=True)
            return sound_logits
        if self.mode in {"neurocap_full", "neurocap_resnet10"}:
            main, sound, _neuro, _gate_logits, _lat, _feat, eff_gate, _aux = self.model(spec, return_gate=True)
            prob = torch.softmax(sound, dim=1)
            uncertainty = 1.0 - prob.max(dim=1).values
            guard = ((uncertainty >= float(self.calibration.get("uncertainty_min", 0.0))) & (eff_gate.squeeze(1) >= float(self.calibration.get("gate_min", 0.0)))).float().unsqueeze(1)
            alpha = float(self.calibration.get("alpha", 1.0))
            return sound + alpha * guard * (main - sound)
        if self.mode.startswith("baseline_"):
            return self.model(spec)
        out = self.model(spec)
        return out[0] if isinstance(out, (tuple, list)) else out

    @torch.no_grad()
    def predict_batch(self, windows: list[np.ndarray]) -> list[Prediction]:
        if not windows:
            return []
        preds: list[Prediction] = []
        for start in range(0, len(windows), self.batch_size):
            batch_np = windows[start : start + self.batch_size]
            wav = torch.from_numpy(np.stack(batch_np)).float().unsqueeze(1)
            wav = trim_or_pad(wav, TARGET_SAMPLES).to(self.device)
            t0 = time.perf_counter()
            spec = self.wav2spec(wav)
            logits = self._logits_from_spec(spec)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            latency = (time.perf_counter() - t0) * 1000.0 / max(1, len(batch_np))
            for row in probs:
                pred = int(row.argmax())
                preds.append(Prediction(pred=pred, confidence=float(row[pred]), probs=row.astype(float).tolist(), latency_ms=latency, model_mode=self.mode))
        return preds

    def predict_one(self, window: np.ndarray) -> Prediction:
        return self.predict_batch([window])[0]
