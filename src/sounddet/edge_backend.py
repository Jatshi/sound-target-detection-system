from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import CLASS_NAMES
from .features import trim_or_pad
from .model_adapter import Prediction
from .model_registry import registry
from .neurocap_model import LATENT_DIM, ODE_STEPS, TARGET_SAMPLES, DeformableConv2d, WaveToSpec


@dataclass
class BackendInfo:
    backend: str
    model_key: str
    model_path: str
    device: str
    input_mode: str
    available: bool
    message: str = ""


class DeterministicNeuroCAPWrapper(nn.Module):
    """Deployment wrapper with deterministic virtual-EEG generation."""

    def __init__(self, model: nn.Module, calibration: dict | None = None, mode: str = "neurocap_full"):
        super().__init__()
        self.model = model
        self.calibration = calibration or {"alpha": 1.0, "uncertainty_min": 0.0, "gate_min": 0.0}
        self.mode = mode

    def deterministic_latent_and_eeg(self, cond_seq: torch.Tensor):
        b = cond_seq.shape[0]
        curr_z = torch.zeros(b, LATENT_DIM, device=cond_seq.device, dtype=cond_seq.dtype)
        dt = 1.0 / ODE_STEPS
        for i in range(ODE_STEPS):
            t = torch.ones(b, 1, device=cond_seq.device, dtype=cond_seq.dtype) * (i * dt)
            curr_z = curr_z + self.model.flow_net(curr_z, t, cond_seq) * dt
        return curr_z, self.model.vae.decode(curr_z)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        _, feat_global, feat_time = self.model.sound_encoder(spec)
        if self.mode == "neurocap_sound_only":
            return self.model.sound_head(feat_global)
        _z_gen, virtual_eeg = self.deterministic_latent_and_eeg(feat_time)
        feat_n_seq, feat_cog_64 = self.model.interpret_seq_and_global(virtual_eeg)
        sound_logits = self.model.sound_head(feat_global)
        main, _residual, _gate_logits, _quality_logits, eff_gate, _a_emb, _n_emb, _align = self.model.qarn_fusion(
            feat_time, feat_n_seq, feat_global, feat_cog_64, sound_logits, neuro_mode="full"
        )
        prob = torch.softmax(sound_logits, dim=1)
        uncertainty = 1.0 - prob.max(dim=1).values
        guard = (
            (uncertainty >= float(self.calibration.get("uncertainty_min", 0.0)))
            & (eff_gate.squeeze(1) >= float(self.calibration.get("gate_min", 0.0)))
        ).float().unsqueeze(1)
        alpha = float(self.calibration.get("alpha", 1.0))
        return sound_logits + alpha * guard * (main - sound_logits)


class StaticDeformableConv2d(nn.Module):
    """ONNX-compatible approximation of DeformableConv2d using its learned kernel."""

    def __init__(self, source: DeformableConv2d):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.stride = source.stride
        self.padding = source.padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, bias=None, stride=self.stride, padding=self.padding)


def replace_deformable_with_static(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, DeformableConv2d):
            setattr(module, name, StaticDeformableConv2d(child))
        else:
            replace_deformable_with_static(child)
    return module


def load_calibration(app_root: Path) -> dict:
    path = app_root / "reference" / "calibration" / "selected_calibration.json"
    if not path.exists():
        return {"alpha": 1.0, "uncertainty_min": 0.0, "gate_min": 0.0}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    while "base_global_calibration" in cfg:
        cfg = cfg["base_global_calibration"]
    return cfg


def build_deploy_module(model_key: str, device: str | None = None, deformable_policy: str = "native") -> tuple[nn.Module, WaveToSpec, BackendInfo]:
    from .model_adapter import ModelAdapter

    spec = registry()[model_key]
    if not spec.available:
        raise FileNotFoundError(f"Model checkpoint unavailable: {spec.checkpoint}")
    adapter = ModelAdapter(spec.checkpoint, mode=spec.mode, device=device)
    if spec.mode.startswith("baseline_"):
        info = BackendInfo(
            backend="pytorch-deploy",
            model_key=model_key,
            model_path=str(spec.checkpoint),
            device=str(adapter.device),
            input_mode="spec",
            available=True,
            message="baseline_direct",
        )
        return adapter.model.eval(), adapter.wav2spec, info
    if deformable_policy == "static-conv":
        adapter.model = replace_deformable_with_static(adapter.model)
    elif deformable_policy != "native":
        raise ValueError(f"Unknown deformable policy: {deformable_policy}")
    app_root = Path(__file__).resolve().parents[2]
    wrapper = DeterministicNeuroCAPWrapper(adapter.model, load_calibration(app_root), mode=spec.mode).to(adapter.device).eval()
    wav2spec = WaveToSpec().to(adapter.device).eval()
    info = BackendInfo(
        backend="pytorch-deploy",
        model_key=model_key,
        model_path=str(spec.checkpoint),
        device=str(adapter.device),
        input_mode="spec",
        available=True,
        message=f"deformable_policy={deformable_policy}",
    )
    return wrapper, wav2spec, info


class EdgeInferenceBackend:
    def __init__(
        self,
        model_key: str = "neurocap_full",
        backend: str = "pytorch",
        model_path: str | Path | None = None,
        device: str | None = None,
        deformable_policy: str = "native",
    ):
        self.model_key = model_key
        self.backend = backend
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.wav2spec = WaveToSpec().to(self.device).eval()
        self.model_path = Path(model_path) if model_path else None
        self.deformable_policy = deformable_policy
        self.info = BackendInfo(backend, model_key, str(self.model_path or ""), str(self.device), "spec", False)
        if backend == "pytorch":
            self.model, self.wav2spec, self.info = build_deploy_module(model_key, str(self.device), deformable_policy=deformable_policy)
        elif backend.startswith("onnxruntime"):
            self._load_onnx()
        elif backend == "tensorrt":
            raise RuntimeError("TensorRT backend requires a device-specific engine runner; use scripts/build_tensorrt.py to prepare an engine first.")
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _load_onnx(self) -> None:
        _prepare_cuda_dll_path()
        import onnxruntime as ort

        if self.model_path is None:
            self.model_path = Path(__file__).resolve().parents[2] / "models" / "edge" / f"{self.model_key}.onnx"
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = int(os.environ.get("SOUNDDET_ORT_INTRA_THREADS", "4"))
        sess_opts.inter_op_num_threads = int(os.environ.get("SOUNDDET_ORT_INTER_THREADS", "1"))
        sess_opts.enable_mem_pattern = True
        sess_opts.enable_cpu_mem_arena = True
        if os.environ.get("SOUNDDET_ORT_PROFILE", "0").lower() in {"1", "true", "yes"}:
            sess_opts.enable_profiling = True

        available = set(ort.get_available_providers())
        providers: list[str | tuple[str, dict[str, str]]] = ["CPUExecutionProvider"]
        if self.backend == "onnxruntime-cuda":
            providers = _select_existing_providers(
                available,
                [
                    (
                        "CUDAExecutionProvider",
                        {
                            "cudnn_conv_algo_search": os.environ.get("SOUNDDET_CUDNN_CONV_ALGO_SEARCH", "HEURISTIC"),
                            "arena_extend_strategy": "kNextPowerOfTwo",
                        },
                    ),
                    "CPUExecutionProvider",
                ],
            )
        elif self.backend == "onnxruntime-tensorrt":
            providers = _select_existing_providers(
                available,
                [
                    (
                        "TensorrtExecutionProvider",
                        {
                            "trt_fp16_enable": os.environ.get("SOUNDDET_TRT_FP16", "1"),
                            "trt_engine_cache_enable": "1",
                            "trt_engine_cache_path": str(Path(__file__).resolve().parents[2] / "models" / "edge" / "trt_cache"),
                        },
                    ),
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
        self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        first_dim = self.session.get_inputs()[0].shape[0]
        self.fixed_batch = int(first_dim) if isinstance(first_dim, int) else None
        actual = ",".join(self.session.get_providers())
        self.info = BackendInfo(self.backend, self.model_key, str(self.model_path), str(self.device), "spec", True, message=f"providers={actual}; graph_optimization=ORT_ENABLE_ALL")

    @torch.no_grad()
    def predict_batch(self, windows: list[np.ndarray]) -> list[Prediction]:
        if not windows:
            return []
        wav = torch.from_numpy(np.stack(windows)).float().unsqueeze(1)
        wav = trim_or_pad(wav, TARGET_SAMPLES).to(self.device)
        spec = self.wav2spec(wav)
        t0 = time.perf_counter()
        if self.backend == "pytorch":
            logits = self.model(spec)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        else:
            spec_np = spec.detach().cpu().numpy()
            if self.fixed_batch:
                chunks = []
                for i in range(0, spec_np.shape[0], self.fixed_batch):
                    chunk = spec_np[i : i + self.fixed_batch]
                    if chunk.shape[0] < self.fixed_batch:
                        pad = np.repeat(chunk[-1:], self.fixed_batch - chunk.shape[0], axis=0)
                        chunk = np.concatenate([chunk, pad], axis=0)
                        out = self.session.run([self.output_name], {self.input_name: chunk})[0][: spec_np.shape[0] - i]
                    else:
                        out = self.session.run([self.output_name], {self.input_name: chunk})[0]
                    chunks.append(out)
                logits = np.concatenate(chunks, axis=0)
            else:
                logits = self.session.run([self.output_name], {self.input_name: spec_np})[0]
            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        latency = (time.perf_counter() - t0) * 1000.0 / max(1, len(windows))
        out: list[Prediction] = []
        for row in probs:
            pred = int(row.argmax())
            out.append(Prediction(pred=pred, confidence=float(row[pred]), probs=row.astype(float).tolist(), latency_ms=latency, model_mode=self.backend))
        return out

    def predict_one(self, window: np.ndarray) -> Prediction:
        return self.predict_batch([window])[0]


def backend_summary(backend: EdgeInferenceBackend) -> dict:
    return {
        "backend": backend.info.backend,
        "model_key": backend.info.model_key,
        "model_path": backend.info.model_path,
        "device": backend.info.device,
        "input_mode": backend.info.input_mode,
        "classes": CLASS_NAMES,
    }


def _prepare_cuda_dll_path() -> None:
    try:
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.exists():
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(torch_lib))
    except Exception:
        pass


def _select_existing_providers(available: set[str], requested: list[str | tuple[str, dict[str, str]]]) -> list[str | tuple[str, dict[str, str]]]:
    selected: list[str | tuple[str, dict[str, str]]] = []
    for item in requested:
        name = item[0] if isinstance(item, tuple) else item
        if name in available:
            selected.append(item)
    if not selected:
        selected.append("CPUExecutionProvider")
    return selected
