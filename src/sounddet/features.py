from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from .neurocap_model import TARGET_SAMPLES, TARGET_SR, WaveToSpec


def load_audio_mono(path: str, sample_rate: int = TARGET_SR, samples: int | None = None) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = T.Resample(sr, sample_rate)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if samples is not None:
        wav = trim_or_pad(wav, samples)
    return wav.float()


def trim_or_pad(wav: torch.Tensor, samples: int = TARGET_SAMPLES) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    length = wav.shape[-1]
    if length > samples:
        return wav[..., :samples]
    if length < samples:
        return F.pad(wav, (0, samples - length))
    return wav


def peak_normalize(wav: torch.Tensor, peak: float = 0.95) -> torch.Tensor:
    mx = wav.abs().max().item()
    if mx > peak and mx > 0:
        return wav / mx * peak
    return wav


def rms(x: torch.Tensor) -> torch.Tensor:
    return (x.pow(2).mean() + 1e-9).sqrt()


def mix_at_snr(background: torch.Tensor, event: torch.Tensor, start: int, snr_db: float | None = None) -> torch.Tensor:
    out = background.clone()
    if event.dim() == 2:
        event = event[0]
    if out.dim() == 2:
        out1 = out[0]
    else:
        out1 = out
    end = min(out1.numel(), start + event.numel())
    if end <= start:
        return out
    event = event[: end - start].clone()
    if snr_db is not None:
        bg_seg = out1[start:end]
        target_rms = rms(bg_seg) * (10 ** (float(snr_db) / 20.0))
        event = event * (target_rms / (rms(event) + 1e-8))
    out1[start:end] = out1[start:end] + event
    return peak_normalize(out1.unsqueeze(0))


def make_spec_transform() -> WaveToSpec:
    return WaveToSpec()
