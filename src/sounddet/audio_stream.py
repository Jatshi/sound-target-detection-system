from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
try:
    import sounddevice as sd
except Exception:  # optional runtime dependency
    sd = None


class WavFileStream:
    def __init__(self, path: str | Path, hop_samples: int, dtype=np.float32):
        self.path = Path(path)
        self.hop_samples = hop_samples
        self.dtype = dtype

    def __iter__(self) -> Iterator[np.ndarray]:
        with sf.SoundFile(str(self.path)) as handle:
            while True:
                chunk = handle.read(self.hop_samples, dtype="float32", always_2d=False)
                if chunk is None or len(chunk) == 0:
                    break
                if getattr(chunk, "ndim", 1) > 1:
                    chunk = chunk.mean(axis=1)
                yield np.asarray(chunk, dtype=self.dtype)


class ArrayStream:
    def __init__(self, wav: np.ndarray, hop_samples: int):
        self.wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        self.hop_samples = hop_samples

    def __iter__(self):
        for pos in range(0, len(self.wav), self.hop_samples):
            yield self.wav[pos : pos + self.hop_samples]


def list_audio_devices() -> list[dict]:
    if sd is None:
        return []
    return [dict(d, index=i) for i, d in enumerate(sd.query_devices())]


class MicrophoneStream:
    def __init__(self, sample_rate: int, hop_samples: int, device: int | None = None, channels: int = 1):
        if sd is None:
            raise RuntimeError("sounddevice is not installed; microphone input is unavailable.")
        self.sample_rate = sample_rate
        self.hop_samples = hop_samples
        self.device = device
        self.channels = max(1, int(channels))

    def __iter__(self):
        with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, dtype="float32", device=self.device, blocksize=self.hop_samples) as stream:
            while True:
                data, overflow = stream.read(self.hop_samples)
                chunk = np.asarray(data, dtype=np.float32)
                if self.channels == 1:
                    chunk = chunk[:, 0]
                if overflow:
                    chunk = np.nan_to_num(chunk)
                yield chunk


class NetworkAudioStream:
    """Interface adapter for RTSP/RTP/WebRTC audio sources."""

    def __init__(self, uri: str, sample_rate: int, hop_samples: int):
        self.uri = uri
        self.sample_rate = sample_rate
        self.hop_samples = hop_samples

    def __iter__(self):
        raise NotImplementedError(
            "Network audio adapters are reserved for device-level deployment. "
            "Use WAV, folder, microphone, or online replay streams in this release."
        )
