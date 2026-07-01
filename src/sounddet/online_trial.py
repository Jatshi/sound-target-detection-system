from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from . import CLASS_NAMES
from .config import AppConfig
from .features import load_audio_mono, mix_at_snr, peak_normalize, trim_or_pad


DATASET_META = {
    "OOD-A": ("OOD_SET_A_v3_dynamic", "OOD_SET_A_v3_dynamic_metadata_V5_FINAL_CLEANED_BALANCED_D_SCALE.csv"),
    "OOD-B": ("OOD_SET_B_v2", "OOD_SET_B_v2_metadata_cleaned_V5_FINAL_CLEANED_BALANCED.csv"),
    "OOD-C": ("OOD_SET_C_v3_dynamic", "OOD_SET_C_v3_dynamic_metadata_V5_FINAL_CLEANED_BALANCED.csv"),
    "OOD-D": ("OOD_SET_D_v2", "OOD_SET_D_v2_metadata_cleaned_V5_FINAL_CLEANED_BALANCED.csv"),
}


@dataclass
class GroundTruthEvent:
    stream_id: str
    dataset: str
    start: float
    end: float
    label: int
    category: str
    source_path: str
    snr_db: float | None


@dataclass
class StreamSpec:
    stream_id: str
    dataset: str
    wav_path: str
    duration_sec: float


def label_from_category(category: str) -> int:
    cat = str(category).lower()
    if "background" in cat:
        return 4
    if "gunshot" in cat:
        return 0
    if "glass" in cat:
        return 1
    if "cry" in cat:
        return 2
    return 3


def resolve_audio_path(row: pd.Series, base_dir: Path, prefer_event: bool = False) -> Path | None:
    candidates = []
    if prefer_event and "original_event_path" in row and pd.notna(row["original_event_path"]):
        candidates.append(str(row["original_event_path"]))
    if "file_path" in row and pd.notna(row["file_path"]):
        candidates.append(str(row["file_path"]))
    if "file_name" in row and pd.notna(row["file_name"]):
        candidates.append(str(row["file_name"]))
    for cand in candidates:
        p = Path(cand)
        if not p.is_absolute():
            p = base_dir / p.name
        if p.exists():
            return p
    return None


class OODCatalog:
    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)
        self.tables: dict[str, pd.DataFrame] = {}
        self.base_dirs: dict[str, Path] = {}
        for name, (folder, meta) in DATASET_META.items():
            path = self.dataset_root / folder / meta
            if path.exists():
                self.tables[name] = pd.read_csv(path)
                self.base_dirs[name] = path.parent

    def datasets(self) -> list[str]:
        return list(self.tables.keys())

    def sample_rows(self, dataset: str, rng: random.Random, label_ids: set[int], n: int) -> list[pd.Series]:
        df = self.tables[dataset].copy()
        df["_label"] = df["category"].map(label_from_category)
        sub = df[df["_label"].isin(label_ids)]
        if sub.empty:
            sub = df
        idxs = [rng.randrange(len(sub)) for _ in range(n)]
        return [sub.iloc[i] for i in idxs]

    def sample_balanced_event_rows(self, dataset: str, rng: random.Random, n: int) -> list[pd.Series]:
        labels = [0, 1, 2, 3]
        rows: list[pd.Series] = []
        for i in range(n):
            rows.extend(self.sample_rows(dataset, rng, {labels[i % len(labels)]}, 1))
        rng.shuffle(rows)
        return rows[:n]


def make_background(catalog: OODCatalog, dataset: str, rng: random.Random, duration_samples: int, sample_rate: int) -> torch.Tensor:
    chunks = []
    rows = catalog.sample_rows(dataset, rng, {4}, max(1, int(duration_samples / sample_rate) + 2))
    for row in rows:
        p = resolve_audio_path(row, catalog.base_dirs[dataset])
        if p is None:
            continue
        wav = load_audio_mono(str(p), sample_rate=sample_rate)
        chunks.append(wav[0])
        if sum(c.numel() for c in chunks) >= duration_samples:
            break
    if not chunks:
        return torch.zeros(1, duration_samples)
    bg = torch.cat(chunks)[:duration_samples]
    if bg.numel() < duration_samples:
        bg = torch.nn.functional.pad(bg, (0, duration_samples - bg.numel()))
    return peak_normalize(bg.unsqueeze(0), peak=0.65)


def build_streams(
    cfg: AppConfig,
    out_dir: str | Path,
    minutes: float | None = None,
    streams_per_dataset: int | None = None,
    seed: int | None = None,
    write_audio: bool = True,
    trial_mode: str | None = None,
) -> tuple[list[StreamSpec], list[GroundTruthEvent]]:
    out_dir = Path(out_dir)
    stream_dir = out_dir / "streams"
    stream_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed if seed is None else seed)
    catalog = OODCatalog(cfg.dataset_root)
    trial_mode = trial_mode or cfg.trial_mode
    duration_sec = float(minutes if minutes is not None else cfg.stream_minutes) * 60.0
    duration_samples = int(round(duration_sec * cfg.sample_rate))
    n_streams = int(streams_per_dataset if streams_per_dataset is not None else cfg.streams_per_dataset)
    streams: list[StreamSpec] = []
    events: list[GroundTruthEvent] = []

    if trial_mode == "aligned":
        for dataset in catalog.datasets():
            for stream_idx in range(n_streams):
                stream_id = f"{dataset}_{stream_idx:02d}"
                slot_count = max(1, int(duration_sec // cfg.window_sec))
                rows = catalog.sample_balanced_event_rows(dataset, rng, max(1, slot_count // 2))
                bg_rows = catalog.sample_rows(dataset, rng, {4}, slot_count)
                sequence: list[pd.Series] = []
                event_iter = iter(rows)
                bg_iter = iter(bg_rows)
                for i in range(slot_count):
                    if i % 3 == 1:
                        try:
                            sequence.append(next(event_iter))
                            continue
                        except StopIteration:
                            pass
                    sequence.append(next(bg_iter))
                chunks = []
                for i, row in enumerate(sequence):
                    label = label_from_category(str(row.get("category", "")))
                    p = resolve_audio_path(row, catalog.base_dirs[dataset], prefer_event=False)
                    if p is None:
                        chunk = torch.zeros(1, cfg.window_samples)
                    else:
                        chunk = trim_or_pad(load_audio_mono(str(p), sample_rate=cfg.sample_rate), cfg.window_samples)
                    chunks.append(chunk[0])
                    if label != 4:
                        start_sec = i * cfg.window_sec
                        snr = None
                        if "snr_db" in row and pd.notna(row["snr_db"]):
                            try:
                                snr = float(row["snr_db"])
                            except Exception:
                                snr = None
                        events.append(
                            GroundTruthEvent(stream_id, dataset, start_sec, start_sec + cfg.window_sec, label, str(row.get("category", "")), str(p), snr)
                        )
                wav = torch.cat(chunks).unsqueeze(0)
                wav_path = stream_dir / f"{stream_id}.wav"
                if write_audio:
                    sf.write(str(wav_path), wav.squeeze(0).numpy(), cfg.sample_rate)
                streams.append(StreamSpec(stream_id, dataset, str(wav_path), wav.shape[-1] / cfg.sample_rate))
        pd.DataFrame([asdict(x) for x in streams]).to_csv(out_dir / "streams.csv", index=False)
        pd.DataFrame([asdict(x) for x in events]).to_csv(out_dir / "events_gt.csv", index=False)
        return streams, events

    for dataset in catalog.datasets():
        for stream_idx in range(n_streams):
            stream_id = f"{dataset}_{stream_idx:02d}"
            wav = make_background(catalog, dataset, rng, duration_samples, cfg.sample_rate)
            event_count = max(3, int(duration_sec / 12.0))
            starts = sorted(rng.uniform(2.0, max(2.1, duration_sec - 2.0)) for _ in range(event_count))
            rows = catalog.sample_balanced_event_rows(dataset, rng, event_count)
            last_end = -10.0
            for start_sec, row in zip(starts, rows):
                if start_sec - last_end < 1.2:
                    continue
                label = label_from_category(str(row.get("category", "")))
                p = resolve_audio_path(row, catalog.base_dirs[dataset], prefer_event=False)
                if p is None:
                    continue
                event_wav = load_audio_mono(str(p), sample_rate=cfg.sample_rate)
                event_wav = trim_or_pad(event_wav, int(cfg.sample_rate * 1.0))
                snr = None
                if "snr_db" in row and pd.notna(row["snr_db"]):
                    try:
                        snr = float(row["snr_db"])
                    except Exception:
                        snr = None
                start_sample = int(round(start_sec * cfg.sample_rate))
                # The metadata file_path is already the final one-second OOD
                # evaluation window. Insert it directly to preserve the model's
                # training/evaluation input distribution.
                wav[:, start_sample : start_sample + event_wav.shape[1]] = 0.0
                wav = mix_at_snr(wav, event_wav, start_sample, snr_db=None)
                end_sec = min(duration_sec, start_sec + event_wav.shape[1] / cfg.sample_rate)
                events.append(
                    GroundTruthEvent(stream_id, dataset, start_sec, end_sec, label, str(row.get("category", "")), str(p), snr)
                )
                last_end = end_sec
            wav_path = stream_dir / f"{stream_id}.wav"
            if write_audio:
                sf.write(str(wav_path), wav.squeeze(0).numpy(), cfg.sample_rate)
            streams.append(StreamSpec(stream_id, dataset, str(wav_path), duration_sec))

    pd.DataFrame([asdict(x) for x in streams]).to_csv(out_dir / "streams.csv", index=False)
    pd.DataFrame([asdict(x) for x in events]).to_csv(out_dir / "events_gt.csv", index=False)
    return streams, events
