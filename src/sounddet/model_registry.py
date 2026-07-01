from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelSpec:
    key: str
    label: str
    checkpoint: Path
    mode: str

    @property
    def available(self) -> bool:
        return self.checkpoint.exists()

    def sha256(self) -> str | None:
        if not self.available:
            return None
        h = hashlib.sha256()
        with self.checkpoint.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()


def registry() -> dict[str, ModelSpec]:
    main = APP_ROOT / "models" / "neurocap_full" / "model.pt"
    resnet10_neuro = APP_ROOT / "models" / "neurocap_resnet10" / "model.pt"
    baseline = APP_ROOT / "models" / "baselines"
    return {
        "neurocap_full": ModelSpec("neurocap_full", "NeuroCAP full", main, "neurocap_full"),
        "neurocap_sound_only": ModelSpec("neurocap_sound_only", "NeuroCAP sound-only", main, "neurocap_sound_only"),
        "neurocap_resnet10": ModelSpec("neurocap_resnet10", "NeuroCAP ResNet10", resnet10_neuro, "neurocap_resnet10"),
        "baseline_deformable": ModelSpec("baseline_deformable", "Baseline deformable", baseline / "deformable.pt", "baseline_deformable"),
        "baseline_resnet10": ModelSpec("baseline_resnet10", "Baseline ResNet10", baseline / "resnet10.pt", "baseline_resnet10"),
        "baseline_dilated": ModelSpec("baseline_dilated", "Baseline dilated", baseline / "dilated.pt", "baseline_dilated"),
        "baseline_paulnet": ModelSpec("baseline_paulnet", "Baseline PaulNet", baseline / "paulnet.pt", "baseline_paulnet"),
        "baseline_gtcnn": ModelSpec("baseline_gtcnn", "Baseline GT-CNN", baseline / "gtcnn.pt", "baseline_gtcnn"),
        "baseline_dualbranch": ModelSpec("baseline_dualbranch", "Baseline DB-CNN", baseline / "dualbranch.pt", "baseline_dualbranch"),
    }


def default_model_key() -> str:
    return "neurocap_full"


def registry_status() -> list[dict[str, str | bool | None]]:
    out = []
    for spec in registry().values():
        out.append(
            {
                "key": spec.key,
                "label": spec.label,
                "checkpoint": str(spec.checkpoint),
                "mode": spec.mode,
                "available": spec.available,
                "sha256": spec.sha256(),
            }
        )
    return out
