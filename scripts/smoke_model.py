from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from sounddet.config import load_config
from sounddet.model_adapter import ModelAdapter
from sounddet.model_registry import registry
from sounddet.online_trial import OODCatalog, resolve_audio_path
from sounddet.features import load_audio_mono


def main() -> int:
    cfg = load_config()
    spec = registry()[cfg.default_model]
    if not spec.available:
        print(f"missing checkpoint: {spec.checkpoint}")
        return 2
    adapter = ModelAdapter(spec.checkpoint, mode=spec.mode, batch_size=8)
    catalog = OODCatalog(cfg.dataset_root)
    windows = []
    labels = []
    for ds in catalog.datasets():
        df = catalog.tables[ds].head(20).copy()
        for _, row in df.iterrows():
            p = resolve_audio_path(row, catalog.base_dirs[ds])
            if p and p.exists():
                wav = load_audio_mono(str(p), sample_rate=cfg.sample_rate, samples=cfg.window_samples)
                windows.append(wav.squeeze(0).numpy())
                labels.append(row.get("category", "unknown"))
            if len(windows) >= 8:
                break
        if len(windows) >= 8:
            break
    preds = adapter.predict_batch(windows)
    probs = np.asarray([p.probs for p in preds])
    assert probs.shape == (len(windows), 5)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    out = pd.DataFrame({"category": labels, "pred": [p.pred for p in preds], "confidence": [p.confidence for p in preds]})
    print(out.to_string(index=False))
    print("status: model smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

