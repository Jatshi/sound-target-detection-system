# Sound Target Detection System

An online sound target detection system for five-class acoustic event monitoring:
`Gunshot`, `Glass`, `Babycry`, `NonTarget`, and `Background`.

The repository contains the deployable source code, GUI, FastAPI service, Web console,
online replay evaluation tools, ONNX export utilities, and edge-deployment templates.
Large model checkpoints, ONNX engines, local datasets, runtime outputs, and personal
research documents are intentionally excluded from the GitHub package.

## Features

- PyQt5 desktop console for live microphone input, online replay, model selection,
  thresholds, event review, waveform/spectrogram display, and report export.
- FastAPI service with REST endpoints, WebSocket event stream, `/metrics`, and a
  browser-based operations console at `/console`.
- SQLite event store for sessions, events, window predictions, audio clips, model
  registry, and audit logs.
- Unified detection engine using 44.1 kHz mono audio, 1.0 s windows, and 0.5 s hop.
- Model registry for NeuroCAP, sound-only, ResNet10, Dilated, Deformable, PaulNet,
  GT-CNN, and DB-CNN style entries.
- Edge-inference utilities for ONNX export, ONNX Runtime validation, backend
  benchmarking, continuous batching, and Triton deployment templates.

## Repository Layout

```text
configs/       Runtime configuration templates.
src/sounddet/  Core package: engine, GUI, API service, models, metrics, storage.
scripts/       Training-free runtime, evaluation, export, benchmark, and utility commands.
web/           Static Web operations console.
deploy/        Docker, Triton, and Kubernetes deployment templates.
packaging/     Packaging helpers.
models/        Placeholder directories for user-provided model files.
reference/     Small calibration metadata only.
docs/          Public user, API, deployment, database, and troubleshooting docs.
```

## Model Files

This GitHub package does not include checkpoints or ONNX files. Place model files under:

```text
models/neurocap_full/model.pt
models/neurocap_resnet10/model.pt
models/baselines/resnet10.pt
models/baselines/dilated.pt
models/baselines/deformable.pt
models/baselines/paulnet.pt
models/baselines/gtcnn.pt
models/baselines/dualbranch.pt
models/edge/neurocap_resnet10_opt.onnx
```

The application reports missing models as unavailable instead of silently falling back.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m compileall src scripts
python scripts\run_service.py
```

Open:

- API docs: `http://127.0.0.1:8765/docs`
- Web console: `http://127.0.0.1:8765/console`

Desktop GUI:

```powershell
python scripts\run_desktop.py
```

Online replay smoke evaluation:

```powershell
python scripts\run_online_eval.py --quick
```

## Configuration

Edit `configs/default.yaml` or override paths with environment variables where supported.
For real evaluation, set `dataset_root` to a local OOD dataset directory. Runtime outputs
are written to `outputs/`, which is ignored by Git.

## Edge Runtime

Export and validate an ONNX model:

```powershell
python scripts\export_onnx.py --model neurocap_resnet10 --mode spec
python scripts\validate_export.py --backend onnxruntime-cpu --model neurocap_resnet10 --onnx models\edge\neurocap_resnet10_opt.onnx --n 64
python scripts\benchmark_backend.py --backend onnxruntime-cpu --model neurocap_resnet10 --model-path models\edge\neurocap_resnet10_opt.onnx --batch-sizes 1,4,16
```

## Git Policy

Do not commit local datasets, outputs, logs, checkpoints, ONNX files, TensorRT engines,
SQLite databases, screenshots, or packaged binaries. These are intentionally covered by
`.gitignore`.
