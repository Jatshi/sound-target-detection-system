# Edge Deployment Guide

## Export Policy

The edge graph uses log-mel spectrogram input with fixed shape `[1, 1, 64, 87]`. Audio preprocessing remains in Python to avoid deployment drift in `torchaudio` spectrogram conversion.

The main NeuroCAP model contains `torchvision::deform_conv2d`, which is not a standard ONNX operator. The export command therefore uses:

```powershell
python scripts\export_onnx.py --model neurocap_full --mode spec --deformable-policy static-conv
```

This replaces deformable convolution with its learned static kernel for the deployable graph. Treat this as an edge variant and always run validation before deployment.

## Validation

```powershell
python scripts\validate_export.py --backend onnxruntime-cpu --onnx models\edge\neurocap_full.onnx --n 64 --deformable-policy static-conv
python scripts\run_backend_shadow_eval.py --backend onnxruntime-cpu --model-path models\edge\neurocap_full.onnx --deformable-policy static-conv
```

Acceptance defaults:

- top-1 agreement with PyTorch deploy wrapper: `>= 0.995`
- max probability error: `<= 1e-3`
- shadow prediction agreement: `>= 0.995`

## Benchmark

```powershell
python scripts\benchmark_backend.py --backend onnxruntime-cpu --model-path models\edge\neurocap_full.onnx --batch-sizes 1,4 --deformable-policy static-conv
```

Use batch=1 for online deployment decisions.

## TensorRT

TensorRT engine generation is device-specific. Generate a command file with:

```powershell
python scripts\build_tensorrt.py --fp16
```

Then run the emitted `trtexec` command on the target device.

## Edge Runtime Package

```powershell
python scripts\package_edge_runtime.py
```

The package is written to `edge_runtime/` and contains service code, config, web console, model artifacts, calibration and startup scripts.
