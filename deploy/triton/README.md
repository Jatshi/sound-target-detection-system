# Triton Deployment Notes

This repository is prepared for the ONNX classifier graph. Audio-to-mel
preprocessing stays in the application runtime, and Triton receives one-second
log-mel tensors with shape `[B, 1, 64, 87]`.

Prepare the repository:

```powershell
python scripts\prepare_triton_repository.py
```

Run Triton on a GPU host:

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ${PWD}/deploy/triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.06-py3 \
  tritonserver --model-repository=/models
```

The current package includes dynamic batching configuration, warmup, and GPU
instance-group settings. Treat Triton throughput numbers as unmeasured until the
container is launched and benchmarked on the target edge machine.
