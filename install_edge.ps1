$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  throw "python is not available on PATH. Install Python or use the env-specific launcher."
}

python -c "import torch, onnxruntime; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('onnxruntime', onnxruntime.__version__)"
python scripts\startup_self_check.py
python scripts\verify_manifest.py

Write-Host "Edge runtime self-check complete."

