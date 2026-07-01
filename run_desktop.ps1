$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "Starting Sound Target Detection desktop console..."
python scripts\run_desktop.py
