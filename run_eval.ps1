$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "Running quick online replay evaluation..."
python scripts\run_online_eval.py --quick
