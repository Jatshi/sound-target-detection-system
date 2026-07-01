$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "Starting Sound Target Detection service..."
Write-Host "OpenAPI: http://127.0.0.1:8765/docs"
Write-Host "Web console: http://127.0.0.1:8765/console"
python scripts\run_service.py
