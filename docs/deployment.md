# Deployment Guide

## Windows Desktop

```powershell
cd "<sound_target_detection_app>"
.\run_desktop.ps1
```

## Windows Service Mode

```powershell
.\run_service.ps1
```

Open `http://127.0.0.1:8765/docs` for interactive API testing.

## Linux systemd

Use `deploy/sounddet.service` as a template. Update `WorkingDirectory`, `Environment`, and `ExecStart` to match the deployment host.

## Docker

```bash
docker build -t sounddet:local .
docker run --rm -p 8765:8765 -v /data/sounddet:/app/data sounddet:local
```

GPU inference inside Docker requires host-specific CUDA runtime setup. CPU service mode remains useful for API and report testing.

## Configuration

Configuration files live in `configs/`. Environment variables can override any config field using `SOUNDDET_` plus the uppercase field name, for example:

```powershell
$env:SOUNDDET_SERVICE_PORT="9000"
$env:SOUNDDET_DB_PATH=".\data\sounddet.db"
```
