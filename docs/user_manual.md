# User Manual

## 1. Desktop Workflow

1. Start the desktop client with `run_desktop.ps1`.
2. Select an available model. Unavailable models remain visible but cannot be started.
3. Choose `Stream` or `Aligned` trial mode.
4. Adjust target thresholds, EMA, confirmation frames and merge gap if needed.
5. Click `Start` to run a online replay session.
6. Watch the `Runtime observability` panel for CPU, memory, GPU, disk, window count, event count, latency P95/P99 and queue/dropped-window status.
7. Inspect the event timeline and event table.
8. Select an event and mark it as `TP`, `FP` or `FN`; notes are written to the audit log.
9. Click `Export latest report` to create CSV, Markdown, PNG, PDF and ZIP artifacts.

## 2. Classes

The detector uses five labels:

- Gunshot
- Glass
- Babycry
- NonTarget
- Background

Only Gunshot, Glass and Babycry trigger alerts by default.

## 3. Runtime Health

Each session writes `input_health.json`, including silent-window count, clipped-window count, NaN count, RMS, peak level and sample-rate status. Review this file first if predictions look abnormal.

The desktop GUI also refreshes runtime observability every two seconds. The same runtime layer is exposed through the Web console at `/console` and the Prometheus-style endpoint at `/metrics`, so an operator can monitor the system either locally or through the service API.

## 4. Event Review

Manual review is stored in the SQLite `events` table and each operation is mirrored to `audit_logs`. This keeps a traceable record for later threshold tuning and hard-example export.


