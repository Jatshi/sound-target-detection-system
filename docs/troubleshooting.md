# Troubleshooting

## Model unavailable

Run:

```powershell
python scripts\smoke_model.py
```

Check that the checkpoint path in `src/sounddet/model_registry.py` exists.

## Predictions look different from offline paper results

Run:

```powershell
python scripts\diagnose_offline_consistency.py --dataset OOD-B --model neurocap_sound_only --n 128
```

Expected agreement is `1.0` for the fixed consistency subset.

## No microphone devices

Install `sounddevice` and confirm that Windows privacy settings allow microphone access. The application still supports WAV, folder and online replay replay without a microphone.

## Too many false alarms

Increase target thresholds, increase confirmation frames, or enlarge merge gap. Use `scan_online_thresholds.py` to tune from saved `window_predictions.csv` without rerunning model inference.

## Database locked

SQLite uses WAL mode, but long report exports can briefly hold read handles. Stop active sessions before moving or deleting `data/sounddet.db`.


