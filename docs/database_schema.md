# Database Schema

Default database: `data/sounddet.db`.

## sessions

One row per run. Stores `session_id`, start/end timestamps, model key, input source, config JSON, output directory and status.

## events

One row per merged event alert. Stores stream, dataset, start/end time, label, confidence, audio clip path, match/review status and note.

## window_predictions

One row per hop. Stores stream, dataset, window time, predicted class, confidence, probability vector, latency and optional label.

## audio_clips

One row per saved alert-centered WAV clip. Linked to `events.id` when available.

## model_registry

Stores public model key, label, checkpoint path, mode, availability and checkpoint hash.

## audit_logs

Records session start/stop, model/config changes, event review and report export operations.

