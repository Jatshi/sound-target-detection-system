# API Reference

The service is implemented with FastAPI and exposes OpenAPI docs at `/docs`.

## Endpoints

- `GET /health`: runtime status, active session, database path and model status.
- `GET /metrics`: Prometheus-style text metrics.
- `GET /models`: model registry with checkpoint availability and SHA256.
- `POST /models/select`: select default model. Body: `{"model_key": "neurocap_full"}`.
- `GET /config`: current runtime config.
- `POST /config`: update selected config fields.
- `POST /config/rollback`: restore the previous in-memory config snapshot.
- `POST /sessions/start`: start a online replay session.
- `POST /sessions/stop`: request stop.
- `GET /sessions/{session_id}`: session row.
- `GET /events`: paginated event query, with optional `session_id`, `label`, and `review_status` filters.
- `POST /events/{event_id}/review`: mark event as reviewed.
- `GET /reports/{session_id}`: export session report package.
- `WS /stream/events`: WebSocket event and session messages.

If `api_token` is configured, send it as `x-api-token`.

