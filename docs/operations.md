# Operations

This document covers runtime ports, service startup, logs, request audit, and
deployment notes.

## Public and Local URLs

| Entry | URL |
|-------|-----|
| Public API | `http://60.31.21.42:22065` |
| Public Swagger | `http://60.31.21.42:22065/docs` |
| Public logs | `http://60.31.21.42:22065/logs` |
| Public request audit | `http://60.31.21.42:22065/logs/request-audit` |
| Local API | `http://localhost:9061` |
| Optional local static logs | `http://localhost:9091/logs` |

`22065` is the external mapped port. The deployment host may not be able to
access the public address through loopback, but external clients can use it.

## Start the API

Recommended:

```bash
python service_watchdog.py
```

Stop:

```bash
python service_watchdog.py stop
```

Development:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 9061
```

## Optional Static Log Server

The API itself serves `/logs`, so this is only needed when you want a separate
static port:

```bash
python scripts/static_logs_server.py --host 0.0.0.0 --port 9091 --root /workspace/embedding-api
```

## Log Files

| File | Purpose |
|------|---------|
| `logs/uvicorn.log` | FastAPI/uvicorn access logs, exceptions, model inference logs |
| `logs/watchdog.log` | Watchdog lifecycle, health checks, restart attempts |
| `logs/watchdog.console.log` | Watchdog stdout/stderr |
| `logs/request_audit.jsonl` | Structured request audit records |
| `logs/static_tmp_9091.log` | Optional static log server access log |

## Request Audit

Open:

```text
http://60.31.21.42:22065/logs/request-audit
```

The audit page shows:

- method and path
- query parameters
- JSON body snapshot
- response status code
- elapsed time
- client and user-agent

The audit logger redacts common secrets:

- `Authorization`
- `X-API-Key`
- `token`
- `password`
- `secret`

It skips non-business noise:

- `/logs/*`
- `/docs`
- `/redoc`
- `/openapi.json`
- `/favicon.ico`
- `/health`

Rotation defaults:

| Variable | Default |
|----------|---------|
| `REQUEST_AUDIT_LOG` | `logs/request_audit.jsonl` |
| `REQUEST_AUDIT_ROTATE_BYTES` | `20971520` |
| `REQUEST_AUDIT_BACKUP_COUNT` | `5` |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CONFIG_PATH` | `./config.yaml` | Runtime config path |
| `CORS_ORIGINS` | empty | Comma-separated browser origins |
| `DOCS_URL` | `/docs` | Swagger path; set `none` to disable |
| `REDOC_URL` | `/redoc` | ReDoc path; set `none` to disable |
| `WATCHDOG_CHECK_INTERVAL` | `60` | Watchdog health-check interval |
| `WATCHDOG_HEALTH_TIMEOUT` | `20` | Watchdog health-check timeout |

## Docker

```bash
docker-compose up -d
```

The Docker service exposes `8000` by default. Bare-metal deployment currently
uses `9061`, with public `22065` mapped by the host/reverse proxy.

## Verification

```bash
curl -s http://localhost:9061/health
curl -s http://localhost:9061/logs/request-audit >/tmp/request-audit.html
python -m pytest tests/ -q
```

For a broader live API audit:

```bash
python scripts/full_api_audit.py
```
