# Embedding API

FastAPI service for remote-sensing embeddings, downstream monitoring tasks,
custom model training, batch inference, and SAM3 interactive segmentation.

The API currently serves:

| Region | ID | Patches | Notes |
|--------|----|---------|-------|
| Harbin New Area | `harbin` | 424 | Embeddings, monitoring tasks, SAM3, system models |
| Haidian District | `haidian` | 320 | Latest ModelScope assets, building/road/water heads, SAM3 |

---

## Live Service

| Entry | URL |
|-------|-----|
| Base URL | `http://60.31.21.42:22065` |
| Swagger UI | `http://60.31.21.42:22065/docs` |
| ReDoc | `http://60.31.21.42:22065/redoc` |
| Logs | `http://60.31.21.42:22065/logs` |
| Request audit | `http://60.31.21.42:22065/logs/request-audit` |

`22065` is the public mapped port for the deployed service. The deployment
machine itself may not be able to access that public address because of network
loopback rules; external users should use the public URL above.

---

## What This Service Provides

- **Embeddings**: PNG/NPY/NPZ/JSON access by region, patch, version, and month.
- **Task results**: building, road, water, construction, land-use/cover, and
  change-detection outputs.
- **System models**: official task heads for supported regions.
- **Custom training**: frontend-submitted GeoJSON annotations train user task heads.
- **Batch inference**: custom and system model inference across multiple patches.
- **SAM3 segmentation**: WGS84 point prompts in, WGS84 GeoJSON polygons out.
- **Operational logs**: browser-readable backend logs and request audit traces.

---

## Quick Start

```bash
pip install -r requirements.txt
python service_watchdog.py
```

Local endpoints:

```text
API:      http://localhost:9061
Swagger: http://localhost:9061/docs
Logs:    http://localhost:9061/logs
```

Development server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 9061
```

Run tests:

```bash
python -m pytest tests/ -q
```

---

## Common API Examples

```bash
export BASE="http://60.31.21.42:22065"

curl -s "$BASE/health"
curl -s "$BASE/regions"
curl -s "$BASE/regions/haidian/patches/patch_000000/embedding?format=json&version=v1&month=202512"
curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/road_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_road.png
```

For frontend integration, use Swagger first:

```text
http://60.31.21.42:22065/docs
```

---

## Documentation

| Document | Use it for |
|----------|------------|
| [`docs/API.md`](docs/API.md) | Full endpoint reference and curl checklist |
| [`docs/api-quickstart.md`](docs/api-quickstart.md) | Short frontend-oriented API examples |
| [`docs/operations.md`](docs/operations.md) | Deployment, ports, logs, watchdog, request audit |
| [`docs/haidian-assets.md`](docs/haidian-assets.md) | Latest Haidian ModelScope assets and replacement flow |
| [`docs/custom-training-workflow.md`](docs/custom-training-workflow.md) | Custom training and batch inference workflow |
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | Production update notes and verification plan |
| [`AGENTS.md`](AGENTS.md) | Project instructions for AI coding agents |

---

## Latest Haidian Assets

Haidian `v1` uses the latest ModelScope dataset:

```text
https://modelscope.cn/datasets/WeijieWu/xuannv_haidian_embdding
```

Current source prefix:

```text
artifacts/haidian-embedding-v1
```

Recommended integration values:

```text
region_id = haidian
patch_id  = patch_000000
version   = v1
month     = 202512
```

See [`docs/haidian-assets.md`](docs/haidian-assets.md) for download commands
and installed paths.

---

## Project Layout

```text
app/          FastAPI app, routers, schemas, services
data/         Region metadata, embeddings, task outputs
docs/         Human-facing docs
models/       System checkpoints and task heads
pipelines/    ModelScope download and preparation scripts
scripts/      Audit, visualization, static log server scripts
tests/        pytest suite
```

---

## Operational Notes

- Public production URL is HTTP. If the frontend is HTTPS, proxy this service
  through the same HTTPS origin to avoid browser mixed-content blocks.
- Request audit logs redact common secret fields and skip non-business noise
  such as `/logs`, `/docs`, `/openapi.json`, `/favicon.ico`, and `/health`.
- Set `CORS_ORIGINS` for browser access in production.
- Set `DOCS_URL=none` and `REDOC_URL=none` if public docs should be disabled.

---

## README Style

The root README is intentionally short: it is an entry point, not the full API
manual. Detailed how-to and reference material lives in [`docs/`](docs/).
