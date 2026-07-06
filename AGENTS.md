# AGENTS.md - Embedding API

This file is for AI coding agents working in this repository. It should stay
grounded in the current code and API docs. When behavior differs between docs
and implementation, verify against `app/`, `config.yaml`, and tests before
editing.

## Project Overview

Embedding API is a Python FastAPI service for remote-sensing embeddings,
regional patch metadata, downstream monitoring results, custom model training,
system model inference, region mosaics, and SAM3 interactive segmentation.

The service currently targets two configured regions:

- `harbin` - Harbin New Area, 424 patches.
- `haidian` - Haidian District, 320 patches.

Core API areas:

- Health and region metadata: `/health`, `/regions`, `/regions/{region_id}`.
- Patch listing and detail with WGS84 bbox filtering.
- Embedding retrieval as `png`, `npy`, `npz`-backed arrays, `json`, or `cache`.
- Downstream task summaries, patch results, predictions, labels, and tile lists.
- Region mosaic generation: `/regions/{region_id}/mosaic`.
- Custom user models: `/models/*`.
- System pre-trained models: `/system-models/*`.
- SAM3 embed/segment/status endpoints under `/regions/{region_id}/sam3/*`.

Primary frontend/API docs:

- `README.md` - high-level feature and deployment overview.
- `docs/API.md` - frontend-facing API reference in Chinese.
- `docs/custom-training-workflow.md` - custom training workflow for frontend
  annotations.

## Technology Stack

- Runtime: Python 3.9+.
- Web framework: FastAPI with Pydantic v2.
- Server: uvicorn, optionally managed by `service_watchdog.py`.
- Config: YAML via `config.yaml`; hot-reloaded through `watchdog`.
- ML: PyTorch, torchvision, local vendored `sam3` package in `sam3_pkg/`.
- Geo/image: Pillow, rasterio, numpy.
- Containerization: Docker and docker-compose.

There is no root `pyproject.toml`. Install runtime dependencies from
`requirements.txt`. The `sam3_pkg/pyproject.toml` belongs only to the bundled
SAM3 package.

## Repository Map

```text
app/
  main.py                 FastAPI app, CORS, docs URLs, router inclusion
  config.py               Config manager, hot reload, patch metadata cache
  routers/
    regions.py            Region metadata and `/regions/{id}/mosaic`
    patches.py            Patch list/detail
    embeddings.py         Embedding retrieval and safety limits
    tasks.py              Downstream task summary/result/prediction/label/tile APIs
    models.py             Custom model CRUD, training, inference, result download
    system_models.py      System model listing/classes/inference/result download
    sam3.py               SAM3 embed/segment/status API
  schemas/                Pydantic request/response models
  services/
    data_service.py       Secure path resolution for embeddings/tasks
    mosaic_service.py     Region-wide S2/S1/Landsat mosaic building and cache
    tile_service.py       Patch tile listing; XYZ tile serving is stubbed
    sam3_service.py       Lazy SAM3 model loading, imagery loading, cache
    auth_service.py       Optional API-key auth
    user_paths.py         Per-user storage paths
    geojson_adapter.py    WGS84 GeoJSON to pixel-mask rasterization
    model_registry.py     Custom model/job metadata
    training_engine.py    Lightweight downstream head training
    inference_engine.py   Custom/system inference helpers
data/                     Regional metadata, embeddings, task outputs
docs/                     API and workflow docs
models/                   Checkpoints and generated model artifacts
pipelines/                ModelScope asset download/preparation scripts
sam3_pkg/                 Vendored SAM3 package
tests/                    Pytest suite
```

## Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Local development:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 9061
```

The current `app/main.py` enables Swagger/ReDoc by default at `/docs` and
`/redoc`. Set `DOCS_URL=none` and/or `REDOC_URL=none` to disable them. Some
older docs mention defaults of `none`; check `app/main.py` before changing this
behavior.

Watchdog:

```bash
python service_watchdog.py
python service_watchdog.py stop
```

Docker:

```bash
docker-compose up -d
```

Useful environment variables:

| Variable | Purpose |
| --- | --- |
| `CONFIG_PATH` | YAML config path, defaults to `./config.yaml` |
| `CORS_ORIGINS` | Comma-separated allowed browser origins; empty rejects CORS |
| `DOCS_URL` | Swagger path, `/docs` by default in code, `none` disables |
| `REDOC_URL` | ReDoc path, `/redoc` by default in code, `none` disables |

## Data and Asset Notes

`config.yaml` is the source of truth for regions, embedding versions, task
directories, SAM3 paths, and system model checkpoints. Changes are hot-reloaded.

Asset download scripts live under `pipelines/`:

- `pipelines/haidian/download_modelscope_assets.py` downloads full Haidian V1
  assets from ModelScope dataset `WeijieWu/xuannv_haidian_embdding`.
- `pipelines/haidian/download_embeddings.py` downloads only Haidian embeddings.
- `pipelines/harbin/download_modelscope_assets.py` downloads Harbin archives and
  can verify checksums.

The latest Haidian source is `artifacts/haidian-embedding-v1` in
`WeijieWu/xuannv_haidian_embdding`. It installs:

- embeddings to `data/haidian/embeddings/v1/{YYYYMM}/{patch_id}.npy|png|json`
- the embedding checkpoint to
  `models/haidian/v1/embedding/haidian_embedding_v1_p10c_epoch800.pt`
- MLP task heads to `models/haidian/v1/task_heads/*.pt`

Do not restore older Haidian checkpoints or old `api_ready` layouts unless the
user explicitly asks. Do not commit tokens. ModelScope access should use
`MODELSCOPE_TOKEN` from the environment when needed.

## API-Specific Behavior

### Regions and Tasks

The docs refer to a unified task vocabulary, but the current config also exposes
region-specific P2A tasks. Always inspect `config.yaml` and task router tests
when changing task behavior.

Common task IDs include:

- `change_detection`
- `building_extraction`
- `road_extraction`
- `construction`
- `construction_joint`
- `land_use_classification`
- `land_cover_classification`
- `water_extraction`

Important current caveats:

- Haidian V1 uses xuannv P2A embeddings.
- Haidian V1 has pre-generated results for P2A-style tasks such as
  `building_extraction`, `road_extraction`, `construction`, and water-related
  outputs depending on config/data availability.
- Some tasks may be listed but require system-model inference rather than
  pre-generated result files.

### Embeddings

Embedding paths are config-driven. Harbin and Haidian use different layouts and
month formats, so avoid hardcoding paths or date assumptions.

- Harbin examples usually use months like `2025-04`.
- Haidian examples may use compact P2A months/dates such as `202512` or
  `20260115`; `app/services/time_utils.py` and related tests cover accepted
  normalization.
- `format=cache` should prefer a displayable PNG when available, then fall back
  to binary array data.

### Mosaics

`GET /regions/{region_id}/mosaic` is implemented in `app/routers/regions.py`
and `app/services/mosaic_service.py`.

- Supports sensor types such as `s2`, `s1`, and `landsat`.
- Supports output formats such as `png` and `tif`.
- Can restrict generation with repeated `patch_ids` query parameters.
- Caches generated files under `users/default/mosaic` by default.

### Custom Models

Custom model APIs consume a frontend-submitted GeoJSON annotation package in
`POST /models`.

- Supported model types include `classification` and `change_detection`.
- GeoJSON geometries are WGS84 `Polygon` or `MultiPolygon`.
- Features carry `patch_id`, `region_id`, `class_id`, `task_type`, and either
  `month` or `before_month`/`after_month`.
- The backend rasterizes polygons into 128x128 masks and trains a lightweight
  downstream head. Empty bodies and demo fallbacks are not valid behavior.
- `class_ids` may select a subset of classes; annotation packages may still
  include unselected classes, which are ignored during training.
- When auth is configured, `/models/*` is per-user and requires API key headers.

### System Models

System models live under `/system-models`.

- `GET /system-models?region_id=...`
- `GET /system-models/{task_id}/classes?region_id=...&version=...`
- `POST /system-models/{task_id}/infer?...`
- `GET /system-models/results/{filename}`

Generated system model result URLs should use `/system-models/results/...`.

### SAM3

SAM3 is lazy-loaded in `app/services/sam3_service.py`.

- Model path and tokenizer path come from `config.yaml` under `sam3`.
- `/sam3/segment` accepts WGS84 point coordinates, required `date`, and
  `sensor_type` (`s2`, `s1`, `landsat`). It auto-selects the patch, computes or
  reuses a cached embedding, and returns WGS84 GeoJSON polygon boxes.
  `point_labels` is optional; when omitted, every prompt point is treated as a
  positive foreground point (`1`). Do not reintroduce a `month` field for this
  endpoint.
- `/sam3/embed` remains available for preloading by explicit `patch_id`, `month`,
  and optional `sensor_type`; cached IDs include sensor type.
- Raw imagery is resolved through the same path helpers as mosaics, including
  each region's `s2_dir` and `/workspace/raw`.
- Embeddings are cached with an LRU cache controlled by `max_cache_size`.
- GPU/model inference is serialized with an `asyncio.Lock`.
- Integration tests are marked slow and should tolerate missing GPU/data/model
  by returning availability-style responses rather than hard failing.

### Tiles

Patch-based tile listing is implemented. Standard XYZ tile image serving
`/tiles/{z}/{x}/{y}.png` is intentionally not implemented and should return
`501` unless the project explicitly adds real XYZ tile support.

## Security Rules

This service serves local files based on user-controlled path fragments. Preserve
the existing defenses when changing file-serving code.

- Validate patch IDs strictly as `patch_\d{6}`.
- Validate `month`, `period`, dates, and task/version fragments before using
  them in paths.
- Use secure path resolution with `Path.resolve()` and containment checks.
- Keep symlink blocking based on `os.lstat()` and parent-chain checks.
- Keep file size limits and image/array safety caps aligned with tests.
- Do not replace path safety with string concatenation or naive `exists()`.
- Keep CORS restrictive by default; require explicit `CORS_ORIGINS`.
- Do not hardcode the documented production URL in application code.
- Do not commit API keys, ModelScope tokens, generated PID/log files, or local
  server artifacts.

## Code Style

- Keep Python 3.9 compatibility; use `typing.Dict`, `List`, `Optional`, etc.
- Follow the existing 88-character line-length style.
- Group imports as standard library, third-party, first-party.
- Routers should remain thin: validate request shape, check region/task
  existence, call services, translate service errors to HTTP exceptions.
- Put filesystem, ML, rasterization, and inference logic in services.
- Offload blocking I/O or scans from async routes with `asyncio.to_thread` where
  appropriate.
- Use `logging.getLogger(__name__)`.
- Add concise comments only for non-obvious logic.

## Testing

Preferred full project test command:

```bash
python -m pytest tests/ -v
```

Slow SAM3 integration tests:

```bash
python -m pytest tests/ -v -m slow
```

Common focused test files:

- `tests/test_api.py` - regions, patches, embeddings, task APIs, path traversal.
- `tests/test_api_time_formats.py` - month/date normalization and task dates.
- `tests/test_mosaic.py` - mosaic builder behavior and cache.
- `tests/test_auth.py` - API key auth and user isolation.
- `tests/test_models.py` - custom/system model endpoints and result URLs.
- `tests/test_geojson_adapter.py` - GeoJSON rasterization.
- `tests/test_system_models.py` - system model endpoints.
- `tests/test_sam3_*.py` - SAM3 schemas/router/service/config/integration.

Always run tests as `python -m pytest tests/ ...`. Do not run bare `pytest`
over the entire repository unless you intend to collect `sam3_pkg/`; vendored
SAM3 scripts are not part of this service's normal test suite.

## Common Changes

- Add a region: edit `config.yaml` and matching Docker config if needed; add
  patch metadata, embedding versions, raw imagery dirs, and task dirs.
- Add a task: edit the region's `tasks` config and update docs/tests. If it has
  pre-generated outputs, wire `results`, `predictions`, `labels`, and summary
  paths as needed.
- Add an endpoint: add a router or extend an existing router, add schemas if
  needed, keep business logic in `app/services/`, include the router in
  `app/main.py`, and update docs/tests.
- Change file serving: update service validation, size limits, symlink/path
  containment tests, and API docs together.
- Change auth/user storage: check `auth_service.py`, `user_paths.py`, model
  registry behavior, and tests for user isolation.
- Update API docs: keep `README.md`, `docs/API.md`, and this file consistent
  with implementation.

Before considering a code change complete, run the relevant focused tests and,
when practical, `python -m pytest tests/ -v`.
