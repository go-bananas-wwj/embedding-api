# AGENTS.md — Embedding API

> This file is written for AI coding agents. It assumes no prior knowledge of the project. Information is derived directly from the codebase; do not rely on external assumptions.

## Project Overview

**Embedding API** is a Python FastAPI service that exposes a unified REST API for remote sensing embeddings and downstream monitoring task results. It currently serves two geographic regions:

- **Harbin New Area (哈尔滨新区)** — 424 patches
- **Haidian District (海淀区)** — 320 patches

The API provides:

- Embedding queries in PNG, NPY, NPZ, JSON, and cache-fallback formats.
- 5 unified downstream thematic tasks: change detection, building extraction, land use classification, land cover classification, and water extraction.
- Patch metadata, bbox filtering, pagination.
- Map tile listing (XYZ tile serving is stubbed but not implemented).
- SAM3 interactive segmentation (embed + segment with point prompts on Sentinel-2 imagery).
- User-generated annotation storage (classes + geometries) for custom training data.
- Custom model training and inference (classification and change-detection heads).
- System pre-trained model inference.
- Optional API-Key authentication with per-user data isolation.
- Hot-reload configuration via `watchdog`.

### Technology Stack

- **Runtime**: Python >= 3.9
- **Web Framework**: FastAPI 0.104+, Pydantic v2
- **Server**: uvicorn (ASGI)
- **Configuration**: YAML (`config.yaml`), hot-reloaded with `watchdog`
- **ML/DL**: PyTorch 2.5.1, torchvision 0.20.1, `sam3` (local package in `sam3_pkg/`)
- **Image/Geo**: Pillow >= 11, rasterio >= 1.3, numpy 1.26.4
- **Containerization**: Docker + docker-compose
- **Process Supervision**: `service_watchdog.py` (optional)

### Repository Layout

```
embedding-api/
├── app/                      # Application source
│   ├── main.py               # FastAPI app factory, CORS, router inclusion
│   ├── config.py             # YAML config manager with hot-reload
│   ├── routers/              # FastAPI route handlers
│   │   ├── regions.py
│   │   ├── patches.py
│   │   ├── embeddings.py
│   │   ├── tasks.py
│   │   ├── annotations.py
│   │   ├── models.py
│   │   ├── system_models.py
│   │   └── sam3.py
│   ├── schemas/              # Pydantic request/response models
│   │   ├── models.py
│   │   └── sam3.py
│   └── services/             # Business logic
│       ├── data_service.py
│       ├── tile_service.py
│       ├── sam3_service.py
│       ├── auth_service.py
│       ├── annotation_service.py
│       ├── model_registry.py
│       ├── training_engine.py
│       └── inference_engine.py
├── tests/                    # pytest test suite
├── data/                     # Regional patch metadata and task outputs
├── models/                   # Trained model checkpoints
├── sam3_pkg/                 # Local SAM3 package (has its own pyproject.toml)
├── docs/API.md               # Frontend-oriented API documentation (Chinese)
├── config.yaml               # Default runtime configuration
├── config.docker.yaml        # Docker-specific configuration
├── requirements.txt          # Python dependencies
├── Dockerfile
├── docker-compose.yml
└── service_watchdog.py       # Optional process supervisor
```

There is **no root `pyproject.toml`**. Dependency management is done through `requirements.txt`. The `sam3_pkg/pyproject.toml` belongs only to the bundled SAM3 package.

## Build and Run Commands

### Install Dependencies

```bash
pip install -r requirements.txt
```

The `sam3` package is installed from the local `sam3_pkg/` directory; `requirements.txt` does not include it, but the import path works because the package is vendored in the repo.

### Local Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 9061
```

- Swagger UI (if enabled): `http://localhost:9061/docs`
- ReDoc (if enabled): `http://localhost:9061/redoc`

### Production / Watchdog

```bash
# Run the optional watchdog, which restarts uvicorn if it becomes unhealthy
python service_watchdog.py

# Stop the watchdog
python service_watchdog.py stop
```

The watchdog:

- Manages a uvicorn process on port `9061`.
- Health-checks `http://localhost:9061/health` every 30 seconds.
- Uses exponential backoff restarts: `[5, 10, 30, 60, 120]` seconds.
- Stores its PID in `service_watchdog.pid`.

### Docker

```bash
docker-compose up -d
```

- Internal port: `8000`
- Mapped host port: `8000`
- Uses `config.docker.yaml` mounted as `/app/config.yaml`.
- Data/model directories are expected to be mounted as read-only volumes (see `docker-compose.yml`).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_PATH` | `./config.yaml` | Path to the YAML configuration file |
| `CORS_ORIGINS` | *(empty)* | Comma-separated allowed origins; empty means cross-origin requests are rejected |
| `DOCS_URL` | `none` | Swagger UI path; set `/docs` to enable; default disabled |
| `REDOC_URL` | `none` | ReDoc path; set `/redoc` to enable; default disabled |

Note: the code reads `CONFIG_PATH`, not `CONFIG_FILE` (despite what `docs/API.md` says).

## Code Organization and Module Divisions

### `app/main.py`

- Creates the FastAPI app with lifespan startup/shutdown hooks.
- Configures CORS from `CORS_ORIGINS`.
- Disables docs by default; enable via `DOCS_URL` / `REDOC_URL`.
- Includes all routers.

### `app/config.py`

- Thread-safe singleton `ConfigManager`.
- Loads `config.yaml` and exposes `get_config()`.
- Watches the config file with `watchdog`; reloads on change with a 0.5s debounce.
- Loads `patches_meta` JSON files lazily and caches patch lists.
- Provides `register_reload_callback()` for other modules to clear caches on config reload.

### `app/routers/`

Route modules follow standard FastAPI patterns:

- Validate region existence.
- Validate query parameters.
- Delegate to services.
- Translate service exceptions to HTTP exceptions (400, 404, 413, 422, 500, 501, 503).

### `app/services/data_service.py`

- Central path-resolution service for embeddings, task results, predictions, labels, and summaries.
- Validates `patch_id` (`patch_\d{6}`), `month`, and `period` against path-traversal patterns.
- Uses `_resolve_path()` to ensure targets are inside configured base directories and are not symlinks.
- Enforces a 100MB file-size limit (`MAX_FILE_SIZE`).
- Implements an LRU+TTL cache for `available_tasks` and registers a callback to clear it on config reload.

### `app/services/tile_service.py`

- Lists available patch-based tile PNG files.
- XYZ tile serving is intentionally not implemented (returns HTTP 501).

### `app/services/sam3_service.py`

- Singleton service for SAM3 model loading and inference.
- Lazy-loads the model on first use.
- Loads Sentinel-2 RGB imagery via `rasterio` from the region's `s2_dir`.
- Caches embeddings in an LRU cache (`max_cache_size` from config, default 20).
- Uses `asyncio.Lock` to serialize GPU inference.
- Converts model weights to `bfloat16` for autocast compatibility.

## Configuration

All data paths and task definitions are driven by `config.yaml`. Key sections:

- `sam3`: model path, BPE tokenizer path, device, max cache size, image size.
- `models`: downstream-task model checkpoints by region and version.
- `regions`: per-region patch metadata, embedding directories, Sentinel-2 directories, and tasks.

Adding a new region or task only requires editing `config.yaml`; the service picks up changes without restart.

### Embedding Path Templates

Embeddings support config-driven templates such as:

```yaml
embeddings:
  v1:
    path: "data/harbin/embeddings/v1"
    template: "{month}/{patch_id}.{fmt}"
    formats: ["npy", "png", "json"]
    alt_templates: ["{month}/{patch_id}.png"]
```

For Haidian, the structure uses patch subdirectories and NPZ archives:

```yaml
embeddings:
  v1:
    path: "data/haidian/embeddings/v1"
    template: "{patch_id}/patch_{patch_id}_{month}.npz"
    formats: ["npz", "png", "json"]
```

## Code Style Guidelines

- **Python version**: 3.9+. Avoid syntax newer than 3.9 unless absolutely necessary.
- **Line length**: Follow the existing 88-character convention (the bundled `sam3_pkg` uses Black with `line-length = 88`).
- **Imports**: standard library, third-party, first-party, grouped with blank lines.
- **Type hints**: use `typing` generics (e.g., `Dict[str, Any]`, `Optional[str]`) for 3.9 compatibility.
- **Async**: routers are async; blocking I/O and filesystem scans are offloaded with `asyncio.to_thread`.
- **Logging**: use `logging.getLogger(__name__)`; format is `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`.
- **Docstrings**: module-level and class-level docstrings are expected.
- **Security-sensitive code**: path validation, symlink checks, and size limits must be preserved or updated together.

## Testing Instructions

### Run Tests

```bash
# Run only the project test suite
python -m pytest tests/ -v

# Run all tests including the slow SAM3 integration tests (require GPU + model)
python -m pytest tests/ -v -m slow
```

### Test Organization

- `tests/test_api.py` — Endpoint tests for regions, patches, embeddings, tasks, path traversal.
- `tests/test_auth.py` — Authentication and user isolation tests.
- `tests/test_annotations.py` — Annotation and class management tests.
- `tests/test_models.py` — Custom model CRUD, training, and inference tests.
- `tests/test_system_models.py` — System pre-trained model tests.
- `tests/test_sam3_service.py` — Unit tests for `SAM3Service` with mocked model/image loading.
- `tests/test_sam3_router.py` — Router-level validation tests for SAM3 endpoints.
- `tests/test_sam3_schemas.py` — Pydantic schema validation tests.
- `tests/test_sam3_config.py` — Config integration tests.
- `tests/test_sam3_integration.py` — Real model loading tests; marked with `@pytest.mark.slow`.

The integration tests may return 200, 404, or 503 depending on GPU/data availability; they are designed not to fail hard when the model or data is missing.

### Test Environment Notes

- `sam3_pkg/scripts/qualitative_test.py` is **not** part of the project test suite and may error during collection if pytest discovers it. Always run `python -m pytest tests/` explicitly.
- Tests use `fastapi.testclient.TestClient` and do not require a running server.

## Security Considerations

This service serves files from the local filesystem based on user-supplied path fragments. Security is enforced at multiple layers:

1. **Patch ID validation**: strict regex `patch_\d{6}`. Malformed IDs are rejected before any filesystem access.
2. **Month / period validation**: only alphanumeric, hyphen, and underscore allowed; max length enforced.
3. **Path containment**: `_resolve_path()` resolves paths with `Path.resolve()` and verifies `target.relative_to(base)`. It also checks the parent chain.
4. **Symlink blocking**: `os.lstat()` is used to detect symlinks without following them, mitigating symlink-based escapes and TOCTOU races.
5. **File size limits**: 100MB max for served files; embedding images are capped at 50M pixels; numpy arrays are capped at 500M elements to prevent memory-bomb attacks via malicious `.npy` headers.
6. **CORS**: default `allow_origins=[]`; explicit origins are required via `CORS_ORIGINS`.
7. **Authentication**: API-Key authentication is built-in and optional. When `config.yaml` contains `auth.users`, protected endpoints (`/annotations/*`, `/models/*`, `/system-models/*`) require a valid `X-API-Key` or `Authorization: Bearer <key>` header. Public deployments should still place the service behind a reverse proxy for TLS and additional access control.
8. **Docs**: Swagger/ReDoc are disabled by default in production. Enable only via environment variables.
9. **Docker**: runs as a non-root user (`appuser`, uid 1000); data and models are mounted read-only.

When modifying file-serving code, always preserve or extend these defenses. Do not switch to naive string concatenation or `os.path.exists()` checks without symlink protection.

## Deployment Notes

- The production base URL referenced in documentation is `http://60.31.21.42:22065`. This is documented for frontend teams; do not hardcode it in code.
- The Dockerfile exposes port `8000` and runs uvicorn with `--workers 2`.
- `docker-compose.yml` mounts external data directories. Ensure these paths exist on the host or update the compose file.
- The watchdog is optional and intended for bare-metal deployments where no external process manager is available.

## Common Tasks for Agents

- **Add a new region**: edit `config.yaml` (or `config.docker.yaml` for Docker), add `patches_meta`, embedding paths, and optional tasks. No code change needed.
- **Add a new task**: edit the region's `tasks` section in `config.yaml`; specify `results`, `predictions`, and `labels` directories per version.
- **Change CORS or docs**: set `CORS_ORIGINS`, `DOCS_URL`, or `REDOC_URL` environment variables.
- **Modify file size / cache limits**: update constants in `app/services/data_service.py` or `app/routers/embeddings.py`, and keep tests aligned.
- **Add new endpoints**: create a router under `app/routers/`, add schemas under `app/schemas/`, implement logic under `app/services/`, and include the router in `app/main.py`.
- **Run after changes**: `python -m pytest tests/` must pass before considering a change complete.
