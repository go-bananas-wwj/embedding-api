# Embedding API Implementation Plan

Date: 2026-07-06

This plan records the requested production update for the Haidian model assets,
SAM3 interaction API, custom training, batch inference, API documentation, and
service operation.

## Goals

1. Replace the previous Haidian assets with the latest ModelScope dataset:
   `WeijieWu/xuannv_haidian_embdding`.
2. Keep a background service running on port `9061` with persistent logs for
   later error inspection.
3. Make SAM3 accept WGS84 point prompts and a sensor type, then return WGS84
   GeoJSON annotation boxes.
4. Fix custom training and batch inference so they run with real frontend
   GeoJSON annotation packages.
5. Make API contracts and documentation clear enough for frontend integration.
6. Review the API against common REST/FastAPI best practices and verify the
   service with real endpoint calls.

## Asset Migration

- Download the latest Haidian package from ModelScope.
- Convert Haidian embedding maps into API-ready files:
  `data/haidian/embeddings/v1/{month}/{patch_id}.npy|png|json`.
- Install the latest embedding checkpoint at:
  `models/haidian/v1/embedding/haidian_embedding_v1_p10c_epoch800.pt`.
- Install latest task heads at:
  `models/haidian/v1/task_heads/{building,road,water}_mlp_fold0_best.pt`.
- Remove old Haidian patch-subdirectory embedding layout so the API cannot
  accidentally serve stale assets.
- Update `config.yaml` and Haidian pipeline scripts to make the new ModelScope
  package the default source of truth.

## API and Backend Changes

### SAM3

- `POST /regions/{region_id}/sam3/segment` accepts:
  - `date`
  - `sensor_type`: `s2`, `s1`, or `landsat`
  - `point_coords`: WGS84 `[lon, lat]`
  - optional `point_labels`: foreground/background labels; omitted labels
    default to foreground (`1`) for every point
- The service finds the patch containing the prompt points, loads the matching
  raw image, computes or reuses the SAM3 embedding, and returns a GeoJSON
  `FeatureCollection`.
- Each returned feature contains a WGS84 polygon box plus properties including
  `score`, pixel `bbox`, `bbox_wgs84`, `patch_id`, `sensor_type`, `date`, and
  `geometry_kind`.
- CRS conversion falls back to `pyproj` when the host rasterio/PROJ database is
  stale, preserving correct WGS84 behavior in the current deployment
  environment.

### Custom Training

- `POST /models` now requires a real training request instead of silently
  creating a demo model.
- The request body uses `annotations` as a WGS84 GeoJSON FeatureCollection and
  `classes` as explicit frontend class metadata.
- Training runs through a background job and exposes `job_id` for polling.
- Class subset semantics are explicit: feature `class_id` must exist in
  `classes`; if `class_ids` is provided, unselected classes are ignored during
  training.

### Inference and Batch Inference

- Single inference requires explicit `region_id`, `patch_id`, and time fields.
- Batch inference validates `patch_ids`, limits each request to 100 patches,
  and returns batch-level metadata:
  `total`, `success_count`, `error_count`, and `results`.
- System model inference supports the latest Haidian MLP task heads.

### Security and Robustness

- Preserve path containment, symlink blocking, and file size limits.
- Validate `user_id` before using it as a user directory name.
- Validate downloadable result filenames as single PNG filenames.
- Keep API validation errors as client errors instead of masking them as server
  failures.

## Documentation Updates

- `AGENTS.md`: project-specific agent instructions, current architecture,
  Haidian ModelScope source, SAM3 contract, testing, deployment, and security.
- `README.md`: latest Haidian install flow, service logs, docs defaults, and
  API overview.
- `docs/API.md`: frontend-facing endpoint contracts, SAM3 WGS84 GeoJSON
  request/response, batch inference metadata, and custom training examples.
- `docs/custom-training-workflow.md`: human-readable custom training and batch
  inference workflow.
- `pipelines/haidian/README.md`: latest ModelScope asset layout and install
  commands.

## Verification Plan

1. Run the full project test suite:
   `python -m pytest tests/ -q`.
2. Verify latest Haidian files:
   - 6 months: `202512` through `202605`
   - 1920 `.npy`, 1920 `.png`, 1920 `.json`
   - no old `patch_*` embedding directories
3. Start the service with `service_watchdog.py`.
4. Save API smoke-test outputs under `audit_results/`.
5. Confirm these real endpoint flows return 2xx:
   - health and region listing
   - Haidian patch metadata and embedding JSON
   - Haidian system model inference
   - unified `/models/*/infer_batch`
   - custom model creation, job polling, single inference, batch inference
   - SAM3 WGS84 point segmentation returning GeoJSON boxes
6. For full pre-release verification, run:
   `python scripts/full_api_audit.py`.
   The script reads the live OpenAPI document, calls every public operation,
   saves every response body under `audit_results/api_full_*`, downloads
   generated result PNGs, validates PNG dimensions, checks batch success counts,
   verifies SAM3 cache/mask behavior, and treats the documented XYZ tile route
   as the only expected `501` stub.

## Runtime Operation

- Service URL: `http://127.0.0.1:9061`
- Swagger UI: `http://127.0.0.1:9061/docs`
- Watchdog log: `logs/watchdog.log`
- Uvicorn/API log: `logs/uvicorn.log`
- Latest smoke-test success summary:
  `audit_results/api_final_20260706_025233/summary_passed.json`
- Latest full OpenAPI audit summary:
  `audit_results/api_full_20260706_030125/summary.json`

## Known Follow-Ups

- The current environment has an old GDAL/PROJ database on the rasterio path.
  SAM3 handles this with a `pyproj` fallback, but the deployment image should
  eventually align GDAL, rasterio, and PROJ versions.
- FastAPI/OpenAPI can later be enhanced with a global API-key security scheme
  and a uniform error envelope. These are compatibility improvements rather
  than blockers for the requested flows.
