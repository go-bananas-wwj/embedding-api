# Static Region Mosaics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the online Mosaic endpoint with a memory-bounded offline PNG package and expose each region's shared mosaic geometry and asset inventory from `/regions`.

**Architecture:** A standalone generator scans real sensor files, renders one region/sensor/date PNG at a time, validates it, and writes it to `{regionId}/{sensor}/{date}/mosaic.png` in a ZIP64 package. The API never generates mosaics; it returns canonical region geometry and the package inventory from `/regions` and `/regions/{region_id}`.

**Tech Stack:** Python 3.9, Rasterio/GDAL, NumPy, Pillow, Shapely, FastAPI/Pydantic, ZIP64, Pytest.

## Global Constraints

- Package PNG only; do not generate or distribute GeoTIFF.
- Asset path is `{regionId}/{sensor}/{date}/mosaic.png`.
- Embedding versions use sensor keys such as `embedding-v1` and `embedding-v2`.
- Use one isolated generator process at a time and enforce a 16 GiB RSS ceiling.
- Keep no-Patch pixels transparent.
- Discover dates from real files; never create empty placeholder images.
- Remove `/regions/{region_id}/mosaic` from the API and Swagger UI.
- Do not commit generated PNG/ZIP files to Git.

---

### Task 1: Region Mosaic API Contract

**Files:**
- Modify: `app/schemas/models.py`
- Modify: `app/routers/regions.py`
- Modify: `tests/test_regions.py`
- Delete: `tests/test_mosaic_endpoint.py`
- Modify: `tests/test_openapi_docs.py`

**Interfaces:**
- Produces: `RegionMosaicInfo`, `RegionMosaicAsset`, and canonical WGS84 geometry in both region endpoints. Each asset reports `start_date`, `end_date`, `date_count`, and exact `available_dates`.
- Removes: `GET /regions/{region_id}/mosaic`.

- [ ] Write failing tests asserting `/regions` and `/regions/{region_id}` include `mosaic.crs`, `bounds_wgs84`, `footprint_wgs84`, corners, format, transparent background, package name, and sensor/month inventory.
- [ ] Write a failing OpenAPI test asserting `/regions/{region_id}/mosaic` is absent.
- [ ] Add Pydantic response models with Chinese field descriptions.
- [ ] Build canonical region geometry from Patch `footprint_wgs84` values and add configured asset inventory.
- [ ] Remove the Mosaic route, runtime imports, response schema, and endpoint-specific tests.
- [ ] Run region and OpenAPI tests.

### Task 2: Offline Inventory and PNG Generator

**Files:**
- Create: `scripts/build_static_mosaic_package.py`
- Modify: `app/services/mosaic_service.py`
- Create: `tests/test_static_mosaic_package.py`

**Interfaces:**
- Produces: `discover_assets()`, `build_asset()`, `validate_png()`, and `write_package()`.
- Output path: `{regionId}/{sensor}/{date}/mosaic.png`.

- [ ] Write failing tests for path normalization, real-date inventory, embedding version names, ZIP manifest entries, checksums, and transparent PNG validation.
- [ ] Reuse raw TIFF discovery and WGS84 reprojection helpers without calling the HTTP API.
- [ ] Run each asset in one child process; poll RSS and terminate above 16 GiB.
- [ ] Generate one PNG at a time, validate RGBA/dimensions/effective pixels, append with ZIP64, then delete the temporary PNG.
- [ ] Write `manifest.json` and `{regionId}/region.json`, including exact counts and SHA256.
- [ ] Support checkpoint/resume and atomic final archive rename.
- [ ] Run generator unit tests.

### Task 3: Generate and Audit the Complete Package

**Files:**
- Generate: `Tmp/static_mosaic_package_20260730/regional-mosaics.zip`
- Generate: `Tmp/static_mosaic_package_20260730/inventory.json`
- Generate: `Tmp/static_mosaic_package_20260730/audit-report.json`
- Generate: `Tmp/static_mosaic_package_20260730/preview/index.html`

**Interfaces:**
- Consumes: all discovered Haidian and Harbin sensor/date assets.
- Produces: one frontend-ready archive plus a visual and machine-readable audit.

- [ ] Run inventory-only mode and record exact asset count and estimated output size.
- [ ] Confirm at least 20 GiB free disk and 18 GiB available memory before rendering.
- [ ] Generate assets sequentially with resumable checkpoints.
- [ ] Validate every ZIP member's path, checksum, PNG signature, dimensions, alpha channel, and WGS84 manifest entry.
- [ ] Create a compact preview covering both regions and representative sensors/months.
- [ ] Report total assets, per-region/per-sensor counts, peak RSS, archive size, failures, and skipped missing dates.

### Task 4: Verification and Deployment

**Files:**
- Modify: `README.md` or linked API documentation only if the removed endpoint is mentioned.
- Modify: `scripts/full_api_audit.py`
- Modify: `scripts/pre_deploy_audit.py`

**Interfaces:**
- API serves metadata only.
- Frontend consumes the delivered ZIP directly.

- [ ] Remove Mosaic calls from audit scripts and documentation.
- [ ] Run the full test suite.
- [ ] Restart the Watchdog-managed API.
- [ ] Verify `/health`, `/regions`, `/regions/haidian`, `/regions/harbin`, and OpenAPI.
- [ ] Verify the removed Mosaic URL returns 404.
- [ ] Confirm repeated `/regions` calls do not increase API RSS materially.
- [ ] Present the ZIP, manifest, preview, and frontend integration note for user review.
- [ ] Commit and push only after package and API review.
