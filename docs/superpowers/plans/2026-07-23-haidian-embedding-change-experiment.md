# Haidian Embedding Change Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a browser-based comparison of cosine change distance and normalized Euclidean distance for Haidian P10C embeddings from `202512` to `202604`.

**Architecture:** A focused offline experiment script loads paired 64D embeddings and same-month Sentinel-2 images, computes both change metrics with one shared valid mask, derives global percentile color limits, selects representative patches, and writes PNG assets plus a self-contained HTML gallery. Production API code and model registry remain unchanged.

**Tech Stack:** Python 3.11, NumPy, rasterio, Pillow, Matplotlib, pytest, static HTML/CSS/JavaScript.

## Global Constraints

- Compare Haidian `202512` with `202604`.
- Use Haidian P10C 64D embedding `v1`.
- Blue means low change and red means high change.
- Both methods use the same patches and valid-pixel mask.
- All patches share global color limits.
- Optical previews use robust percentile stretching.
- Output is stored under `Tmp` and supports click-to-enlarge.
- Do not modify the production API.

---

### Task 1: Change metric functions

**Files:**
- Create: `scripts/experiment_haidian_embedding_change.py`
- Create: `tests/test_embedding_change_experiment.py`

**Interfaces:**
- Produces: `change_scores(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
- Returns: cosine distance map, normalized Euclidean map, shared valid mask.

- [x] **Step 1: Write failing metric tests**

Test identical vectors, orthogonal vectors, zero vectors, and `C,H,W` shape validation. Assert identical vectors score zero, orthogonal vectors score one for cosine and `sqrt(2)` for Euclidean, and zero vectors are invalid.

- [x] **Step 2: Run the focused test**

Run: `/opt/conda/envs/pyseims/bin/python -m pytest tests/test_embedding_change_experiment.py -q`

Expected: FAIL because the experiment module does not exist.

- [x] **Step 3: Implement the metric function**

Move channels last, reject mismatched shapes, L2-normalize with epsilon `1e-8`, compute `1 - dot(unit_before, unit_after)` and `norm(unit_before - unit_after)`, and set invalid positions to `NaN`.

- [x] **Step 4: Verify metric tests**

Run the focused test again.

Expected: PASS.

### Task 2: Data loading, global calibration, and representative selection

**Files:**
- Modify: `scripts/experiment_haidian_embedding_change.py`
- Modify: `tests/test_embedding_change_experiment.py`

**Interfaces:**
- Produces: `robust_rgb(path: Path) -> np.ndarray`
- Produces: `global_limits(score_maps: list[np.ndarray]) -> tuple[float, float]`
- Produces: `select_representative_patches(stats: list[dict], count: int = 12) -> list[str]`

- [x] **Step 1: Add failing tests**

Assert robust RGB output is `uint8`, global limits ignore `NaN`, and representative selection covers low, medium, and high P95 scores without duplicate patch IDs.

- [x] **Step 2: Run the focused test and confirm failure**

Run the same focused pytest command.

- [x] **Step 3: Implement loaders and calibration**

Resolve the latest available Sentinel-2 scene within each requested month for each Patch, read RGB bands, apply a per-band 2nd-to-98th percentile stretch, scan all 320 paired embeddings, and derive global P02/P98 display limits for each metric.

- [x] **Step 4: Verify focused tests**

Run the focused pytest command.

Expected: PASS.

### Task 3: Generate and inspect the gallery

**Files:**
- Modify: `scripts/experiment_haidian_embedding_change.py`
- Create at runtime: `Tmp/haidian_embedding_change_<timestamp>/index.html`
- Create at runtime: `Tmp/haidian_embedding_change_<timestamp>/results.json`

**Interfaces:**
- Consumes both metric maps, global limits, optical previews, and selected Patch IDs.
- Produces a static HTML gallery and PNG assets.

- [x] **Step 1: Render six views per Patch**

Render December optical, April optical, cosine heatmap, cosine overlay, Euclidean heatmap, and Euclidean overlay. Use the shared blue-white-red colormap and fixed limits.

- [x] **Step 2: Add quantitative context**

Write mean, P90, P95, and global-high-change pixel share for each method to `results.json` and the gallery. Add the overall Spearman/Pearson relationship between the two methods.

- [x] **Step 3: Add browser interaction**

Create responsive HTML with Chinese titles, fixed image dimensions, visible legends, and a click-to-enlarge modal.

- [x] **Step 4: Run the experiment**

Run: `/opt/conda/envs/pyseims/bin/python scripts/experiment_haidian_embedding_change.py`

Expected: prints one output directory under `Tmp`.

- [x] **Step 5: Inspect generated output**

Open representative PNGs and verify images are nonblank, titles do not overlap, both months are correct, and the same metric uses the same color limits across Patch rows.

- [x] **Step 6: Run regression tests**

Run: `/opt/conda/envs/pyseims/bin/python -m pytest tests/test_embedding_change_experiment.py tests/test_openapi_docs.py -q`

Expected: PASS.
