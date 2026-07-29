# Haidian Land Label Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace unreliable Haidian land-use and noisy land-cover PNGs with reproducible P10C Conv 3x3 predictions.

**Architecture:** Train one seven-class Conv 3x3 head from normalized WorldCover labels, correcting the verified `1=water`, `8=tree` mapping and strengthening built-up supervision with OSM building masks. Render the same semantic prediction through separate land-cover and land-use palettes; omit unsupported flooded-vegetation and snow classes. Remove only tiny isolated water components.

**Tech Stack:** Python, PyTorch, NumPy, Rasterio, Pillow, SciPy.

## Global Constraints

- Keep all existing API paths and request parameters unchanged.
- Use Haidian P10C v1 64-dimensional embeddings.
- Persist checkpoint, mapping, provenance, and metrics.
- Never describe the source snapshot as ESA WorldCover 2021 because local metadata does not prove that year.

---

### Task 1: Reproducible multiclass training

**Files:**
- Create: `scripts/retrain_haidian_land_labels.py`

- [ ] Load WorldCover labels and map values `1,2,3,4,5,6,8` to contiguous logits.
- [ ] Override building-positive pixels with class 5 during supervision construction.
- [ ] Train a weighted seven-class Conv 3x3 head with a fixed split and seed.
- [ ] Save the checkpoint and validation metrics under `models/haidian/v1/task_heads/`.

### Task 2: Monthly result regeneration

**Files:**
- Modify: `scripts/retrain_haidian_land_labels.py`

- [ ] Infer all six P10C months for all available patches.
- [ ] Remove isolated water components smaller than four pixels by local majority replacement.
- [ ] Render land-cover and land-use PNGs with their documented palettes.
- [ ] Save numeric predictions alongside PNG results.

### Task 3: API metadata and verification

**Files:**
- Modify: `app/services/system_model_service.py`
- Modify: `config.yaml`
- Test: `tests/test_system_models.py`

- [ ] Advertise only the seven land-use classes supported by reliable supervision.
- [ ] Add checkpoint and provenance metadata to task descriptions.
- [ ] Verify classes, PNG colors, summary legends, and representative optical comparisons.
