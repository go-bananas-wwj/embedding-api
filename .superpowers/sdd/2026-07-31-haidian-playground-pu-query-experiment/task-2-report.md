# Task 2 Report: Guarded PU + Query Postprocessing

## Status

COMPLETE

## Commit

`d687d53` (`feat: add guarded PU query playground postprocessing`)

P2 validation follow-up: `7d53719` (`fix: validate PU query area guard inputs`).

## Changed Files

- `scripts/playground_pu_postprocess.py`
- `tests/test_playground_pu_postprocess.py`
- `.superpowers/sdd/2026-07-31-haidian-playground-pu-query-experiment/task-2-report.md`

`app/services/pu_query.py` and all API behavior remain unchanged.

## Tests And Results

Red phase:

```text
/opt/conda/envs/pyseims/bin/python -m pytest tests/test_playground_pu_postprocess.py -q
```

Result before implementation: expected collection failure with
`ModuleNotFoundError: No module named 'scripts.playground_pu_postprocess'`.

Final focused verification:

```text
/opt/conda/envs/pyseims/bin/python -m pytest tests/test_playground_pu_postprocess.py -q
```

Result: `8 passed in 0.31s`.

Full service verification:

```text
/opt/conda/envs/pyseims/bin/python -m pytest tests/ -q -rA
```

Result: `305 passed, 31 warnings in 60.26s`. The warnings are pre-existing
third-party deprecation, TIFF metadata, and scikit-learn checkpoint-version
warnings; no test failed.

P2 validation follow-up:

```text
/opt/conda/envs/pyseims/bin/python -m pytest tests/test_playground_pu_postprocess.py -q
```

Result: `18 passed in 0.30s`, including five invalid values each for
`min_pixels` and `max_component_pixels`: Python and NumPy booleans, fractional
floats, `NaN`, and infinity.

```text
/opt/conda/envs/pyseims/bin/python -m pytest tests/ -q
```

Result: `317 passed, 31 warnings in 58.84s`.

## API And Signature

- `strict_threshold(scores: List[np.ndarray], labels: List[np.ndarray]) -> float`
  scans 180 thresholds from the offline production floor `0.247057` through
  the calibration scores' 99.9th percentile. It ranks F1, then precision, then
  smaller predicted area.
- `hysteresis_prediction(score: np.ndarray, high: float, low: float,
  min_pixels: int, max_component_pixels: Optional[int] = None,
  max_total_ratio: Optional[float] = None) -> np.ndarray`
  uses 8-connected candidates and seeds. It rejects unseeded, undersized, and
  optionally oversized components. With an area-ratio cap, it greedily retains
  whole eligible components by descending seed maximum, then component order.
- `binary_metrics(prediction: np.ndarray, reference: np.ndarray) -> Dict[str, float>`
  returns precision, recall, F1, IoU, positive ratio, and 8-connected component
  count.
- `component_statistics(prediction: np.ndarray, score: np.ndarray) -> List[Dict[str, object]]`
  returns each component's area, mean score, maximum score, and exclusive
  raster bounding box `[row_min, column_min, row_max, column_max]`.

## Numerical Edge Cases

- Scores must be finite and match their corresponding labels; score maps and
  diagnostic maps must be two-dimensional where spatial connectivity applies.
- Hysteresis rejects non-finite thresholds and a `high` value below `low`.
- `min_pixels` and `max_component_pixels`, when provided, must be finite
  non-boolean integers; NumPy integer types are accepted. `min_pixels` is at
  least one and `max_component_pixels` cannot be below it. `max_total_ratio`
  must be in `[0, 1]`.
- Empty metric denominators return `0.0`, avoiding NaN/Infinity metrics.
- A fractional total-area cap floors to an integer pixel budget; components are
  never partially retained.

## Concerns

- The production floor is intentionally pinned to the approved current head
  threshold (`0.247057`). Update this offline constant when that head is
  retrained or its threshold changes.
- Area guards are generic raster constraints. Their numerical settings still
  require calibration against held-out references; no OSM data or fixed
  playground dimensions are used during inference.
