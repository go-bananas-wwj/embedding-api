# API Quickstart

This guide gives frontend developers short, copyable requests for the most
common integration flows. Use the full reference in [`API.md`](API.md) when you
need every parameter and response field.

```bash
export BASE="http://60.31.21.42:22065"
```

For local development:

```bash
export BASE="http://localhost:9061"
```

## Health and Metadata

```bash
curl -s "$BASE/health"
curl -s "$BASE/regions"
curl -s "$BASE/regions/haidian"
curl -s "$BASE/regions/haidian/patches?page=1&page_size=10"
curl -s "$BASE/regions/haidian/patches/patch_000000"
```

## Haidian Embedding

```bash
curl -s "$BASE/regions/haidian/patches/patch_000000/embedding?format=json&version=v1&month=202512"

curl -s "$BASE/regions/haidian/patches/patch_000000/embedding?format=png&version=v1&month=202512" \
  -o /tmp/haidian_embedding.png
```

## Task Results

```bash
curl -s "$BASE/regions/haidian/tasks"

curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_building.png

curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/road_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_road.png

curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/water_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_water.png
```

## System Model Inference

```bash
curl -s "$BASE/system-models?region_id=haidian"

curl -s -X POST "$BASE/system-models/road_extraction/infer?region_id=haidian&patch_id=patch_000000&month=202512&version=v1"
```

## SAM3 Segmentation

SAM3 point coordinates use WGS84 `[longitude, latitude]`.

```bash
curl -s -X POST "$BASE/regions/haidian/sam3/segment" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "202512",
    "sensor_type": "s2",
    "point_coords": [[116.3000, 39.9800]],
    "point_labels": [1],
    "multimask_output": true,
    "include_masks": false
  }'
```

Response geometry is WGS84 GeoJSON. Each feature is usually a SAM3 mask
`Polygon` or `MultiPolygon`, not a rectangular bbox.

## Custom Model Batch Inference

After creating a model and waiting for training to finish:

```bash
curl -s -X POST "$BASE/models/{model_id}/infer_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "region_id": "haidian",
    "patch_ids": ["patch_000000", "patch_000001"],
    "month": "202512"
  }'
```

The response includes:

- `total`
- `success_count`
- `error_count`
- `results`

## Debug Frontend Requests

Open:

```text
http://60.31.21.42:22065/logs/request-audit
```

The page shows business API requests only. It filters out `/logs`, `/docs`,
`/openapi.json`, `/favicon.ico`, and `/health`.
