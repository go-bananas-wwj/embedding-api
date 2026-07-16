# API 快速联调

本文档给前端同事提供最常用的接口示例。完整参数和返回结构请看 [`API.md`](API.md) 或 Swagger。

```bash
export BASE="http://60.31.21.42:22065"
```

本地开发时：

```bash
export BASE="http://localhost:9061"
```

## 健康检查和基础元数据

```bash
curl -s "$BASE/health"
curl -s "$BASE/regions"
curl -s "$BASE/regions/haidian"
curl -s "$BASE/regions/haidian/patches?page=1&page_size=10"
curl -s "$BASE/regions/haidian/patches/patch_000000"
```

## 海淀 Embedding

```bash
curl -s "$BASE/regions/haidian/patches/patch_000000/embedding?format=json&version=v1&month=202512"

curl -s "$BASE/regions/haidian/patches/patch_000000/embedding?format=png&version=v1&month=202512" \
  -o /tmp/haidian_embedding.png
```

## 下游任务结果

```bash
curl -s "$BASE/regions/haidian/tasks"

curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_building.png

curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/road_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_road.png

curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/water_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_water.png
```

## 系统模型推理

```bash
curl -s "$BASE/system-models?region_id=haidian"

curl -s -X POST "$BASE/system-models/road_extraction/infer?region_id=haidian&patch_id=patch_000000&month=202512&version=v1"
```

## SAM3 分割

SAM3 点坐标使用 WGS84，经纬度顺序为 `[longitude, latitude]`。

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

返回结果是 WGS84 GeoJSON。每个 feature 通常是 SAM3 mask 的 `Polygon` 或 `MultiPolygon`，不是矩形框。

高分辨率 RGB 光学 GeoTIFF 使用 `"sensor_type": "highres"`。影像需预先切到
对应 patch，包含 CRS 和仿射变换，前三个波段依次为 R/G/B。

## 自定义模型批量推理

创建模型并等待训练完成后：

```bash
curl -s -X POST "$BASE/models/{model_id}/infer_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "region_id": "haidian",
    "patch_ids": ["patch_000000", "patch_000001"],
    "month": "202512"
  }'
```

返回包含：

- `total`
- `success_count`
- `error_count`
- `results`

## 查看前端请求

打开：

```text
http://60.31.21.42:22065/logs/request-audit
```

页面只展示真实业务 API 请求，会过滤 `/logs`、`/docs`、`/openapi.json`、`/favicon.ico`、`/health`。
