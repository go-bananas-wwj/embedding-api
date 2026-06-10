# Embedding API 接口文档

## 概述

统一的后端接口服务，为前端团队提供哈尔滨新区和海淀区的遥感嵌入及下游任务结果查询。

**Base URL**: `http://localhost:8000`

**API 文档**: `/docs` (Swagger UI) | `/redoc` (ReDoc)

---

## 基础接口

### Health Check
```
GET /health
```

**Response**:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "regions": ["harbin", "haidian"]
}
```

---

## 区域管理

### 列出所有区域
```
GET /regions
```

**Response**:
```json
{
  "regions": [
    {
      "id": "harbin",
      "name": "哈尔滨新区",
      "patch_count": 424,
      "tasks": ["construction", "building_change", "farmland", "land_conversion", "demolition"]
    },
    {
      "id": "haidian",
      "name": "海淀区",
      "patch_count": 320,
      "tasks": []
    }
  ]
}
```

### 获取区域详情
```
GET /regions/{region_id}
```

---

## Patch 管理

### 列出 Patch
```
GET /regions/{region_id}/patches?page=1&page_size=20&bbox=minx,miny,maxx,maxy
```

**Parameters**:
- `page` (int, default=1): 页码
- `page_size` (int, default=20, max=100): 每页数量
- `bbox` (string, optional): 地理范围过滤，格式 `minx,miny,maxx,maxy`

**Response**:
```json
{
  "total": 424,
  "page": 1,
  "page_size": 20,
  "patches": [
    {
      "patch_id": "patch_000000",
      "bounds_wgs84": [126.51631, 45.743707, 126.532242, 45.755574],
      "sources": {"s2": 182, "s1": 100, ...},
      "time_range": ["2023-01", "2025-10"],
      "has_embedding": true,
      "available_tasks": ["construction", "building_change", "farmland"]
    }
  ]
}
```

### 获取 Patch 详情
```
GET /regions/{region_id}/patches/{patch_id}
```

---

## 嵌入服务

### 获取嵌入数据
```
GET /regions/{region_id}/patches/{patch_id}/embedding?format=png|npy|json
```

**Parameters**:
- `format` (string, default="png"): 输出格式
  - `png`: 返回可视化图像 (image/png)
  - `npy`: 返回原始数组 (application/octet-stream)
  - `json`: 返回统计信息 (application/json)

**哈尔滨**: 返回 64×64 RGB PNG 可视化图
**海淀**: 返回 64×128×128 float32 NPY 数组或 JSON 统计

**JSON Response**:
```json
{
  "patch_id": "patch_000000",
  "shape": [64, 128, 128],
  "dtype": "float32",
  "min": -0.476,
  "max": 0.404,
  "mean": -0.005
}
```

---

## 下游任务

### 列出任务
```
GET /regions/{region_id}/tasks
```

**Response**:
```json
{
  "tasks": [
    {
      "id": "construction",
      "name": "建筑工地监测",
      "description": "监测建筑工地变化情况",
      "versions": ["v1", "v2"]
    }
  ]
}
```

### 任务统计摘要
```
GET /regions/{region_id}/tasks/{task_type}/summary?version=v1&period=2025-04_vs_2025-06
```

**Parameters**:
- `version` (string, default="v1"): 版本号
- `period` (string, optional): 时间段（V2 格式如 `2025-04_vs_2025-06`）

**Response**:
```json
{
  "task": "construction",
  "name": "建筑工地监测",
  "version": "v2",
  "period": "2025-04_vs_2025-06",
  "grid_size": 64,
  "total_polygons": 24,
  "total_patches": 424,
  "positive_patches": 31,
  "negative_patches": 393
}
```

### 单 Patch 任务结果
```
GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result?format=png|npy&version=v1&period=...
```

**Parameters**:
- `format` (string, default="png"): `png` 或 `npy`
- `version` (string, default="v1"): 版本号
- `period` (string, optional): 时间段

### 原始预测数据
```
GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/prediction?version=v1&period=...
```

返回 `.npy` 文件（application/octet-stream）

### 标签数据
```
GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/label?version=v1&period=...
```

返回 `.npy` 标签文件或 `meta.json`

---

## 瓦片服务

### 列出可用瓦片
```
GET /regions/{region_id}/tasks/{task_type}/tiles?version=v1&period=...
```

### 获取瓦片
```
GET /regions/{region_id}/tasks/{task_type}/tiles/{z}/{x}/{y}.png?version=v1&period=...
```

用于前端地图叠加（Leaflet / Mapbox / OpenLayers 等）

---

## 错误码

| Status | Meaning |
|--------|---------|
| 200 | OK |
| 404 | Region/Patch/Task/Result not found |
| 422 | Invalid query parameters |
| 500 | Internal server error |

---

## 动态扩展

编辑 `config.yaml` 添加新区域或任务，服务自动检测变更，无需重启。

示例：添加新区域
```yaml
regions:
  new_city:
    name: "新城市"
    patches_meta: "/path/to/patches_meta.json"
    embeddings:
      v1: "/path/to/embeddings"
    tasks:
      new_task:
        name: "新监测任务"
        versions:
          v1:
            results: "/path/to/results"
            predictions: "/path/to/predictions"
            labels: "/path/to/labels"
```
