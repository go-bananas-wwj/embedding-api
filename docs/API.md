# Embedding API 接口文档 v2.0

> **文档版本**: 2.0  
> **服务版本**: 0.1.0  
> **最后更新**: 2026-07-06  
> **GitHub**: [go-bananas-wwj/embedding-api](https://github.com/go-bananas-wwj/embedding-api)

---

## ✅ 快速测试清单

下面列出本文档中所有接口的 `curl` 命令，按顺序排列。复制粘贴即可逐条测试。

> **说明**：
> - 基础 URL 统一使用生产环境 `http://60.31.21.42:22065`。
> - 模型相关接口（`/models/*`、`/system-models/*`）在 `config.yaml` 中未配置 `auth` 时，使用默认用户 `default`，无需 API Key。
> - 若配置了 API Key，请在命令中加上 `-H 'X-API-Key: your_key'` 或 `-H 'Authorization: Bearer your_key'`。
> - 带 `model_id`、`job_id`、`filename` 的命令需要先调用创建接口获取真实 ID，再替换示例值。

```bash
# 1. 基础接口
curl -s "http://60.31.21.42:22065/health"
curl -s "http://60.31.21.42:22065/regions"
curl -s "http://60.31.21.42:22065/regions/harbin"

# 2. Patch 列表与详情
curl -s "http://60.31.21.42:22065/regions/harbin/patches?page=1&page_size=5"
curl -s "http://60.31.21.42:22065/regions/harbin/patches?page=1&page_size=20&bbox=126.5,45.74,126.55,45.76"
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000"

# 3. Embedding（多种格式）
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/embedding?format=png&version=v2&month=2025-04" -o /tmp/emb_patch_000000.png
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/embedding?format=npy&version=v2&month=2025-04" -o /tmp/emb_patch_000000.npy
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/embedding?format=json&version=v2&month=2025-04"
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/embedding?format=cache&version=v2&month=2025-04" -o /tmp/emb_patch_000000_cache.png
curl -s "http://60.31.21.42:22065/regions/haidian/patches/patch_000000/embedding?format=json&version=v1&month=202512"

# 4. 下游任务
curl -s "http://60.31.21.42:22065/regions/harbin/tasks"
curl -s "http://60.31.21.42:22065/regions/harbin/tasks/change_detection/summary?version=v1&period=2025-04_vs_2025-06"
curl -s "http://60.31.21.42:22065/regions/harbin/tasks/building_extraction/summary?version=v1"
curl -s "http://60.31.21.42:22065/regions/harbin/tasks/land_use_classification/summary?version=v1"

# 5. 单 Patch 任务结果 / 预测 / 标签
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/change_detection/result?format=png&version=v1&period=2025-04_vs_2025-06" -o /tmp/cd_patch_000000.png
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1" -o /tmp/be_patch_000000.png
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/land_use_classification/result?format=png&version=v1" -o /tmp/lu_patch_000000.png
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/building_extraction/prediction?version=v1" -o /tmp/be_pred_patch_000000.npy
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/building_extraction/label?version=v1"
curl -s "http://60.31.21.42:22065/regions/haidian/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1" -o /tmp/haidian_building.png
curl -s "http://60.31.21.42:22065/regions/haidian/patches/patch_000000/tasks/road_extraction/result?format=png&version=v1" -o /tmp/haidian_road.png
curl -s "http://60.31.21.42:22065/regions/haidian/patches/patch_000000/tasks/construction/result?format=png&version=v1" -o /tmp/haidian_construction.png

# 6. 瓦片
curl -s "http://60.31.21.42:22065/regions/harbin/tasks/change_detection/tiles?version=v1&period=2025-04_vs_2025-06"
curl -s "http://60.31.21.42:22065/regions/harbin/tasks/change_detection/tiles/12/6745/3201.png?version=v1&period=2025-04_vs_2025-06"

# 7. 自定义模型（先创建并训练完成，再使用返回的 model_id / job_id）
curl -s "http://60.31.21.42:22065/models"
curl -s -X POST "http://60.31.21.42:22065/models" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "我的建筑提取模型",
    "model_type": "single_time_detection",
    "region_id": "harbin",
    "embedding_version": "v2",
    "epochs": 20,
    "classes": [
      {"id": "cls_001", "name": "建筑用地", "color": "#FF0000"}
    ],
    "annotations": {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {
            "patch_id": "patch_000000",
            "region_id": "harbin",
            "class_id": "cls_001",
            "task_type": "building_extraction",
            "month": "2025-04"
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [[
              [126.505, 45.742],
              [126.515, 45.742],
              [126.515, 45.748],
              [126.505, 45.748],
              [126.505, 45.742]
            ]]
          }
        }
      ]
    }
  }'
curl -s "http://60.31.21.42:22065/models/model_ghi789"
curl -s -X PUT "http://60.31.21.42:22065/models/model_ghi789" \
  -H 'Content-Type: application/json' \
  -d '{"name": "my-building-head-v2"}'
curl -s -X DELETE "http://60.31.21.42:22065/models/model_ghi789"
curl -s -X POST "http://60.31.21.42:22065/models/model_ghi789/infer" \
  -H 'Content-Type: application/json' \
  -d '{"region_id": "harbin", "patch_id": "patch_000000", "month": "2025-04"}'
curl -s -X POST "http://60.31.21.42:22065/models/model_ghi789/infer_batch" \
  -H 'Content-Type: application/json' \
  -d '{
    "region_id": "harbin",
    "patch_ids": ["patch_000000", "patch_000001"],
    "month": "2025-04"
  }'
curl -s "http://60.31.21.42:22065/models/jobs/job_jkl012"
curl -s "http://60.31.21.42:22065/models/results/infer_model_xxx_harbin_patch_000000_2025-04.png"

# 8. 系统预训练模型
curl -s "http://60.31.21.42:22065/system-models?region_id=harbin"
curl -s "http://60.31.21.42:22065/system-models/land_cover_classification/classes?region_id=harbin&version=v2"
curl -s -X POST "http://60.31.21.42:22065/system-models/land_cover_classification/infer?region_id=harbin&patch_id=patch_000000&month=2025-04&version=v2"
curl -s "http://60.31.21.42:22065/system-models/results/land_cover_classification_harbin_patch_000000_2025-04.png"

# 9. SAM3 交互式分割
curl -s "http://60.31.21.42:22065/regions/harbin/sam3/status"
curl -s -X POST "http://60.31.21.42:22065/regions/harbin/sam3/embed" \
  -H 'Content-Type: application/json' \
  -d '{"patch_id": "patch_000000", "month": "2025-10", "sensor_type": "s2"}'
curl -s -X POST "http://60.31.21.42:22065/regions/harbin/sam3/segment" \
  -H 'Content-Type: application/json' \
  -d '{
    "date": "2025-10",
    "sensor_type": "s2",
    "point_coords": [[126.524, 45.750]],
    "point_labels": [1],
    "multimask_output": true,
    "include_masks": false
  }'
```

---

## 📖 阅读指南

本文档面向**前端开发团队**和**测试人员**，用通俗语言解释每一个接口的用途、参数和返回值。所有示例均已填充真实值，复制粘贴即可运行。如果你遇到不理解的术语，先看下面的「核心概念」章节。

---

## 🧠 核心概念

### 什么是 Patch？

Patch（图块）是我们把一整块大区域切分成的小方块。每个 Patch 对应地球上的一块矩形区域，有固定的经纬度范围。

- **哈尔滨新区**: 共 424 个 Patch，覆盖约 127km²
- **海淀区**: 共 320 个 Patch，覆盖约 80km²
- 每个 Patch 都有唯一 ID，如 `patch_000000`

**类比**: Patch 就像地图上的一个个「瓦片格子」，前端地图上的每个小方块可能对应一个 Patch。

### 什么是 Embedding（嵌入）？

Embedding 是深度学习模型把卫星影像「翻译」成的高维向量。可以理解为：模型「看懂」了这张图，并用一串数字记录了它的特征。

- **哈尔滨**: 嵌入以 **PNG 图片**形式存储（64×64 像素，RGB 三通道），便于直接展示；同时提供 `.npy` 原始数组
- **海淀**: 嵌入以 **NPZ/NPY 数组**形式存储（多通道 × 128×128），是原始数学向量

**类比**: 如果卫星影像是一篇文章，Embedding 就是这篇文章的「摘要」—— 浓缩了关键信息，但不可直接阅读。

### 什么是下游任务？

下游任务是在 Embedding 基础上做的「具体监测」。当前 API 已统一为 **5 个任务 ID**，无论你之前使用的是什么别名，现在都应该用这 5 个 ID：

| 任务英文名 | 中文名 | 监测内容 | 对应老任务（已废弃） |
|-----------|--------|---------|---------------------|
| `change_detection` | 变化检测 | 基于两期 embedding 差分检测变化区域 | `building_change`、`demolition` |
| `building_extraction` | 建筑物提取 | 提取建筑物、建筑工地 | `construction` |
| `road_extraction` | 道路提取 | 提取道路和路网 | 无 |
| `construction` | 施工地检测 | 海淀区施工地检测 | 无 |
| `construction_joint` | 施工地联合检测 | construction_joint 的海淀子集 | 无 |
| `land_use_classification` | 土地利用分类 | 耕地、建设用地等土地利用类型 | `farmland`、`land_conversion` |
| `land_cover_classification` | 土地覆盖分类 | WorldCover / Dynamic World 土地覆盖 | 无 |
| `water_extraction` | 水体提取 | JRC 水体提取 | 无 |

**版本说明**:
- **V1**: 单期监测，只有一个时间点的结果，需指定 `month`（如 `2025-04`）
- **V2**: 变化监测，对比两个时间点的差异，需指定 `period`（如 `2025-04_vs_2025-06`）

> ⚠️ **旧任务 ID 说明**：哈尔滨仍建议使用统一任务 ID，如 `building_extraction`、`land_use_classification`。海淀 V1 额外开放 `road_extraction`、`construction`、`construction_joint`，用于服务 P2A 海淀专题任务。

### 什么是瓦片（Tile）？

瓦片是切成小片的 PNG 图片，前端地图库（如 Leaflet、Mapbox）可以直接叠加到地图上显示监测结果。注意当前版本的 `/tiles/{z}/{x}/{y}.png` 返回 `501 Not Implemented`，请使用 `/tiles` 列表接口获取按 Patch 切分的结果图。

---

## 🔧 快速开始

### Base URL

**生产环境**:
```
http://60.31.21.42:22065
```

**本地开发**:
```
http://localhost:9061
```

### 在线文档

- **Swagger UI**（交互式调试）: `http://60.31.21.42:22065/docs`
- **ReDoc**（美观文档）: `http://60.31.21.42:22065/redoc`

> 默认情况下 Swagger / ReDoc 是关闭的，需要通过环境变量 `DOCS_URL=/docs` 和 `REDOC_URL=/redoc` 开启。

### 启动服务

```bash
# 方式一：直接运行
cd embedding-api
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 9061

# 方式二：Docker
docker-compose up -d
```

> **环境变量配置**（可选，生产环境建议设置）：
> - `CONFIG_PATH` — 指定配置文件路径，默认 `./config.yaml`
> - `CORS_ORIGINS` — 逗号分隔的允许跨域来源，为空则拒绝跨域请求
> - `DOCS_URL` / `REDOC_URL` — 在线文档路径
>
> **CORS**：服务默认 `allow_origins=[]`，前端浏览器调用前必须在服务端设置 `CORS_ORIGINS`。

---

## 📡 接口总览

| 分类 | 接口 | 方法 | 说明 |
|------|------|------|------|
| **基础** | `/health` | GET | 服务健康检查 |
| **区域** | `/regions` | GET | 列出所有区域 |
| **区域** | `/regions/{region_id}` | GET | 区域详情 |
| **Patch** | `/regions/{region_id}/patches` | GET | Patch 列表（支持分页+地理过滤） |
| **Patch** | `/regions/{region_id}/patches/{patch_id}` | GET | 单个 Patch 详情 |
| **嵌入** | `/regions/{region_id}/patches/{patch_id}/embedding` | GET | 查询嵌入数据 |
| **任务** | `/regions/{region_id}/tasks` | GET | 列出下游任务 |
| **任务** | `/regions/{region_id}/tasks/{task_type}/summary` | GET | 任务统计摘要 |
| **任务** | `/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result` | GET | 单 Patch 结果图 |
| **任务** | `/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/prediction` | GET | 原始预测数据 |
| **任务** | `/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/label` | GET | 标签数据 |
| **瓦片** | `/regions/{region_id}/tasks/{task_type}/tiles` | GET | 列出可用瓦片 |
| **瓦片** | `/regions/{region_id}/tasks/{task_type}/tiles/{z}/{x}/{y}.png` | GET | 标准 XYZ 瓦片（暂未实现） |
| **马赛克** | `/regions/{region_id}/mosaic` | GET | 整区域 S2/S1/Landsat 马赛克大图 |
| **自定义模型** | `/models` | GET/POST | 模型列表 / 创建训练 |
| **自定义模型** | `/models/{model_id}` | GET/PUT/PATCH/DELETE | 模型详情 / 重命名 / 删除 |
| **自定义模型** | `/models/{model_id}/infer` | POST | 单 Patch 推理 |
| **自定义模型** | `/models/{model_id}/infer_batch` | POST | 批量推理 |
| **自定义模型** | `/models/jobs/{job_id}` | GET | 训练任务状态 |
| **自定义模型** | `/models/results/{filename}` | GET | 下载推理结果图 |
| **系统模型** | `/system-models` | GET | 列出系统预训练模型 |
| **系统模型** | `/system-models/{task_id}/classes` | GET | 系统模型类别定义 |
| **系统模型** | `/system-models/{task_id}/infer` | POST | 系统模型单 Patch 推理 |
| **系统模型** | `/system-models/results/{filename}` | GET | 下载系统模型结果图 |
| **SAM3** | `/regions/{region_id}/sam3/status` | GET/POST | 状态查询 |
| **SAM3** | `/regions/{region_id}/sam3/embed` | POST | 预加载影像并计算 embedding |
| **SAM3** | `/regions/{region_id}/sam3/segment` | POST | 点选分割 |

---

## 🔍 详细接口

### 1. 健康检查

检查服务是否正常运行。

```
GET /health
```

**什么时候用**: 页面加载时检查后端可用性，或服务监控心跳。

**请求参数**: 无

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/health"
```

**成功响应** (200):
```json
{
  "status": "ok",
  "version": "0.1.0",
  "regions": ["harbin", "haidian"]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 服务状态，"ok" 表示正常 |
| `version` | string | API 版本号 |
| `regions` | string[] | 当前支持的区域列表 |

---

### 2. 列出所有区域

获取系统支持的所有监测区域，以及每个区域的基本信息。

```
GET /regions
```

**什么时候用**: 前端页面初始化时，加载区域选择下拉框。

**请求参数**: 无

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/regions"
```

**成功响应** (200):
```json
{
  "regions": [
    {
      "id": "harbin",
      "name": "哈尔滨新区",
      "patch_count": 424,
      "tasks": [
        "change_detection",
        "building_extraction",
        "land_use_classification",
        "land_cover_classification",
        "water_extraction"
      ]
    },
    {
      "id": "haidian",
      "name": "海淀区",
      "patch_count": 320,
      "tasks": [
        "change_detection",
        "building_extraction",
        "land_use_classification",
        "land_cover_classification",
        "water_extraction"
      ]
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 区域标识符，用于后续接口 |
| `name` | string | 区域中文名 |
| `patch_count` | int | 该区域包含的 Patch 总数 |
| `tasks` | string[] | 该区域支持的下游任务列表 |

**注意**: 海淀区 `v1` 使用玄女 P10C embedding，月份为 `202512` 至 `202605`。当前已部署建筑、道路、施工地、水体、土地利用分类和土地覆盖分类的月度结果。实际可用组合以 `GET /regions/haidian/tasks` 为准。

---

### 3. 获取区域详情

获取某个区域的详细信息，包括所有下游任务的版本信息。

```
GET /regions/{region_id}
```

**什么时候用**: 用户选择某个区域后，展示该区域的任务菜单。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID，如 `harbin` 或 `haidian` |

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/regions/harbin"
```

**成功响应** (200):
```json
{
  "id": "harbin",
  "name": "哈尔滨新区",
  "patch_count": 424,
  "tasks": {
    "change_detection": {
      "name": "变化检测",
      "description": "基于两期 embedding 差分的变化检测",
      "versions": ["v1"]
    },
    "building_extraction": {
      "name": "建筑物提取",
      "description": "建筑物提取与建筑工地监测（映射原 construction 任务）",
      "versions": ["v1", "v2"]
    },
    "land_use_classification": {
      "name": "土地利用分类",
      "description": "土地利用分类与转换监测（映射原 farmland / land_conversion 任务）",
      "versions": ["v1", "v2"]
    },
    "land_cover_classification": {
      "name": "土地覆盖分类",
      "description": "土地覆盖分类（基于预训练 Linear Probe，结果需实时推理）",
      "versions": []
    },
    "water_extraction": {
      "name": "水体提取",
      "description": "水体提取（基于预训练 Linear Probe，结果需实时推理）",
      "versions": []
    }
  },
  "embeddings": ["v1", "v2"]
}
```

**错误响应**:
- `404`: 区域不存在

---

### 4. 列出 Patch

分页获取某个区域的所有 Patch，支持**地理范围过滤**。

```
GET /regions/{region_id}/patches?page=1&page_size=20&bbox=minx,miny,maxx,maxy
```

**什么时候用**:
- 前端地图初始化时加载可见区域的 Patch
- 搜索框输入地理范围后筛选 Patch
- 分页展示 Patch 列表

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `page` | int | 否 | 1 | 页码，从 1 开始 |
| `page_size` | int | 否 | 20 | 每页数量，**最大 100**，超过将返回 422 错误 |
| `bbox` | string | 否 | - | 地理范围过滤，格式：`minx,miny,maxx,maxy`（WGS84 坐标系） |

**坐标顺序**：`minx,miny,maxx,maxy` 分别对应 **最小经度、最小纬度、最大经度、最大纬度**。

**curl 示例**:
```bash
# 不分页获取前 5 个 Patch
curl -s "http://60.31.21.42:22065/regions/harbin/patches?page=1&page_size=5"

# 使用 bbox 过滤哈尔滨新区中心区域
curl -s "http://60.31.21.42:22065/regions/harbin/patches?page=1&page_size=20&bbox=126.5,45.74,126.55,45.76"
```

**成功响应** (200):
```json
{
  "total": 424,
  "page": 1,
  "page_size": 20,
  "has_next": true,
  "patches": [
    {
      "patch_id": "patch_000000",
      "bounds_wgs84": [126.51631, 45.743707, 126.532242, 45.755574],
      "footprint_wgs84": {
        "type": "Polygon",
        "coordinates": [[
          [126.51631, 45.743707],
          [126.53275, 45.744064],
          [126.532242, 45.755574],
          [126.5158, 45.755216],
          [126.51631, 45.743707]
        ]]
      },
      "sources": {
        "dem": 1,
        "dynamic_world": 13,
        "jrc_water": 1,
        "landsat": 38,
        "modis_lst": 13,
        "modis_ndvi": 13,
        "s1": 100,
        "s1_hr": 4,
        "s2": 182,
        "s2_hr": 5,
        "worldcover": 1
      },
      "time_range": ["2023-01", "2025-10"],
      "has_embedding": true,
      "available_months": ["2025-04", "2025-06", "2025-08", "2025-09", "2025-10"],
      "available_tasks": ["change_detection", "building_extraction", "land_use_classification"]
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `total` | int | 符合条件的 Patch 总数 |
| `page` | int | 当前页码 |
| `page_size` | int | 每页数量 |
| `has_next` | bool | 是否有下一页 |
| `patch_id` | string | Patch 唯一标识 |
| `bounds_wgs84` | float[4] | Patch 的 WGS84 外接矩形 `[min_lng, min_lat, max_lng, max_lat]`，适合 bbox 查询和快速定位 |
| `footprint_wgs84` | GeoJSON Polygon | Patch 的真实 WGS84 四边形边界。前端绘制 Patch 边框时请优先使用它，避免相邻 patch 因投影转换出现缝隙或重叠 |
| `sources` | object | 各数据源包含的影像/样本数量 |
| `time_range` | string[2] | 数据时间范围 `[开始年月, 结束年月]`，格式为 `YYYY-MM` |
| `has_embedding` | bool | 是否有嵌入数据 |
| `available_months` | string[] | 该 Patch 有 Embedding 的月份列表 |
| `available_tasks` | string[] | 该 Patch 有结果的下游任务（前端可用此字段动态渲染「查看任务」按钮） |

**前端提示**: `bounds_wgs84` 是外接矩形，不是精确边界。哈尔滨、海淀的 patch 网格是在 UTM 投影坐标系中切出来的正方形，转换到 WGS84 后会变成轻微倾斜的四边形；如果用 `bounds_wgs84` 画经纬度矩形，相邻 patch 之间可能出现几十米级的视觉缝隙或重叠。地图上展示 patch 边界请使用 `footprint_wgs84`。前端可通过 `has_next` 判断是否需要显示「加载更多」按钮。

---

### 5. 获取 Patch 详情

获取单个 Patch 的完整信息。

```
GET /regions/{region_id}/patches/{patch_id}
```

**什么时候用**: 用户点击地图上的某个 Patch 时，展示该 Patch 的详细信息。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `patch_id` | string | 是 | Patch ID，如 `patch_000000` |

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000"
```

**成功响应**: 与「列出 Patch」中的单个 Patch 结构相同。

**错误响应**:
- `404`: Patch 不存在

---

### 6. 获取嵌入数据

获取某个 Patch 的 Embedding（嵌入向量）。支持四种返回格式。

#### 海淀 AEF 2025 PCA 可视化

```http
GET /regions/haidian/patches/{patch_id}/embeddings/aef/pca
```

只需填写海淀 Patch ID，例如 `patch_000106`。接口固定读取本地 AEF
2025 年年度 64 维 embedding，并返回 `image/png`。全部 Patch 使用同一套
全海淀 PCA 主成分和统一的 2%~98% 显示范围，因此不同 Patch 的颜色可以横向比较。

```bash
curl -o aef_pca.png \
  http://localhost:9061/regions/haidian/patches/patch_000106/embeddings/aef/pca
```

该接口不接收月份、年份、版本或输出格式参数，也不返回原始 `.npy`。

```
GET /regions/{region_id}/patches/{patch_id}/embedding?format=png
```

**什么时候用**:
- `format=png`: 前端展示嵌入的可视化热图
- `format=npy`: 下载原始数据用于进一步分析
- `format=json`: 快速查看嵌入的统计信息
- `format=cache`: 自动选择可用格式（优先 PNG）

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `patch_id` | string | 是 | Patch ID |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `format` | string | 否 | `png` | 输出格式：`png`、`npy`、`json`、`cache` |
| `version` | string | 否 | - | Embedding 版本：`v1`（V4 模型）/`v2`（V5 模型）；不传时依次尝试可用版本 |
| `month` | string | 否 | - | 时间序列月份，如 `2025-04`（哈尔滨）或 `20251201`（海淀）；不传时自动回退到该 Patch 第一个可用月份 |

#### format=png（默认）

返回 **PNG 图片**，Content-Type: `image/png`

- **哈尔滨**: 64×64 像素的 RGB 可视化图
- **海淀**: 如果有可视化图则返回，否则尝试返回多源数据合成图

同一区域和 embedding 版本的 PCA 预览使用统一的 PCA 基和全局 2%–98%
分位数范围。前端拼接多个 patch 时不要再对单张 PNG 分别做自动对比度或
颜色归一化，否则会重新引入明显的色彩接缝。PCA 图仅用于特征可视化，
模型推理仍使用原始 NPY embedding。

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/embedding?format=png&version=v2&month=2025-04" -o /tmp/emb_patch_000000.png
```

**前端用法**: 直接放在 `<img>` 标签里展示。
```html
<img src="http://60.31.21.42:22065/regions/harbin/patches/patch_000000/embedding?format=png&version=v2&month=2025-04" />
```

#### format=npy

返回 **NPY 二进制文件**，Content-Type: `application/octet-stream`

- **哈尔滨**: 如果存在 `.npy` 文件则返回
- **海淀**: 返回 NPZ 中解压出的 embedding 数组

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/embedding?format=npy&version=v2&month=2025-04" -o /tmp/emb_patch_000000.npy
```

**前端用法**: 需要在前端用 JavaScript 解析 numpy 格式（可用 `jsnumpy` 库），或下载后交给 Python 后端处理。

#### format=cache

**自动选择可用格式**，优先返回 PNG 图片，没有 PNG 则返回 NPY 二进制。

- **适用场景**：前端不确定某个 Patch 有哪些格式时，让后端自动选择
- **返回 Content-Type**：根据实际返回的数据决定（`image/png` 或 `application/octet-stream`）

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/embedding?format=cache&version=v2&month=2025-04" -o /tmp/emb_patch_000000_cache
```

```html
<!-- 自动选择最佳格式展示 -->
<img src="http://60.31.21.42:22065/regions/harbin/patches/patch_000000/embedding?format=cache&version=v2&month=2025-04" />
```

#### format=json

返回 **JSON 统计信息**，Content-Type: `application/json`

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/embedding?format=json&version=v2&month=2025-04"
```

**成功响应**:
```json
{
  "patch_id": "patch_000000",
  "shape": [64, 64, 3],
  "dtype": "uint8",
  "min": 0.0,
  "max": 255.0,
  "mean": 112.5
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `shape` | int[] | 数组维度。PNG 格式通常为 `[H, W, 3]`；NPY 格式因模型版本而异 |
| `dtype` | string | 数据类型，`float32` 或 `uint8` |
| `min` | float | 最小值 |
| `max` | float | 最大值 |
| `mean` | float | 平均值 |

**前端提示**: `shape` 的维度含义因模型架构不同而变化。Embedding 不是普通 RGB 图片，每个维度代表模型提取的一个特征分量。如需要具体语义，请联系后端团队获取模型说明。

---

### 7. 列出下游任务

获取某个区域支持的所有下游监测任务。

```
GET /regions/{region_id}/tasks
```

**什么时候用**: 前端侧边栏或下拉菜单展示可选的监测任务类型。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/regions/harbin/tasks"
```

**成功响应** (200):
```json
{
  "tasks": [
    {
      "id": "change_detection",
      "name": "变化检测",
      "description": "基于两期 embedding 差分的变化检测",
      "versions": ["v1"]
    },
    {
      "id": "building_extraction",
      "name": "建筑物提取",
      "description": "建筑物提取与建筑工地监测（映射原 construction 任务）",
      "versions": ["v1", "v2"]
    },
    {
      "id": "land_use_classification",
      "name": "土地利用分类",
      "description": "土地利用分类与转换监测（映射原 farmland / land_conversion 任务）",
      "versions": ["v1", "v2"]
    },
    {
      "id": "land_cover_classification",
      "name": "土地覆盖分类",
      "description": "土地覆盖分类（基于预训练 Linear Probe，结果需实时推理）",
      "versions": []
    },
    {
      "id": "water_extraction",
      "name": "水体提取",
      "description": "水体提取（基于预训练 Linear Probe，结果需实时推理）",
      "versions": []
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 任务标识符，用于后续接口 |
| `name` | string | 任务中文名 |
| `description` | string | 任务描述 |
| `versions` | string[] | 可用版本号列表 |

**版本说明**:
- `v1`: 基于单期数据的监测结果
- `v2`: 基于两期对比的变化检测结果，需要指定 `period` 参数

---

### 8. 任务统计摘要

获取任务资产覆盖、模型来源、可用月份、结果颜色含义、目标占比、异常警告和简短中文分析。响应适合前端展示，也适合智能体直接读取。

```
GET /regions/{region_id}/tasks/{task_type}/summary
```

**什么时候用**: 判断任务结果是否齐全、覆盖哪些月份、结果图中每种颜色表示什么，以及目标区域大约占多少。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `task_type` | string | 是 | 单期任务类型，如 `building_extraction`、`road_extraction`、`water_extraction` |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `version` | string | 否 | 按区域选择 | 海淀默认 P10C (`v1`)，哈尔滨默认 V5 (`v2`) |
| `month` | string | 否 | - | 单期影像月份，支持 `YYYYMM` 或 `YYYY-MM` |
| `patch_ids` | string[] | 否 | 全部 Patch | 可重复传入；每个 Patch 独立统计后汇总，最多 100 个 |

**curl 示例**:
```bash
# 海淀建筑任务，版本可省略
curl -s "http://60.31.21.42:22065/regions/haidian/tasks/building_extraction/summary?month=202512&patch_ids=patch_000000&patch_ids=patch_000001"

# 哈尔滨变化检测：同一批 Patch 分别比较各自的前后月份，再汇总
curl -s "http://60.31.21.42:22065/regions/harbin/change-detection/summary?before_month=202504&after_month=202506&patch_ids=patch_000000&patch_ids=patch_000001"
```

**成功响应** (200):
```json
{
  "schema_version": "2.0",
  "task": "building_extraction",
  "name": "建筑物提取",
  "region_id": "haidian",
  "version": "v1",
  "status": "ready",
  "analysis_scope": {
    "mode": "single_time",
    "patch_ids": ["patch_000000", "patch_000001"],
    "patch_count": 2,
    "month": "202512",
    "aggregation": "每个 Patch 独立推理后汇总统计"
  },
  "summary_text": "海淀区建筑物提取分析：月份为 202604，共分析 1 个 Patch，其中 1 个已有结果，覆盖率 100.0%。颜色说明：#FFFFFF 表示背景，#E60000 表示建筑物。",
  "model": {
    "foundation_model": "P10C",
    "feature_source": "P10C 64D embedding",
    "head_type": "binary_conv3x3"
  },
  "data_coverage": {
    "configured_patches": 320,
    "prediction_patches": 320,
    "result_tiles": 320,
    "label_patches": 320,
    "coverage_rate": 1.0
  },
  "color_legend": [
    {"color": "#FFFFFF", "name": "背景", "meaning": "结果图中该颜色表示背景。", "ratio": 0.42},
    {"color": "#E60000", "name": "建筑物", "meaning": "结果图中该颜色表示建筑物。", "ratio": 0.58}
  ],
  "image_analysis": {
    "image_count": 1,
    "total_pixels": 16384,
    "target_pixels": 9500,
    "target_ratio": 0.58,
    "images": [{"patch_id": "patch_000010", "width": 128, "height": 128, "total_pixels": 16384, "target_pixels": 9500, "target_ratio": 0.58}]
  },
  "result_images": [
    {"patch_id": "patch_000010", "image_url": "http://60.31.21.42:22065/task-summary/results/haidian_building_extraction_v1_patch_000010_202604.png", "cleanup_interval_seconds": 7200}
  ],
  "insights": [],
  "warnings": []
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `task` | string | 任务英文名 |
| `name` | string | 任务中文名 |
| `version` | string | 版本号 |
| `status` | string | `ready`、`partial` 或 `unavailable` |
| `summary_text` | string | 面向智能体的简短中文结果介绍 |
| `analysis_scope` | object | 本次月份、Patch 范围及“逐 Patch 独立处理后汇总”的口径 |
| `model` | object | 基座模型、Embedding 和下游头信息 |
| `temporal_coverage` | object | 可用月份和起止时间 |
| `data_coverage` | object | 预测、标签、结果瓦片和缺失 Patch 统计 |
| `prediction_statistics` | object | 阈值、目标像素占比、分位数或颜色分布 |
| `color_legend` | array | 每种结果颜色对应的类别、中文含义、像素数和占比 |
| `image_analysis` | object | 图片宽高、总像素、目标像素和目标占比，可包含逐 Patch 明细 |
| `result_images` | array | 逐 Patch 公网临时图片完整 URL；后台每两小时清空一次临时目录 |
| `insights` | array | 带证据的结构化分析结论 |
| `warnings` | array | 缺失资产、覆盖不足或质量风险 |

> 摘要关注结果内容和颜色语义，不输出 IoU、Precision、Recall 等评估指标。

> 变化检测使用独立接口 `GET /regions/{region_id}/change-detection/summary`。多个 Patch 之间不会互相比较；每个 Patch 只比较它自己的 `before_month` 与 `after_month`，最后再汇总。

> **注意**: 任务列表、预生成结果和实时系统模型是不同能力。前端应先读取 `GET /regions/{region_id}/tasks` 和 `GET /system-models?region_id=...`，不要根据旧的静态表推断可用性。海淀土地利用/覆盖分类 V1 已有 `2025-12` 至 `2026-05` 月度结果。

---

### 9. 单 Patch 任务结果

获取某个 Patch 在特定下游任务上的结果图。

```
GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result?format=png&version=v1&period=...
```

**什么时候用**: 用户点击某个 Patch，查看该 Patch 的监测结果可视化图。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `patch_id` | string | 是 | Patch ID |
| `task_type` | string | 是 | 任务类型 |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `format` | string | 否 | `png` | `png` 返回图片，`npy` 返回原始数组 |
| `version` | string | 否 | `v1` | 版本号 |
| `period` | string | 否 | - | 时间周期，如 `2025-04_vs_2025-06`；若已传 `period`，则优先使用 |
| `month` | string | 否 | - | 单期任务（建筑物提取、土地利用分类等）的月份，如 `2025-10` |
| `before_month` | string | 否 | - | 变化检测任务的起始月份 |
| `after_month` | string | 否 | - | 变化检测任务的结束月份 |

#### 各区域任务可用时间范围

时间格式统一为 `YYYY-MM`（如 `2025-04`、`2025-12`、`2026-05`），接口同时兼容 `YYYYMM` 写法。

| 区域 | 任务 | 版本 | 时间参数 | 可用时间范围 |
|------|------|------|----------|--------------|
| 哈尔滨 | `change_detection` | `v1` / `v2` | `period` 或 `before_month`+`after_month` | 2025-04_vs_2025-06、2025-04_vs_2025-10、2025-06_vs_2025-08、2025-06_vs_2025-10、2025-08_vs_2025-09、2025-08_vs_2025-10、2025-09_vs_2025-10 |
| 哈尔滨 | `building_extraction` | `v1` | `month` | 2025-04 ~ 2025-10（仅 2025-10 预生成，其余实时推理） |
| 哈尔滨 | `building_extraction` | `v2` | `period` 或 `before_month`+`after_month` | 2025-04_vs_2025-06、2025-08_vs_2025-09、2025-09_vs_2025-10 |
| 哈尔滨 | `road_extraction` | `v1` | `month` | 2025-04 ~ 2025-10 |
| 哈尔滨 | `land_use_classification` | `v1` | `month` | 2025-04 ~ 2025-10（仅 2025-10 预生成，其余实时推理） |
| 哈尔滨 | `land_use_classification` | `v2` | `period` 或 `before_month`+`after_month` | 2025-04_vs_2025-06、2025-08_vs_2025-09、2025-09_vs_2025-10 |
| 哈尔滨 | `land_cover_classification` | `v1` / `v2` | `month` | 2025-04 ~ 2025-10（实时推理） |
| 哈尔滨 | `water_extraction` | `v1` / `v2` | `month` | 2025-04 ~ 2025-10（实时推理） |
| 海淀 | `building_extraction` | `v1` | `month` | 2025-12 ~ 2026-05 |
| 海淀 | `road_extraction` | `v1` | `month` | 2025-12 ~ 2026-05 |
| 海淀 | `construction` | `v1` | `month` | 2025-12 ~ 2026-05 |
| 海淀 | `land_use_classification` | `v1` | `month` | 2025-12 ~ 2026-05 |
| 海淀 | `land_cover_classification` | `v1` | `month` | 2025-12 ~ 2026-05 |
| 海淀 | `water_extraction` | `v1` | `month` | 2025-12 ~ 2026-05 |

> **说明**:
> - 哈尔滨 `land_cover_classification`、`water_extraction` 没有预生成结果图，接口会根据 `month` 调用系统预训练模型实时推理；`building_extraction`、`land_use_classification` 的 `v1` 优先使用预生成结果，缺失月份同样会回退到系统模型推理。
> - 海淀区 `road_extraction`、`construction` 与 `building_extraction` 等任务使用相同的时间范围。

#### 海淀土地覆盖分类图例

海淀 `land_cover_classification` V1 结果集一共使用 **7 种颜色**。单个 Patch
可能只出现其中 6 种或更少，表示该 Patch 不包含其他地类，不是结果缺失。

| 项目类别值 | 颜色 | RGB | 中文含义 | 说明 |
|--------------|------|-----|----------|------|
| `1` | `#1E64DC` | `30, 100, 220` | 永久性水体 | 湖泊、水库、河流等长期有水的区域 |
| `2` | `#B4D250` | `180, 210, 80` | 灌木地 | 以灌木或低矮木本植被为主的区域 |
| `3` | `#F5DC5A` | `245, 220, 90` | 草地 | 以草本植被为主的区域 |
| `4` | `#D23C3C` | `210, 60, 60` | 耕地 | 农作物种植地、农田或周期性耕作区域 |
| `5` | `#BEAA82` | `190, 170, 130` | 建成区 | 建筑、道路及其他人工不透水表面为主的区域 |
| `6` | `#A0DCDC` | `160, 220, 220` | 裸地/稀疏植被 | 裸土、裸岩或植被覆盖度很低的区域 |
| `8` | `#006400` | `0, 100, 0` | 树木覆盖 | 林地、公园或其他以乔木树冠为主的区域 |

> **注意**：上表是海淀项目当前 PNG 结果的实际调色板，类别值为项目内部
> WorldCover 归一化值。它不是 ESA WorldCover 原始 11 类的完整调色板，前端不应
> 使用 PCA 嵌入图的颜色解释土地覆盖类别。

**curl 示例**:
```bash
# 单期结果（传 month）
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1&month=2025-10" -o /tmp/be_patch_000000.png
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/land_use_classification/result?format=png&version=v1&month=2025-10" -o /tmp/lu_patch_000000.png

# 变化检测结果（传 before_month / after_month）
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/change_detection/result?format=png&version=v1&before_month=2025-04&after_month=2025-06" -o /tmp/cd_patch_000000.png

# 也可以直接传 period
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/change_detection/result?format=png&version=v1&period=2025-04_vs_2025-06" -o /tmp/cd_patch_000000.png
```

**成功响应**:
- `format=png`: 返回 PNG 图片 (`image/png`)
- `format=npy`: 返回 NPY 文件 (`application/octet-stream`)

**前端用法**:
```html
<img src="http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1" />
```

**错误响应**:
- `404`: 该 Patch 在该任务下没有结果

> **海淀区 V1 示例**（mask-only 结果，已按月份归档）：
> 单期任务使用 `month` 参数指定月份（如 `2026-05`）。
> ```bash
> curl -s "http://60.31.21.42:22065/regions/haidian/patches/patch_000000/tasks/building_extraction/result?format=png&month=2026-05" -o /tmp/hd_be.png
> curl -s "http://60.31.21.42:22065/regions/haidian/patches/patch_000000/tasks/road_extraction/result?format=png&month=2026-05" -o /tmp/hd_re.png
> curl -s "http://60.31.21.42:22065/regions/haidian/patches/patch_000000/tasks/construction/result?format=png&month=2026-05" -o /tmp/hd_con.png
> curl -s "http://60.31.21.42:22065/regions/haidian/patches/patch_000000/tasks/water_extraction/result?format=png&month=2026-05" -o /tmp/hd_water.png
> ```
> 返回结果为 128×128 PNG，无卫星底图，仅保留前景掩膜。
> 所有任务结果均来自预生成结果或系统预训练模型实时推理。

---

### 10. 原始预测数据

下载某个 Patch 的原始预测数组（未经阈值化的概率图）。

```
GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/prediction?version=v1&period=...
```

**什么时候用**: 前端需要做自定义可视化（如调整阈值显示不同置信度区域），或下载给分析师进一步处理。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `patch_id` | string | 是 | Patch ID |
| `task_type` | string | 是 | 任务类型 |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `version` | string | 否 | `v1` | 版本号 |
| `period` | string | 否 | - | 时间周期（V2 任务需要） |

**curl 示例**:
```bash
# V1 单期预测
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/building_extraction/prediction?version=v1" -o /tmp/be_pred_patch_000000.npy
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/land_use_classification/prediction?version=v1" -o /tmp/lu_pred_patch_000000.npy

# V2 变化检测预测
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/change_detection/prediction?version=v1&period=2025-04_vs_2025-06" -o /tmp/cd_pred_patch_000000.npy
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/building_extraction/prediction?version=v2&period=2025-04_vs_2025-06" -o /tmp/be_v2_pred_patch_000000.npy
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/land_use_classification/prediction?version=v2&period=2025-04_vs_2025-06" -o /tmp/lu_v2_pred_patch_000000.npy
```

**返回**: NPY 二进制文件 (`application/octet-stream`)

**数据说明**: 预测值范围通常在 0~1 之间，表示模型认为该像素属于目标类别的概率。值越高，置信度越高。

**前端提示**: 可以用 JavaScript 的 `numpy-js` 或 `numjs` 库解析 NPY 文件。

---

### 11. 标签数据

获取某个 Patch 的真实标签（Ground Truth）。标签来源可能是离线标注或自动生成。

```
GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/label?version=v1&period=...
```

**什么时候用**: 对比「模型预测」和「真实标签」，计算准确率或生成混淆矩阵。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `patch_id` | string | 是 | Patch ID |
| `task_type` | string | 是 | 任务类型 |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `version` | string | 否 | `v1` | 版本号 |
| `period` | string | 否 | - | 时间周期（V2 任务需要） |

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/building_extraction/label?version=v1"
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_000000/tasks/change_detection/label?version=v1&period=2025-04_vs_2025-06"
```

**返回**:
- 如果存在 `.npy` 标签文件：返回 NPY 二进制数组 (`application/octet-stream`)
- 如果存在 `meta.json` 元数据文件：返回 JSON 元数据 (`application/json`)
- 如果两者都不存在：返回 `404`

**错误响应**:
- `404`: 标签或元数据不存在

---

### 12. 列出可用瓦片

获取某个任务的所有瓦片文件列表。

```
GET /regions/{region_id}/tasks/{task_type}/tiles?version=v1&period=...
```

**什么时候用**: 前端地图组件初始化时，预加载瓦片索引。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `task_type` | string | 是 | 任务类型 |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `version` | string | 否 | `v1` | 版本号 |
| `period` | string | 否 | - | 时间周期（V2 任务需要） |

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/regions/harbin/tasks/change_detection/tiles?version=v1&period=2025-04_vs_2025-06"
curl -s "http://60.31.21.42:22065/regions/harbin/tasks/building_extraction/tiles?version=v1"
curl -s "http://60.31.21.42:22065/regions/harbin/tasks/building_extraction/tiles?version=v2&period=2025-04_vs_2025-06"
curl -s "http://60.31.21.42:22065/regions/harbin/tasks/land_use_classification/tiles?version=v1"
```

**成功响应**:
```json
{
  "tiles": [
    {"patch_id": "patch_000000", "period": "2025-04_vs_2025-06", "filename": "patch_000000_2025-04_vs_2025-06.png"},
    {"patch_id": "patch_000001", "period": "2025-04_vs_2025-06", "filename": "patch_000001_2025-04_vs_2025-06.png"}
  ],
  "total": 2
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `tiles` | object[] | 瓦片信息列表 |
| `tiles[].patch_id` | string | Patch ID |
| `tiles[].period` | string | 时间段（V2 任务） |
| `tiles[].filename` | string | 瓦片文件名 |
| `total` | int | 瓦片总数 |

---

### 13. 获取地图瓦片

标准的 XYZ 瓦片接口，可被 Leaflet、Mapbox、OpenLayers 等地图库直接使用。

```
GET /regions/{region_id}/tasks/{task_type}/tiles/{z}/{x}/{y}.png?version=v1&period=...
```

**什么时候用**: 在地图上叠加监测结果图层。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `task_type` | string | 是 | 任务类型 |
| `z` | int | 是 | 缩放级别 (zoom level) |
| `x` | int | 是 | 瓦片 X 坐标 |
| `y` | int | 是 | 瓦片 Y 坐标 |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `version` | string | 否 | `v1` | 版本号 |
| `period` | string | 否 | - | 时间周期（V2 任务需要） |

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/regions/harbin/tasks/change_detection/tiles/12/6745/3201.png?version=v1&period=2025-04_vs_2025-06"
```

**前端用法** (Leaflet 示例):
```javascript
L.tileLayer(
  'http://60.31.21.42:22065/regions/harbin/tasks/change_detection/tiles/{z}/{x}/{y}.png?version=v1&period=2025-04_vs_2025-06',
  { opacity: 0.7 }
).addTo(map);
```

**注意**: 当前版本 XYZ 瓦片服务**暂未实现**，调用会返回 `501 Not Implemented`。如需查看结果图，请使用：
- `/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result?format=png`
- `/regions/{region_id}/tasks/{task_type}/tiles` 列表接口

---

## 🔐 认证

服务支持 **API-Key 用户隔离**。在 `config.yaml` 中配置：

```yaml
auth:
  type: "api_key"
  users:
    key_alice_xxx:
      user_id: "alice"
      name: "Alice"
```

请求时携带：

```bash
curl -H 'X-API-Key: key_alice_xxx' http://60.31.21.42:22065/models
# 或
curl -H 'Authorization: Bearer key_alice_xxx' http://60.31.21.42:22065/models
```

未配置 `auth` 时，所有请求使用默认用户 `default`。

受保护的接口包括：
- `/models/*`
- `/system-models/*`

---

## 🏷️ 自定义训练工作流

前端在本地管理用户标注；当用户确认训练时，一次性将 GeoJSON 标注包通过 `POST /models` 提交给后端，后端解析 GeoJSON、提取训练样本并启动训练。

流程：

1. 前端本地管理标注，组织为 GeoJSON FeatureCollection。
2. 调用 `POST /models` 提交模型配置与 GeoJSON 标注包。
3. 后端解析标注包并训练下游任务头。

---

## 🤖 自定义模型

### 14. 列出模型（自定义 + 系统预设）

列出当前用户训练好的模型。传入 `region_id` 后，会同时返回该区域可用的系统预训练模型。

```
GET /models
GET /models?region_id=harbin
```

**查询参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 否 | 区域 ID，传入后会合并系统预训练模型 |

**curl 示例**:
```bash
# 只列自定义模型
curl -s "http://60.31.21.42:22065/models"

# 同时列出系统预训练模型
curl -s "http://60.31.21.42:22065/models?region_id=harbin"
```

**成功响应** (200):
```json
[
  {
    "id": "model_ghi789",
    "name": "my-building-head",
    "type": "single_time_detection",
    "task_type": "building_extraction",
    "status": "completed",
    "created_at": "2026-06-26T10:00:00",
    "completed_at": "2026-06-26T10:05:00",
    "classes": [
      {"id": "cls_abc123", "name": "建筑物", "color": "#ff0000"}
    ],
    "accuracy": 0.92,
    "n_samples": 120,
    "model_path": "users/default/models/model_ghi789.pkl",
    "description": "基于用户标注的建筑提取模型",
    "message": null,
    "job_id": "job_jkl012",
    "source": "custom"
  },
  {
    "id": "building_extraction",
    "name": "建筑物提取",
    "type": "single_time_detection",
    "task_type": "building_extraction",
    "status": "ready",
    "created_at": "1970-01-01T00:00:00",
    "completed_at": "1970-01-01T00:00:00",
    "classes": [...],
    "accuracy": null,
    "n_samples": null,
    "model_path": null,
    "description": "OSM 建筑物提取",
    "message": null,
    "job_id": null,
    "source": "system",
    "versions": ["v2", "v1"]
  }
]
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 模型唯一 ID |
| `name` | string | 模型名称 |
| `type` | string | 模型类型：`single_time_detection` 或 `change_detection` |
| `task_type` | string | 任务类型 |
| `status` | string | 状态：`training` / `completed` / `failed` / `ready`（系统模型） |
| `created_at` | string | 创建时间 |
| `completed_at` | string | 完成时间 |
| `classes` | object[] | 模型类别列表 |
| `accuracy` | float | 训练集有效像素上的 F1 参考值；自定义 few-shot 模型样本少，主要用于判断训练是否收敛，不等同于大规模验证集精度 |
| `n_samples` | int | 实际参与训练的有效 Polygon 数量；MultiPolygon 按独立 Polygon 分别计数 |
| `model_path` | string | 模型文件路径。自定义模型当前保存为 PyTorch few-shot checkpoint，历史模型可能仍是 `.pkl` |
| `description` | string | 模型描述 |
| `message` | string | 失败原因或提示信息 |
| `job_id` | string | 关联的训练任务 ID |
| `source` | string | 模型来源：`custom`（用户训练）或 `system`（系统预训练） |
| `versions` | string[] | 系统模型可用 checkpoint 版本 |

---

### 15. 创建并训练模型

创建模型并启动异步训练任务。请求体中需要携带完整的 GeoJSON 标注包和类别定义，后端解析后提取训练样本。训练完成后才能调用推理接口。

当前自定义训练按类别分别统计有效 Polygon，并为每个有标注的类别训练独立二分类头：某类少于 10 个时使用 `PU + Query`，大于等于 10 个时使用 `binary_conv3x3`。同一个模型可以同时包含两种头；声明但没有 Polygon 标注的类别会被跳过。**Polygon 内部是对应类别的正样本；Polygon 外部只是未标注区域，不会整体当作负样本**。推理仍使用同一个模型 ID，后端自动运行全部类别头并合并结果，前端 API 不变。

选择 `traditional_ml` 时，后端使用 Sentinel-2 六个波段及四个光谱指数，并为每个有标注类别训练独立 Random Forest。选择 `dinov3_sat493m` 时，后端使用对应月份的 DINOv3-SAT493M 特征，并为每个有标注类别训练独立像素 MLP。两种方式都会将全部类别头绑定到同一个 `model_id`；后续单次或批量推理不需要再次提交训练方式和类别列表。

```
POST /models
```

**请求体**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 用户自定义模型名称 |
| `model_type` | string | 是 | 训练类型：`single_time_detection`（单时间检测）或 `change_detection`（双时相变化检测）。旧值 `classification` 仍兼容，会按 `single_time_detection` 处理 |
| `region_id` | string | 是 | 区域 ID，如 `harbin` 或 `haidian` |
| `embedding_version` | string | 否 | 嵌入版本，默认 `v2`。如果该区域没有请求的版本，后端会自动使用该区域可用版本；海淀当前使用 `v1` |
| `epochs` | int | 否 | 训练迭代次数，默认 `100`，范围 `1~1000`。服务端最多执行 `100` 轮以控制耗时 |
| `description` | string | 否 | 模型描述 |
| `annotations` | object | 是 | GeoJSON FeatureCollection，坐标为 WGS84，几何类型支持 `Polygon`、`MultiPolygon` |
| `classes` | object[] | 是 | 类别定义列表，每项包含 `id`、`name`、`color` |
| `class_ids` | string[] | 否 | 候选类别 ID。后端以标注包中实际出现的 `class_id` 为准，有标注的类别分别训练，无标注类别自动跳过 |

**`annotations` 说明**：
- `type` 固定为 `FeatureCollection`。
- 每个 `Feature` 的 `properties` 必须包含 `patch_id`、`region_id`、`class_id`。
- 顶层请求体不再需要 `task_type`；`Feature.properties.task_type` 也可不传。
- 如果标注里传了 `properties.task_type`，后端会从所有 Feature 自动推导，且要求同一个训练包内任务类型一致。
- 如果标注里不传 `properties.task_type`，`single_time_detection` 默认按 `building_extraction` 训练，`change_detection` 默认按 `change_detection` 训练。
- `single_time_detection` 单时间检测需要 `month`；`change_detection` 双时相变化检测需要 `before_month` 和 `after_month`。
- `geometry` 坐标使用 WGS84 `[lon, lat]`。
- 所有 `Feature` 的 `region_id` 必须与请求体顶层 `region_id` 一致。
- 所有 `Feature` 的 `class_id` 必须在 `classes` 中定义。`class_ids` 是候选列表；实际训练以标注中出现的类别为准，没有 Polygon 标注的类别会被跳过。
- 训练时只把 Polygon 覆盖区域作为该类别正样本；未覆盖区域不代表“不是这个类别”。因此前端不需要为了 few-shot 训练额外画满背景。
- 限制：最多 `10000` 个 Feature，总顶点数不超过 `100000`。

**curl 示例 — 单时间检测模型**:
```bash
curl -s -X POST "http://60.31.21.42:22065/models" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "我的建筑提取模型",
    "model_type": "single_time_detection",
    "region_id": "harbin",
    "embedding_version": "v2",
    "epochs": 20,
    "classes": [
      {"id": "cls_001", "name": "建筑用地", "color": "#FF0000"}
    ],
    "annotations": {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {
            "patch_id": "patch_000000",
            "region_id": "harbin",
            "class_id": "cls_001",
            "task_type": "building_extraction",
            "month": "2025-04"
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [[
              [126.51631, 45.743707],
              [126.532242, 45.743707],
              [126.532242, 45.755574],
              [126.51631, 45.755574],
              [126.51631, 45.743707]
            ]]
          }
        }
      ]
    }
  }'
```

**curl 示例 — 变化检测模型**:
```bash
curl -s -X POST "http://60.31.21.42:22065/models" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "我的变化检测模型",
    "model_type": "change_detection",
    "region_id": "harbin",
    "embedding_version": "v2",
    "epochs": 30,
    "classes": [
      {"id": "cls_002", "name": "新增建筑", "color": "#00FF00"}
    ],
    "annotations": {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {
            "patch_id": "patch_000000",
            "region_id": "harbin",
            "class_id": "cls_002",
            "task_type": "change_detection",
            "before_month": "2025-04",
            "after_month": "2025-06"
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [[
              [126.51631, 45.743707],
              [126.532242, 45.743707],
              [126.532242, 45.755574],
              [126.51631, 45.755574],
              [126.51631, 45.743707]
            ]]
          }
        }
      ]
    }
  }'
```

**成功响应** (200):
```json
{
  "id": "model_ghi789",
  "name": "我的建筑提取模型",
  "type": "single_time_detection",
  "task_type": "building_extraction",
  "status": "training",
  "created_at": "2026-06-26T10:00:00",
  "completed_at": null,
  "classes": [
    {"id": "cls_001", "name": "建筑用地", "color": "#FF0000"}
  ],
  "accuracy": null,
  "n_samples": null,
  "model_path": null,
  "description": "基于用户标注的建筑提取模型",
  "message": null,
  "job_id": "job_jkl012"
}
```

---

### 16. 获取模型详情

获取单个模型（自定义或系统预设）的详情。

```
GET /models/{model_id}
GET /models/building_extraction?region_id=harbin
```

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_id` | string | 是 | 自定义模型 ID 或系统预训练任务 ID（`building_extraction` / `land_cover_classification` / `water_extraction`） |

**查询参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 条件 | `model_id` 为系统任务 ID 时必填 |
| `version` | string | 否 | 系统模型 checkpoint 版本；不传时后端按区域自动选择可用版本（哈尔滨通常为 `v2`，海淀为 `v1`） |

**curl 示例**:
```bash
# 自定义模型
curl -s "http://60.31.21.42:22065/models/model_ghi789"

# 系统预训练模型
curl -s "http://60.31.21.42:22065/models/building_extraction?region_id=harbin"
```

**成功响应** (200): 与列出模型中的单个模型结构相同。

**错误响应**:
- `404`: 模型不存在
- `422`: 系统模型缺少 `region_id`

---

### 17. 重命名模型

修改模型名称。

```
PUT /models/{model_id}
```

> 前端推荐使用 `PUT`。旧版 `PATCH /models/{model_id}` 仍保留兼容。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_id` | string | 是 | 模型 ID |

**请求体**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 新名称 |

**curl 示例**:
```bash
curl -s -X PUT "http://60.31.21.42:22065/models/model_ghi789" \
  -H 'Content-Type: application/json' \
  -d '{"name": "my-building-head-v2"}'
```

**成功响应** (200):
```json
{
  "status": "ok"
}
```

---

### 18. 删除模型

删除模型及其关联文件。

```
DELETE /models/{model_id}
```

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_id` | string | 是 | 模型 ID |

**curl 示例**:
```bash
curl -s -X DELETE "http://60.31.21.42:22065/models/model_ghi789"
```

**成功响应** (200):
```json
{
  "status": "ok"
}
```

---

### 19. 单 Patch 推理

使用训练完成的自定义模型，或系统预训练模型，对单个 Patch 进行推理。

```
POST /models/{model_id}/infer
```

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_id` | string | 是 | 自定义模型 ID 或系统预训练任务 ID（`building_extraction` / `land_cover_classification` / `water_extraction`） |

**请求体**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `patch_id` | string | 是 | Patch ID |
| `month` | string | 条件 | 单期任务必填，如 `2025-04` |
| `before_month` | string | 条件 | 变化检测必填，如 `2025-04` |
| `after_month` | string | 条件 | 变化检测必填，如 `2025-06` |
| `version` | string | 否 | 系统模型 checkpoint 版本；不传时后端按区域自动选择可用版本。自定义模型忽略 |

**curl 示例 — 自定义单时间检测模型**:
```bash
curl -s -X POST "http://60.31.21.42:22065/models/model_ghi789/infer" \
  -H 'Content-Type: application/json' \
  -d '{
    "region_id": "harbin",
    "patch_id": "patch_000000",
    "month": "2025-04"
  }'
```

**curl 示例 — 变化检测模型**:
```bash
curl -s -X POST "http://60.31.21.42:22065/models/model_ghi789/infer" \
  -H 'Content-Type: application/json' \
  -d '{
    "region_id": "harbin",
    "patch_id": "patch_000000",
    "before_month": "2025-04",
    "after_month": "2025-06"
  }'
```

**curl 示例 — 系统预训练模型**:
```bash
curl -s -X POST "http://60.31.21.42:22065/models/building_extraction/infer" \
  -H 'Content-Type: application/json' \
  -d '{
    "region_id": "harbin",
    "patch_id": "patch_000000",
    "month": "2025-04",
    "version": "v2"
  }'
```

**成功响应** (200):
```json
{
  "result_url": "/models/results/infer_model_ghi789_harbin_patch_000000_2025-04.png"
}
```

> 系统模型的 `result_url` 路径为 `/system-models/results/{filename}`，请用该路径下载图片。

**错误响应**:
- `400`: 自定义模型未训练完成或不存在
- `422`: 系统模型缺少 `month` 或请求参数错误

---

### 20. 批量推理

对最多 100 个 Patch 批量推理。支持自定义模型和系统预训练模型。

自定义模型的 `model_id` 已绑定训练时使用的基座模型版本、特征来源、预处理、
输入维度和下游头。前端不需要再次提交 `training_method` 或 `head_type`；后端会
根据 checkpoint 自动分派玄女、AEF、DINOv3-SAT493M 或传统 Sentinel-2 推理
流程，并拒绝区域或模型绑定不一致的调用。

```
POST /models/{model_id}/infer_batch
```

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_id` | string | 是 | 自定义模型 ID 或系统预训练任务 ID |

**请求体**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `patch_ids` | string[] | 是 | Patch ID 列表，最多 100 个 |
| `month` | string | 条件 | 单期任务必填，如 `2025-04` |
| `before_month` | string | 条件 | 变化检测必填，如 `2025-04` |
| `after_month` | string | 条件 | 变化检测必填，如 `2025-06` |
| `version` | string | 否 | 系统模型 checkpoint 版本；不传时后端按区域自动选择可用版本。自定义模型忽略 |

**curl 示例 — 自定义单时间检测模型**:
```bash
curl -s -X POST "http://60.31.21.42:22065/models/model_ghi789/infer_batch" \
  -H 'Content-Type: application/json' \
  -d '{
    "region_id": "harbin",
    "patch_ids": ["patch_000000", "patch_000001"],
    "month": "2025-04"
  }'
```

**curl 示例 — 变化检测模型**:
```bash
curl -s -X POST "http://60.31.21.42:22065/models/model_ghi789/infer_batch" \
  -H 'Content-Type: application/json' \
  -d '{
    "region_id": "harbin",
    "patch_ids": ["patch_000000", "patch_000001"],
    "before_month": "2025-04",
    "after_month": "2025-06"
  }'
```

**curl 示例 — 系统预训练模型**:
```bash
curl -s -X POST "http://60.31.21.42:22065/models/building_extraction/infer_batch" \
  -H 'Content-Type: application/json' \
  -d '{
    "region_id": "harbin",
    "patch_ids": ["patch_000000", "patch_000001"],
    "month": "2025-04",
    "version": "v2"
  }'
```

**成功响应** (200):
```json
{
  "total": 2,
  "success_count": 2,
  "error_count": 0,
  "results": [
    {
      "patch_id": "patch_000000",
      "status": "success",
      "result_url": "/models/results/infer_model_ghi789_harbin_patch_000000_2025-04.png",
      "error": null
    },
    {
      "patch_id": "patch_000001",
      "status": "success",
      "result_url": "/models/results/infer_model_ghi789_harbin_patch_000001_2025-04.png",
      "error": null
    }
  ]
}
```

> 系统模型的 `result_url` 路径为 `/system-models/results/{filename}`。

| 字段 | 类型 | 说明 |
|------|------|------|
| `total` | int | 请求的 Patch 数量 |
| `success_count` | int | 成功数量 |
| `error_count` | int | 失败数量 |
| `results[].patch_id` | string | Patch ID |
| `results[].status` | string | `success` 或 `error` |
| `results[].result_url` | string | 结果图路径 |
| `results[].error` | string | 错误信息 |

---

### 21. 查询训练任务状态

获取异步训练任务的状态。

```
GET /models/jobs/{job_id}
```

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `job_id` | string | 是 | 任务 ID |

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/models/jobs/job_jkl012"
```

**成功响应** (200):
```json
{
  "job_id": "job_jkl012",
  "status": "completed",
  "model_id": "model_ghi789",
  "accuracy": 0.92,
  "n_samples": 120,
  "model_path": "users/default/models/model_ghi789.pkl",
  "message": null
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `job_id` | string | 任务 ID |
| `status` | string | `running` / `completed` / `failed` |
| `model_id` | string | 关联模型 ID |
| `accuracy` | float | 训练准确率 |
| `n_samples` | int | 实际参与训练的有效 Polygon 数量；MultiPolygon 按独立 Polygon 分别计数 |
| `model_path` | string | 模型文件路径 |
| `message` | string | 失败原因或提示 |

---

### 22. 下载推理结果图

根据文件名下载自定义模型的推理结果 PNG。

```
GET /models/results/{filename}
```

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `filename` | string | 是 | 结果文件名 |

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/models/results/infer_model_ghi789_harbin_patch_000000_2025-04.png" -o /tmp/infer_result.png
```

**成功响应** (200): 返回 PNG 图片 (`image/png`)，尺寸统一为 **128×128** 像素。

---

## 🛰 区域马赛克大图

### 23. 获取整区域马赛克大图

将某个区域、某个月份下所有 Patch 按地理范围拼接成一张大图。既可以返回原始卫星影像，也可以返回 Embedding PCA 色彩可视化。

```
GET /regions/{region_id}/mosaic?date=YYYY-MM&sensor_type=s2&format=png
```

**路径参数**:

| 参数 | 类型 | 必填 | 可取值 | 说明 |
|------|------|------|--------|------|
| `region_id` | string | 是 | `harbin` / `haidian` | 区域 ID |

**查询参数**:

| 参数 | 类型 | 必填 | 可取值 | 默认值 | 说明 |
|------|------|------|--------|--------|------|
| `date` | string | 是 | `YYYY-MM` / `YYYYMM` / `YYYYMMDD` | - | 日期/月。两个区域均支持带横杠和不带横杠的月份；月度请求按日期倒序选择当月最新一景 |
| `sensor_type` | string | 否 | `s2` / `s1` / `landsat` / `highres` / `embedding` | `s2` | 大图数据源；`embedding` 返回 PCA 色彩图 |
| `version` | string | 否 | `v1` / `v2` | 按区域自动选择 | 仅 Embedding 大图使用；海淀默认 P10C (`v1`)，哈尔滨默认 V5 (`v2`) |
| `format` | string | 否 | `png` / `tif` | `png` | 输出格式：`png` 可视化；`tif` GeoTIFF 原始数据 |
| `patch_ids` | string[] | 否 | 如 `patch_000000`、`patch_000001` | 全区域 | 只拼接指定 Patch ID 列表，可多次传入 |

> **多景影像选择规则**：如果同一个 patch、同一个传感器、同一个月下有多张日级 TIFF，
> 后端按文件名日期倒序排列，固定选择最新的一景。
> 如果前端要指定某一天，请传 `YYYYMMDD`，例如 `20251214`，此时必须精确命中该日期，
> 不会自动改用其它日期。

**波段合成规则**：

| 传感器 | PNG 合成 | 说明 |
|--------|----------|------|
| `s2` | B4(R) / B3(G) / B2(B) | Sentinel-2 真彩色 |
| `landsat` | B4(R) / B3(G) / B2(B) | Landsat 真彩色 |
| `s1` | R=VV, G=VH, B=VH/VV | Sentinel-1 伪彩色合成 |
| `embedding` | 全局 PCA 色彩投影 | 使用该区域默认 Embedding 版本的逐月 PNG |

**完整请求示例**：

```bash
# 1. 哈尔滨全区域 Sentinel-2 真彩色 PNG（最大图，首次较慢）
curl -s "http://60.31.21.42:22065/regions/harbin/mosaic?date=2025-04&sensor_type=s2&format=png" -o /tmp/harbin_s2_2025-04.png

# 2. 只拼前两个 patch 的 Sentinel-1 伪彩色预览（快）
curl -s "http://60.31.21.42:22065/regions/harbin/mosaic?date=2025-04&sensor_type=s1&format=png&patch_ids=patch_000000&patch_ids=patch_000001" -o /tmp/harbin_s1_preview.png

# 3. Landsat 全区域 GeoTIFF 原始数据（保留多波段与坐标）
curl -s "http://60.31.21.42:22065/regions/harbin/mosaic?date=2025-04&sensor_type=landsat&format=tif" -o /tmp/harbin_landsat_2025-04.tif

# 4. 本地调试地址（服务重启后默认端口 9061）
curl -s "http://localhost:9061/regions/harbin/mosaic?date=2025-04&sensor_type=s2&format=png&patch_ids=patch_000000&patch_ids=patch_000001" -o /tmp/harbin_s2_preview.png

# 5. 海淀 P10C Embedding PCA 大图（version 可省略）
curl -s "http://localhost:9061/regions/haidian/mosaic?date=202512&sensor_type=embedding&format=png" -o /tmp/haidian_embedding_202512.png
```

**成功响应** (200): 返回 PNG 或 GeoTIFF 图片。

> 首次生成后会缓存到 `users/default/mosaic/{region_id}_{sensor_type}_{date}.{format}`，后续直接读取。
> 默认按 Patch 文件名排序后拼接，因此多次请求同一组 `patch_ids` 会命中缓存。

---

## 🧩 系统预训练模型

### 23. 列出系统预训练模型

列出某个区域可用的系统预训练模型。

```
GET /system-models?region_id=harbin
```

**查询参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/system-models?region_id=harbin"
```

**成功响应** (200):
```json
[
  {
    "id": "land_cover_classification",
    "name": "土地覆盖分类",
    "description": "WorldCover / Dynamic World 土地覆盖分类",
    "versions": ["v1", "v2"]
  },
  {
    "id": "water_extraction",
    "name": "水体提取",
    "description": "JRC Global Surface Water 水体提取",
    "versions": ["v1", "v2"]
  },
  {
    "id": "building_extraction",
    "name": "建筑物提取",
    "description": "OSM 建筑物提取",
    "versions": ["v1", "v2"]
  }
]
```

---

### 24. 获取系统模型类别定义

获取系统预训练模型的类别定义（颜色、名称等）。

```
GET /system-models/{task_id}/classes?region_id=harbin&version=v2
```

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | 是 | 任务 ID，如 `land_cover_classification` |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `region_id` | string | 是 | - | 区域 ID |
| `version` | string | 否 | 自动选择 | 模型版本。不传时按区域选择可用版本：哈尔滨优先 `v2`，海淀使用 `v1` |

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/system-models/land_cover_classification/classes?region_id=harbin&version=v2"
```

**成功响应** (200):
```json
[
  {"id": "sys_land_cover_classification_0", "name": "No data", "color": "#000000"},
  {"id": "sys_land_cover_classification_1", "name": "Tree cover", "color": "#006400"},
  {"id": "sys_land_cover_classification_2", "name": "Shrubland", "color": "#ffbb22"}
]
```

---

### 25. 系统模型单 Patch 推理

使用系统预训练模型对单个 Patch 实时推理。

```
POST /system-models/{task_id}/infer?region_id=harbin&patch_id=patch_000000&month=2025-04&version=v2
```

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | 是 | 任务 ID |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `region_id` | string | 是 | - | 区域 ID |
| `patch_id` | string | 是 | - | Patch ID |
| `month` | string | 是 | - | 月份 |
| `version` | string | 否 | 自动选择 | 模型版本。不传时按区域选择可用版本：哈尔滨优先 `v2`，海淀使用 `v1` |

**curl 示例**:
```bash
curl -s -X POST "http://60.31.21.42:22065/system-models/land_cover_classification/infer?region_id=harbin&patch_id=patch_000000&month=2025-04&version=v2"
```

**成功响应** (200):
```json
{
  "result_url": "/system-models/results/land_cover_classification_harbin_patch_000000_2025-04.png"
}
```

---

### 26. 下载系统模型结果图

根据文件名下载系统模型推理结果 PNG。

```
GET /system-models/results/{filename}
```

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `filename` | string | 是 | 结果文件名 |

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/system-models/results/land_cover_classification_harbin_patch_000000_2025-04.png" -o /tmp/sys_lc_result.png
```

**成功响应** (200): 返回 PNG 图片 (`image/png`)

---

## 🎯 SAM3 交互式分割

SAM3（Segment Anything with Concepts）是基于 Meta AI 的交互式实例分割模型。前端用户可以在卫星影像上点击点坐标，实时获取分割掩码。

### 27. SAM3 状态查询

查询 SAM3 模型加载状态和 GPU 内存使用情况。

```
GET /regions/{region_id}/sam3/status
```

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |

**curl 示例**:
```bash
curl -s "http://60.31.21.42:22065/regions/harbin/sam3/status"
```

**成功响应** (200):
```json
{
  "model_loaded": true,
  "device": "cuda:0",
  "gpu_memory": {
    "allocated_mb": 3842,
    "reserved_mb": 4096
  },
  "cache": {
    "size": 3,
    "max_size": 20,
    "entries": ["harbin_patch_000000_2025-10"]
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_loaded` | bool | 模型是否已加载到 GPU |
| `device` | string | 当前使用的设备（`cuda:N` 或 `cpu`，未加载时为 `not_loaded`） |
| `gpu_memory.allocated_mb` | int | 已分配的 GPU 显存（MB） |
| `gpu_memory.reserved_mb` | int | 预留的 GPU 显存（MB） |
| `cache.size` | int | 当前缓存的 embedding 数量 |
| `cache.max_size` | int | 缓存上限（默认 20） |
| `cache.entries` | string[] | 缓存的 embedding_id 列表 |

**错误响应**:
- `404`: 区域不存在

**缓存淘汰语义**:
嵌入缓存采用 LRU（最近最少使用）策略。当 `cache.size` 达到 `max_size` 后，新的嵌入请求会自动淘汰最旧的缓存项，并释放对应的 GPU 张量。缓存项在服务器重启后全部清空。

---

### 28. SAM3 Embed — 预加载影像

预加载 patch 的遥感影像，计算 SAM3 embedding，返回影像和 embedding_id。

```
POST /regions/{region_id}/sam3/embed
```

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |

**请求体**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `patch_id` | string | 是 | Patch ID |
| `month` | string | 是 | 日期/月，如 `2025-10`、`202510`、`20251214`。`YYYYMMDD` 精确到某一天；月级请求会在同月多景中按日期倒序取最新一景 |
| `sensor_type` | string | 否 | 传感器类型：`s2`、`s1`、`landsat`、`highres`，默认 `s2`；`highres` 表示高分辨率 RGB 光学 GeoTIFF |

**curl 示例**:
```bash
curl -s -X POST "http://60.31.21.42:22065/regions/harbin/sam3/embed" \
  -H 'Content-Type: application/json' \
  -d '{
    "patch_id": "patch_000000",
    "month": "2025-10",
    "sensor_type": "s2"
  }'
```

**成功响应** (200):
```json
{
  "embedding_id": "harbin_patch_000000_s2_202510",
  "status": "ready",
  "source_scene": "20251015",
  "selected_image_date": "20251015",
  "image": {
    "width": 256,
    "height": 256,
    "format": "png",
    "data": "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEAC..."
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `embedding_id` | string | 缓存标识符，格式包含 region、patch、sensor 和日期 |
| `status` | string | 状态，通常为 `ready` |
| `source_scene` | string | 实际加载的原始影像文件 stem，例如 `20251015` |
| `selected_image_date` | string | 实际选中的影像日期，前端可展示“本次使用影像：20251015” |
| `image.width` | int | 影像宽度 |
| `image.height` | int | 影像高度 |
| `image.format` | string | 图片格式 |
| `image.data` | string | 遥感影像可视化后的 base64 PNG |

**错误码**:
- `400`: 请求参数错误（如非法 patch_id、路径穿越尝试）
- `404`: Patch 或所选传感器影像不存在
- `503`: GPU 内存不足或模型加载失败

**前端提示**: `image.data` 可直接解码为 `<img>` 标签显示，让用户在影像上点击。

---

### 29. SAM3 Segment — 点选分割

前端传入 WGS84 标注点坐标和传感器类型。服务端自动定位 patch、加载影像、
计算或复用 SAM3 embedding，并返回 WGS84 GeoJSON 标注框。

```
POST /regions/{region_id}/sam3/segment
```

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |

**请求体**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `date` | string | 是 | 影像日期/月，用于选择要分割的遥感影像。支持 `2025-10`、`202510`、`20251001`。`YYYYMMDD` 精确到某一天；月级请求会在同月多景中按日期倒序取最新一景 |
| `sensor_type` | string | 否 | `s2`=Sentinel-2，`s1`=Sentinel-1，`landsat`=Landsat，`highres`=高分辨率 RGB 光学 GeoTIFF；默认 `s2` |
| `point_coords` | float[][] | 是 | 用户点击的 WGS84 经纬度点列表，每个点为 `[longitude, latitude]`，即 `[经度, 纬度]` |
| `point_labels` | int[] | 否 | 可选点标签。`1`=前景目标点，`0`=背景排除点。当前前端不用传；不传时后端默认所有点都是 `1` |
| `multimask_output` | bool | 否 | 是否返回多个候选结果。`false`=只返回一个最优候选，适合常规交互；`true`=返回多个候选供用户二次选择。默认 `false` |
| `include_masks` | bool | 否 | 是否在 GeoJSON 标注框之外额外返回 base64 PNG mask。默认 `false`，响应更小 |

#### 高分辨率光学影像约定

`highres` 不是特定卫星品牌，而是通用的高分辨率 RGB 光学数据源。部署前需按
以下任一目录结构放置已经切到对应 patch 的 GeoTIFF：

```text
/workspace/data/raw/{region_id}/highres/{patch_id}/{YYYYMMDD}.tif
{region.s2_dir}/{patch_id}/highres/{YYYYMMDD}.tif
{region.highres_dir}/highres_optical_{YYYYMMDD}_{patch_id}.tif
```

文件必须包含 CRS 和仿射变换，至少 3 个波段，前三个波段依次为 R、G、B。
接口保持原始长宽比；最长边超过 1024 像素时缩至 1024 再送入 SAM3。月级
请求仍按文件日期倒序选择当月最新一景。当前接口不负责把任意整幅大图上传后
自动切 patch；这类数据需要先完成切片和地理配准。

**curl 示例**:
```bash
curl -s -X POST "http://60.31.21.42:22065/regions/harbin/sam3/segment" \
  -H 'Content-Type: application/json' \
  -d '{
    "date": "2025-10",
    "sensor_type": "s2",
    "point_coords": [[126.524, 45.750]],
    "multimask_output": false,
    "include_masks": false
  }'
```

**成功响应** (200):
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[126.5231, 45.7501], [126.5244, 45.7501], [126.5244, 45.7510], [126.5231, 45.7510], [126.5231, 45.7501]]]
      },
      "properties": {
        "score": 0.95,
        "bbox": [120, 110, 35, 42],
        "bbox_wgs84": [126.5231, 45.7501, 126.5244, 45.7510],
        "patch_id": "patch_000000",
        "sensor_type": "s2",
        "date": "2025-10",
        "source_scene": "20251015",
        "selected_image_date": "20251015",
        "candidate_index": 0
      }
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `FeatureCollection` |
| `features[].geometry` | object | WGS84 GeoJSON Polygon，可直接作为标注框 |
| `features[].properties.score` | float | 模型置信度 `[0, 1]` |
| `features[].properties.bbox` | int[] | SAM 输入图像上的像素框 `[x, y, width, height]` |
| `features[].properties.bbox_wgs84` | float[] | WGS84 bbox `[min_lon, min_lat, max_lon, max_lat]` |
| `features[].properties.source_scene` | string | 实际参与 SAM3 推理的原始影像文件 stem，例如 `20251015` |
| `features[].properties.selected_image_date` | string | 实际选中的影像日期，用于解释月度请求最终使用哪一景 |
| `masks` | array | 仅当 `include_masks=true` 时返回 base64 PNG 掩码 |

**错误码**:
- `400`: 点不在同一个 patch、日期格式非法或源影像无 CRS
- `404`: 点不在区域覆盖范围内，或指定日期/传感器影像不存在
- `422`: 坐标格式错误或经纬度越界
- `503`: GPU 推理失败

**前端提示**:
- `/sam3/segment` 已取消 `month` 字段，只使用 `date`。
- 如果一个月内存在多景影像，后端按日期倒序固定选择当月最新一景；需要指定某一天时传 `YYYYMMDD`。
- 返回的 `selected_image_date` / `source_scene` 是本次实际使用的影像，前端可以直接展示给用户或写入调试日志。
- 前端一般不需要传 `point_labels`；后端会把所有 `point_coords` 默认当成前景目标点 `1`。
- 掩码 PNG 可叠加在原始影像上显示（白色区域半透明覆盖）
- `multimask_output=true` 时，前端可让用户从 3 个候选掩码中选择最佳结果

---

## ⚠️ 错误处理

### 统一错误格式

所有错误返回统一的 JSON 结构：

```json
{
  "detail": "错误描述信息"
}
```

### 状态码说明

| 状态码 | 含义 | 场景 |
|--------|------|------|
| `200` | 成功 | 请求正常处理 |
| `400` | 请求格式错误 | Patch ID 格式非法、路径穿越尝试、模型未训练完成 |
| `401` | 未认证 | 缺少或错误的 API Key（仅在配置了 auth 时） |
| `403` | 禁止访问 | 访问了其他用户的 job |
| `404` | 未找到 | 区域/Patch/任务/结果/标签/模型不存在 |
| `413` | 请求体过大 | 文件超过 100MB 限制或图像超过 5000 万像素 |
| `422` | 参数校验失败 | 查询参数格式不正确（如 page=-1、bbox=nan、page_size=999） |
| `500` | 服务器错误 | 内部异常，需联系后端排查 |
| `501` | 未实现 | 标准 XYZ 瓦片服务等功能尚未实现 |
| `503` | 服务不可用 | 服务正在启动中、GPU 内存不足或模型加载失败 |

> **注意**：请求 `/regions/{region_id}/patches/{patch_id}/embedding?format=png` 时，如果该区域该 Patch 只有 `.npy` 格式，响应头会包含 `X-Available-Format: npy`，提示前端可以请求 `format=npy` 获取原始数据。

### 常见错误场景

**区域不存在**:
```bash
curl -s "http://60.31.21.42:22065/regions/beijing"
# → 404 {"detail": "Region 'beijing' not found"}
```

**Patch 不存在**:
```bash
curl -s "http://60.31.21.42:22065/regions/harbin/patches/patch_999999"
# → 404 {"detail": "Patch 'patch_999999' not found"}
```

**嵌入不存在**:
```bash
curl -s "http://60.31.21.42:22065/regions/haidian/patches/patch_999999/embedding"
# → 404 {"detail": "Embedding not found for patch 'patch_999999'"}
```

---

## 🔧 配置与扩展

### 配置文件位置

```
config.yaml
```

### 热重载机制

服务通过 `watchdog` 监听 `config.yaml` 的文件变化。修改配置后**无需重启服务**，约 1 秒内自动生效。

### 添加新区域

在 `config.yaml` 的 `regions` 下添加：

```yaml
regions:
  shenzhen:
    name: "深圳"
    patches_meta: "data/shenzhen/patches_meta.json"
    embeddings:
      v1:
        path: "data/shenzhen/embeddings/v1"
        template: "{month}/{patch_id}.{fmt}"
        formats: ["npy", "png", "json"]
    tasks:
      change_detection:
        name: "变化检测"
        description: "基于两期 embedding 差分的变化检测"
        versions:
          v1:
            results: "data/shenzhen/tasks/change_detection/v1/results"
            predictions: "data/shenzhen/tasks/change_detection/v1/predictions"
            labels: "data/shenzhen/tasks/change_detection/v1/labels"
```

保存后，访问 `/regions` 即可看到新区域。

### 添加新任务

在对应区域的 `tasks` 下添加：

```yaml
      green_space:
        name: "绿地变化监测"
        description: "监测城市绿地变化"
        versions:
          v1:
            results: "data/results/green_space"
            predictions: "data/predictions/green_space"
            labels: "data/labels/green_space"
```

---

## 💡 前端集成最佳实践

### 1. 页面加载流程

```
1. GET /health          → 检查服务可用
2. GET /regions         → 加载区域选择器
3. GET /regions/{id}/patches?page=1&bbox={map_view}  → 加载地图可见区域的 Patch
4. 用户点击 Patch → GET /regions/{id}/patches/{pid} → 展示 Patch 详情
```

### 2. 地图叠加流程

```
1. GET /regions/{id}/tasks → 加载任务选择器
2. 用户选择任务 → GET /regions/{id}/tasks/{task}/tiles → 加载瓦片列表
3. 按 Patch 展示结果图 → GET /regions/{id}/patches/{pid}/tasks/{task}/result?format=png
```

> 当前版本 `/tiles/{z}/{x}/{y}.png` 返回 501，建议先使用单 Patch 结果图或瓦片列表接口。

### 3. 前端 Fetch 示例

```typescript
// 1. 配置基地址（开发/生产环境切换）
const API_BASE = process.env.VITE_API_URL || 'http://60.31.21.42:22065';

// 2. 获取带 bbox 过滤的 Patch 列表
async function fetchPatches(regionId: string, bbox: string, page = 1, pageSize = 20) {
  const url = new URL(`${API_BASE}/regions/${regionId}/patches`);
  url.searchParams.set('page', String(page));
  url.searchParams.set('page_size', String(pageSize));
  if (bbox) url.searchParams.set('bbox', bbox);

  const resp = await fetch(url.toString());
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(`API ${resp.status}: ${err.detail}`);
  }
  return resp.json(); // { total, page, page_size, patches, has_next }
}

// 3. 加载任务结果图
function getResultImageUrl(regionId: string, patchId: string, taskType: string, version = 'v1', period?: string) {
  const url = new URL(`${API_BASE}/regions/${regionId}/patches/${patchId}/tasks/${taskType}/result`);
  url.searchParams.set('format', 'png');
  url.searchParams.set('version', version);
  if (period) url.searchParams.set('period', period);
  return url.toString();
}
```

### 4. 性能建议

- **分页加载**: Patch 数量多（400+），始终使用分页，通过 `has_next` 判断是否还有更多数据
- **bbox 过滤**: 地图移动时只请求可视区域的 Patch，减少数据传输
- **图片缓存**: PNG 结果图可设置浏览器缓存（内容不变），避免重复请求
- **懒加载**: 嵌入大图和任务结果图按需加载，不要一次性请求所有 Patch
- **错误降级**: 遇到 `404` 时不应中断整个应用，而是显示「暂无数据」提示；遇到 `503` 时提示用户稍后重试

---

## 📊 数据速查表

### 哈尔滨新区 (harbin)

| 属性 | 数值 |
|------|------|
| Patch 总数 | 424 |
| 覆盖范围 | 126.5°E ~ 126.7°E, 45.7°N ~ 45.8°N |
| 时间范围 | 2023-01 ~ 2025-10 |
| 数据源 | Sentinel-2, Sentinel-1, Landsat, DEM, WorldCover 等 |
| 嵌入格式 | PNG 64×64 RGB / NPY 原始数组 |
| 下游任务 | 5 个（`change_detection`, `building_extraction`, `land_use_classification`, `land_cover_classification`, `water_extraction`） |
| 有预生成结果的任务 | `change_detection` V1、<br>`building_extraction` V1/V2、<br>`land_use_classification` V1/V2 |
| 需实时推理的任务 | `land_cover_classification`、`water_extraction`（通过 `/system-models`） |

### 海淀区 (haidian)

| 属性 | 数值 |
|------|------|
| Patch 总数 | 320 |
| 覆盖范围 | 116.2°E ~ 116.3°E, 39.9°N ~ 40.1°N |
| 时间范围 | 2025-12 ~ 2026-05 |
| 数据源 | Sentinel-2, Sentinel-1, Landsat, 高分光学, 高分 SAR, WorldCover 等 |
| 嵌入格式 | NPY 多通道数组 / PNG PCA 预览 / JSON 统计 |
| 下游任务 | `building_extraction`, `road_extraction`, `construction`, `construction_joint`, `land_use_classification`, `land_cover_classification`, `water_extraction` |
| 有预生成结果的任务 | `building_extraction` V1、`road_extraction` V1、`construction` V1、`construction_joint` V1、`water_extraction` V1、`land_use_classification` V1、`land_cover_classification` V1 |
| 可用接口 | `/regions/haidian/patches/*`、`/regions/haidian/patches/*/embedding`、`/regions/haidian/patches/*/tasks/*/result` |

---

## 📞 技术支持

- **GitHub Issues**: [go-bananas-wwj/embedding-api/issues](https://github.com/go-bananas-wwj/embedding-api/issues)
- **服务监控**: `GET http://60.31.21.42:22065/health`
- **在线调试**: `http://60.31.21.42:22065/docs`
- **本地调试**: `http://localhost:9061/docs`
