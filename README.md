# Embedding API

Unified RESTful API for remote sensing embeddings and downstream task results from Harbin New Area and Haidian District.

## Features

- **Multi-region support**: Harbin (哈尔滨新区) & Haidian (海淀区)
- **Embedding queries**: PNG visualization, NPY arrays, JSON statistics
- **Downstream tasks**: 5 unified thematic tasks (change detection, building extraction, land use classification, land cover classification, water extraction)
- **Tile service**: Map tile serving for web GIS integration
- **SAM3 interactive segmentation**: Point-based instance segmentation on Sentinel-2 imagery
- **Hot-reload config**: Add new regions/tasks without restarting

## Requirements

- Python >= 3.9
- Key dependencies:
  - FastAPI >= 0.104
  - uvicorn >= 0.24
  - numpy == 1.26.4
  - Pillow >= 11.0
  - PyYAML >= 6.0
  - watchdog >= 3.0
  - torch == 2.5.1
  - torchvision == 0.20.1
  - rasterio >= 1.3.0

```bash
pip install -r requirements.txt
```

## Quick Start

### Production

**Base URL**: `http://60.31.21.42:22065` (HTTP, 内网/专线环境)

> ⚠️ 当前为 HTTP 协议。若前端页面部署在 HTTPS 域名下，浏览器会拦截混合内容请求。建议通过 Nginx 反向代理添加 HTTPS，或将前端与 API 部署在同源域名下。

- Swagger UI: http://60.31.21.42:22065/docs
- ReDoc: http://60.31.21.42:22065/redoc
- OpenAPI JSON: http://60.31.21.42:22065/openapi.json

> **注意**：上述在线文档默认已关闭（`DOCS_URL=none`），如需在线调试需显式开启。

### 哈尔滨新区任务说明

哈尔滨新区（`harbin`）当前提供以下统一后的下游监测任务：

| 任务 ID | 名称 | 版本 | 说明 |
|---------|------|------|------|
| `change_detection` | 变化检测 | v1 | 系统级两期 embedding 差分，按 period 子目录存 summary |
| `building_extraction` | 建筑物提取 | v1, v2 | v1 平铺；v2 按对比期分子目录 |
| `land_use_classification` | 土地利用分类 | v1, v2 | v1 平铺；v2 按对比期分子目录 |
| `land_cover_classification` | 土地覆盖分类 | - | 已配置，待补充数据 |
| `water_extraction` | 水体提取 | - | 已配置，待补充数据 |

`GET /regions/{region_id}/patches/{patch_id}` 返回的 `available_tasks` 字段只会列出对该 patch **有实际数据** 的任务。`land_cover_classification` 和 `water_extraction` 目前仅有配置/汇总数据，因此不会出现在 per-patch 的 `available_tasks` 中。

> 旧任务 ID（`construction`、`building_change`、`farmland`、`land_conversion`、`demolition`）已不再暴露给前端，原有数据目录通过 `config.yaml` 中的别名映射到新的统一任务 ID。

### Local Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 9061
```

- Swagger UI: http://localhost:9061/docs
- ReDoc: http://localhost:9061/redoc

### Docker

```bash
docker-compose up -d
```

Docker 内部监听 `8000`，`docker-compose.yml` 将宿主机 `8000` 映射到容器 `8000`。

### Environment Variables

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CONFIG_PATH` | `./config.yaml` | 配置文件路径 |
| `CORS_ORIGINS` | *(空)* | CORS 允许来源，逗号分隔；默认拒绝跨域请求 |
| `DOCS_URL` | `none` | Swagger UI 路径，设为 `/docs` 开启，默认关闭 |
| `REDOC_URL` | `none` | ReDoc 路径，设为 `/redoc` 开启，默认关闭 |

## Frontend Quick Start

```bash
# 1. 获取区域列表
curl http://60.31.21.42:22065/regions

# 2. 获取 Patch 列表（分页）
curl "http://60.31.21.42:22065/regions/harbin/patches?page=1&page_size=20"

# 3. 获取单个 Patch 的 Embedding 图片
# 直接放入 <img> 标签
<img src="http://60.31.21.42:22065/regions/harbin/patches/patch_000000/embedding?format=png" />

# 4. 获取 Embedding 统计（JSON）
curl "http://60.31.21.42:22065/regions/haidian/patches/patch_000000/embedding?format=json"
```

### TypeScript Types (from OpenAPI)

前端团队可以从 OpenAPI JSON 自动生成 TypeScript 类型：

```bash
npx openapi-typescript http://60.31.21.42:22065/openapi.json -o api-types.ts
```

### CORS

CORS 默认**不开放**（`allow_origins=[]`）。如需前端跨域访问，需通过 `CORS_ORIGINS` 环境变量显式设置允许的域名，例如：

```bash
CORS_ORIGINS="https://your-frontend.com,http://localhost:5173"
```

### Custom Models

```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"name": "my-building-head", "model_type": "classification", "task_type": "building_extraction", "region_id": "harbin"}' \
  -H 'X-API-Key: your-api-key' \
  http://60.31.21.42:22065/models
```

### Authentication

当前版本支持 **API-Key 用户隔离**。如果 `config.yaml` 中配置了 `auth.users`，请求需携带有效 key；未配置时默认使用 `default` 用户。

支持两种传 key 方式：

```bash
# 方式一：header
curl -H 'X-API-Key: your-api-key' http://60.31.21.42:22065/models

# 方式二：Bearer token
curl -H 'Authorization: Bearer your-api-key' http://60.31.21.42:22065/models
```

`config.yaml` 示例：

```yaml
auth:
  type: "api_key"
  users:
    key_alice_xxx:
      user_id: "alice"
      name: "Alice"
```

> 当前实现为可插拔依赖 `get_current_user`，后续可替换为 JWT/OAuth2 而不影响路由代码。

## API Endpoints

### Base
- `GET /health` - Health check

### Regions
- `GET /regions` - List all regions
- `GET /regions/{region_id}` - Get region details

### Patches
- `GET /regions/{region_id}/patches` - List patches (supports bbox filtering)
- `GET /regions/{region_id}/patches/{patch_id}` - Get patch details

### Embeddings
- `GET /regions/{region_id}/patches/{patch_id}/embedding?format=png|npy|json|cache`

### Downstream Tasks
- `GET /regions/{region_id}/tasks` - List 5 unified thematic tasks
- `GET /regions/{region_id}/tasks/{task_type}/summary` - Task summary
- `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result` - Per-patch result image
- `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/prediction` - Per-patch raw prediction
- `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/label` - Per-patch label data

> `result` 端点严格返回 patch 级别结果，不再回退到整幅 mosaic/summary。

### Annotations
- `GET /annotations/classes` - List classes
- `POST /annotations/classes` - Create class
- `PATCH /annotations/classes/{class_id}` - Rename class
- `DELETE /annotations/classes/{class_id}` - Delete class (cascades to annotations)
- `GET /annotations` - List annotations
- `POST /annotations` - Create annotation
- `GET /annotations/{ann_id}` - Get annotation
- `DELETE /annotations/{ann_id}` - Delete annotation

### Custom Models
- `GET /models` - List trained models
- `POST /models` - Create and train a model (async)
- `GET /models/{model_id}` - Get model status
- `POST /models/{model_id}/infer` - Single-patch inference
- `POST /models/{model_id}/infer_batch` - Batch inference (max 100)
- `GET /models/jobs/{job_id}` - Training job status
- `GET /models/results/{filename}` - Get inference result image

### System Models
- `GET /system-models?region_id={region_id}` - List system pre-trained heads
- `GET /system-models/{task_id}/classes?region_id={region_id}` - Get model classes
- `POST /system-models/{task_id}/infer?region_id={region_id}&patch_id={patch_id}&month={month}` - Inference

### Tiles
- `GET /regions/{region_id}/tasks/{task_type}/tiles` - List tiles
- `GET /regions/{region_id}/tasks/{task_type}/tiles/{z}/{x}/{y}.png` - Map tile

### SAM3 Interactive Segmentation
- `POST /regions/{region_id}/sam3/embed` - Preload S2 image and compute embedding
- `POST /regions/{region_id}/sam3/segment` - Segment with point prompts
- `GET /regions/{region_id}/sam3/status` - Model loading status and cache info

See `docs/API.md` for detailed SAM3 usage.

## Configuration

Edit `config.yaml` to add new regions or tasks. Changes are detected automatically without restart.

### 哈尔滨任务目录结构

不同任务版本采用两种目录布局，统一任务 ID 通过 `config.yaml` 中的 `versions` 映射到实际目录：

**v1 平铺布局**（如 `building_extraction` v1）：
```text
data/harbin/tasks/{task}/v1/predictions/patch_000000_2025-10.npy
data/harbin/tasks/{task}/v1/labels/patch_000000.npy
```

**v2 对比期子目录布局**（如 `building_extraction`、`land_use_classification` v2）：
```text
data/harbin/tasks/{task}/v2/predictions/2025-08_vs_2025-09/patch_000000.npy
data/harbin/tasks/{task}/v2/labels/2025-08_vs_2025-09/patch_000000.npy
```

`app/services/data_service.py` 已支持自动识别这两种布局，`available_tasks` 会正确返回 v2 子目录中的任务数据。

### 旧任务数据映射

原任务数据目录（`construction`、`building_change`、`farmland`、`land_conversion`、`demolition`）仍保留在磁盘上，`config.yaml` 通过版本别名将其映射到新的统一任务 ID，无需移动或重命名现有数据。

## Service Watchdog

```bash
# Start watchdog (auto-restart if service crashes)
python service_watchdog.py

# Stop watchdog
python service_watchdog.py stop
```

Watchdog 直接管理 `uvicorn` 子进程，支持 SIGTERM 优雅关闭和指数退避重试。

## Data Structure

See `docs/API.md` for detailed API documentation.
