# Embedding API

Unified RESTful API for remote sensing embeddings and downstream task results from Harbin New Area and Haidian District.

## Features

- **Multi-region support**: Harbin (哈尔滨新区) & Haidian (海淀区)
- **Embedding queries**: PNG visualization, NPY arrays, JSON statistics
- **Downstream tasks**: Construction, building change, farmland, land conversion, demolition monitoring
- **Tile service**: Map tile serving for web GIS integration
- **Hot-reload config**: Add new regions/tasks without restarting

## Requirements

- Python >= 3.9
- Key dependencies:
  - FastAPI >= 0.100
  - uvicorn >= 0.20
  - numpy >= 1.20
  - Pillow >= 9.0
  - PyYAML >= 6.0
  - watchdog >= 3.0

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

Docker 内部使用 `config.docker.yaml`，数据路径映射到容器内的 `/data/`。

### Environment Variables

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CONFIG_PATH` | `./config.yaml` | 配置文件路径 |
| `CORS_ORIGINS` | `*` | CORS 允许来源，生产环境建议设为具体域名 |
| `DOCS_URL` | `/docs` | Swagger UI 路径，设为 `none` 禁用 |
| `REDOC_URL` | `/redoc` | ReDoc 路径，设为 `none` 禁用 |

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

当前版本 CORS 已全局开放（`allow_origins=["*"]`），前端可直接跨域请求。
生产环境建议通过 `CORS_ORIGINS` 环境变量限制来源。

### Authentication

当前版本 **无认证机制**，API 完全开放。公网部署时建议通过 Nginx/反向代理添加 Basic Auth 或 API Key。

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
- `GET /regions/{region_id}/tasks` - List tasks
- `GET /regions/{region_id}/tasks/{task_type}/summary` - Task summary
- `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result` - Result image
- `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/prediction` - Raw prediction
- `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/label` - Label data

### Tiles
- `GET /regions/{region_id}/tasks/{task_type}/tiles` - List tiles
- `GET /regions/{region_id}/tasks/{task_type}/tiles/{z}/{x}/{y}.png` - Map tile

## Configuration

Edit `config.yaml` to add new regions or tasks. Changes are detected automatically without restart.

## Service Watchdog

```bash
# Start watchdog (auto-restart if service crashes)
python service_watchdog.py

# Stop watchdog
python service_watchdog.py stop
```

## Data Structure

See `docs/API.md` for detailed API documentation.
