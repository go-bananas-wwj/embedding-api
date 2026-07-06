# Embedding API

遥感 embedding、下游监测结果、自定义训练与 SAM3 交互式分割的统一 FastAPI 服务。

本仓库面向三类使用者：

- 前端同事：查接口、联调参数、看返回效果和请求日志。
- 算法/数据同事：替换区域 embedding、模型权重和下游任务头。
- 部署/运维同事：启动服务、查看后台日志、定位接口错误。

---

## 快速入口

| 场景 | 地址 |
|------|------|
| 在线服务 Base URL | `http://60.31.21.42:22065` |
| Swagger UI | `http://60.31.21.42:22065/docs` |
| ReDoc | `http://60.31.21.42:22065/redoc` |
| 后台日志首页 | `http://60.31.21.42:22065/logs` |
| 前端请求审计日志 | `http://60.31.21.42:22065/logs/request-audit` |
| 本地 API | `http://localhost:9061` |
| 本地静态日志端口 | `http://localhost:9091/logs` |

> 说明：`22065` 是当前部署机映射到公网的访问端口。部署机本机不一定能通过公网地址回环访问，这是网络环境问题，不代表外部访问不可用。

---

## 当前能力

| 能力 | 说明 |
|------|------|
| 多区域数据 | 支持 `harbin`（哈尔滨新区，424 patches）和 `haidian`（海淀区，320 patches）。 |
| Embedding 查询 | 支持 PNG、NPY、NPZ、JSON 和 cache fallback 格式。 |
| 下游任务结果 | 支持变化检测、建筑物提取、道路提取、施工地检测、土地利用/覆盖分类、水体提取等任务。 |
| 海淀最新模型 | 海淀区使用 ModelScope 数据集 `WeijieWu/xuannv_haidian_embdding` 的 `artifacts/haidian-embedding-v1` 最新版本。 |
| SAM3 分割 | 前端传 WGS84 点提示和传感器类型，后端返回 WGS84 GeoJSON `Polygon` / `MultiPolygon` 标注轮廓。 |
| 自定义训练 | 前端提交 GeoJSON 标注包，后端训练分类或变化检测任务头。 |
| 批量推理 | 支持自定义模型和系统模型批量推理，返回每个 patch 的结果状态和结果图 URL。 |
| 日志审计 | `/logs/request-audit` 展示真实业务 API 的 method、path、query、JSON body、状态码和耗时。 |
| 配置热重载 | `config.yaml` 修改后会自动加载，减少重启成本。 |

---

## 目录

- [快速开始](#快速开始)
- [在线调试与日志](#在线调试与日志)
- [海淀最新模型与数据](#海淀最新模型与数据)
- [常用 API 示例](#常用-api-示例)
- [自定义训练与批量推理](#自定义训练与批量推理)
- [SAM3 分割接口](#sam3-分割接口)
- [配置与环境变量](#配置与环境变量)
- [开发、测试与部署](#开发测试与部署)
- [项目结构](#项目结构)
- [安全与上线注意事项](#安全与上线注意事项)

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 API 服务

推荐使用 watchdog，它会在服务不健康时自动重启 uvicorn：

```bash
python service_watchdog.py
```

默认监听：

```text
http://0.0.0.0:9061
```

停止 watchdog：

```bash
python service_watchdog.py stop
```

开发模式也可以直接运行：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 9061
```

### 3. 启动本地静态日志页

如果需要把日志页单独暴露在 `9091`：

```bash
python scripts/static_logs_server.py --host 0.0.0.0 --port 9091 --root /workspace/embedding-api
```

访问：

```text
http://localhost:9091/logs
```

---

## 在线调试与日志

### Swagger / ReDoc

- Swagger UI：`http://60.31.21.42:22065/docs`
- ReDoc：`http://60.31.21.42:22065/redoc`

Swagger 中每个参数都补充了中文说明、默认值、示例值和填写建议。前端联调优先使用 Swagger。

### 请求审计日志

访问：

```text
http://60.31.21.42:22065/logs/request-audit
```

页面展示真实业务请求：

- HTTP method
- API path
- query 参数
- JSON request body
- 响应状态码
- 请求耗时
- 调用方 IP / user-agent

不会展示这些噪音请求：

- `/logs/*`
- `/docs`
- `/redoc`
- `/openapi.json`
- `/favicon.ico`
- `/health`

敏感字段会自动脱敏：

- `Authorization`
- `X-API-Key`
- `token`
- `password`
- `secret`

日志存储策略：

- 原始文件：`logs/request_audit.jsonl`
- 页面默认只读取最近 200 条
- 单文件默认 20MB 自动轮转
- 默认保留 5 个历史文件

### 后台服务日志

| 文件 | 用途 |
|------|------|
| `logs/uvicorn.log` | FastAPI / uvicorn 请求日志、异常堆栈、模型推理日志。 |
| `logs/watchdog.log` | watchdog 启动、健康检查和重启记录。 |
| `logs/watchdog.console.log` | watchdog stdout / stderr。 |
| `logs/static_tmp_9091.log` | 9091 静态日志服务访问日志。 |

---

## 海淀最新模型与数据

海淀区当前版本必须使用 ModelScope 数据集：

```text
https://modelscope.cn/datasets/WeijieWu/xuannv_haidian_embdding
```

当前部署来源：

```text
artifacts/haidian-embedding-v1
```

旧版 `haidian/v1/api_ready` 已不再作为当前部署来源。

### 下载并替换海淀资产

```bash
export MODELSCOPE_TOKEN="..."  # 私有数据集需要，不要写入代码

python pipelines/haidian/download_modelscope_assets.py \
  --repo WeijieWu/xuannv_haidian_embdding \
  --prefix artifacts/haidian-embedding-v1 \
  --target .
```

安装后关键文件位置：

| 类型 | 路径 |
|------|------|
| Haidian embedding | `data/haidian/embeddings/v1/{YYYYMM}/{patch_id}.npy|png|json` |
| Embedding checkpoint | `models/haidian/v1/embedding/haidian_embedding_v1_p10c_epoch800.pt` |
| 建筑物任务头 | `models/haidian/v1/task_heads/building_mlp_fold0_best.pt` |
| 道路任务头 | `models/haidian/v1/task_heads/road_mlp_fold0_best.pt` |
| 水体任务头 | `models/haidian/v1/task_heads/water_mlp_fold0_best.pt` |

当前可用月份：

```text
202512, 202601, 202602, 202603, 202604, 202605
```

### 推荐联调参数

```text
region_id = haidian
patch_id  = patch_000000
version   = v1
month     = 202512
```

---

## 常用 API 示例

下面示例使用在线服务地址。若本地联调，将 `BASE` 改成 `http://localhost:9061`。

```bash
export BASE="http://60.31.21.42:22065"
```

### 健康检查

```bash
curl -s "$BASE/health"
```

### 区域与 Patch

```bash
curl -s "$BASE/regions"
curl -s "$BASE/regions/haidian"
curl -s "$BASE/regions/haidian/patches?page=1&page_size=10"
curl -s "$BASE/regions/haidian/patches/patch_000000"
```

### Embedding

```bash
curl -s "$BASE/regions/haidian/patches/patch_000000/embedding?format=json&version=v1&month=202512"

curl -s "$BASE/regions/haidian/patches/patch_000000/embedding?format=png&version=v1&month=202512" \
  -o /tmp/haidian_embedding.png
```

### 下游任务结果

```bash
curl -s "$BASE/regions/haidian/tasks"

curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_building.png

curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/road_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_road.png

curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/water_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_water.png
```

### 系统模型推理

```bash
curl -s "$BASE/system-models?region_id=haidian"

curl -s -X POST "$BASE/system-models/road_extraction/infer?region_id=haidian&patch_id=patch_000000&month=202512&version=v1"
```

### 区域马赛克

```bash
curl -s "$BASE/regions/haidian/mosaic?date=202512&sensor_type=s2&format=png&patch_ids=patch_000000&patch_ids=patch_000001" \
  -o /tmp/haidian_mosaic.png
```

说明：

- `format=png` 返回可视化 RGB 图。
- `format=tif` 返回 GeoTIFF。
- `patch_ids` 不传时拼全区域，生成时间和内存占用会更高。

---

## 自定义训练与批量推理

自定义训练由前端提交 GeoJSON 标注包，后端异步训练任务头。

### 创建训练任务

```bash
curl -s -X POST "$BASE/models" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "haidian-building-demo",
    "model_type": "classification",
    "task_type": "building_extraction",
    "region_id": "haidian",
    "embedding_version": "v1",
    "epochs": 20,
    "classes": [
      {"id": "cls_001", "name": "建筑物", "color": "#FF3B30"}
    ],
    "annotations": {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {
            "patch_id": "patch_000000",
            "region_id": "haidian",
            "class_id": "cls_001",
            "task_type": "building_extraction",
            "month": "202512"
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [[
              [116.3000, 39.9800],
              [116.3050, 39.9800],
              [116.3050, 39.9850],
              [116.3000, 39.9850],
              [116.3000, 39.9800]
            ]]
          }
        }
      ]
    }
  }'
```

返回中会包含：

- `model_id`
- `job_id`
- `status`

### 查询训练状态

```bash
curl -s "$BASE/models/jobs/{job_id}"
```

### 单 Patch 推理

```bash
curl -s -X POST "$BASE/models/{model_id}/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "region_id": "haidian",
    "patch_id": "patch_000000",
    "month": "202512"
  }'
```

### 批量推理

```bash
curl -s -X POST "$BASE/models/{model_id}/infer_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "region_id": "haidian",
    "patch_ids": ["patch_000000", "patch_000001"],
    "month": "202512"
  }'
```

更多细节见：

- [`docs/custom-training-workflow.md`](docs/custom-training-workflow.md)
- [`docs/API.md`](docs/API.md)

---

## SAM3 分割接口

SAM3 用于基于点提示做交互式实例分割。输入坐标使用 WGS84，经纬度顺序为：

```text
[longitude, latitude]
```

### 请求示例

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

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `date` | 是 | 无 | 影像日期或月份。海淀推荐 `202512`。 |
| `sensor_type` | 是 | `s2` | 传感器类型，当前主要使用 Sentinel-2。 |
| `point_coords` | 是 | 无 | 点提示坐标数组，格式为 `[[lon, lat], ...]`。 |
| `point_labels` | 否 | 全部为 `1` | 点标签。`1` 表示前景点；当前前端一般不需要传排除点。 |
| `multimask_output` | 否 | `true` | 是否返回多个候选 mask。 |
| `include_masks` | 否 | `false` | 是否返回原始 mask 数组。前端通常保持 `false`，使用 GeoJSON 轮廓即可。 |

### 返回说明

返回为 WGS84 GeoJSON FeatureCollection。每个候选结果包含：

- `geometry.type`: `Polygon` 或 `MultiPolygon`
- `properties.score`: SAM3 置信度
- `properties.geometry_kind`: 通常为 `mask_polygon`
- `properties.point_coords`: 本次使用的点提示

---

## 配置与环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CONFIG_PATH` | `./config.yaml` | 运行配置文件路径。 |
| `CORS_ORIGINS` | 空 | 逗号分隔的允许跨域来源。生产环境前端浏览器调用时必须设置。 |
| `DOCS_URL` | `/docs` | Swagger UI 路径；设为 `none` 可关闭。 |
| `REDOC_URL` | `/redoc` | ReDoc 路径；设为 `none` 可关闭。 |
| `REQUEST_AUDIT_LOG` | `logs/request_audit.jsonl` | 请求审计日志文件。 |
| `REQUEST_AUDIT_ROTATE_BYTES` | `20971520` | 请求审计日志单文件最大字节数，默认 20MB。 |
| `REQUEST_AUDIT_BACKUP_COUNT` | `5` | 请求审计日志历史文件保留数量。 |
| `WATCHDOG_CHECK_INTERVAL` | `60` | watchdog 健康检查间隔。 |
| `WATCHDOG_HEALTH_TIMEOUT` | `20` | watchdog 健康检查总超时。 |

---

## 开发、测试与部署

### 本地测试

```bash
python -m pytest tests/ -q
```

SAM3 真实模型加载测试较慢，需要 GPU 和模型文件：

```bash
python -m pytest tests/ -q -m slow
```

### Docker

```bash
docker-compose up -d
```

Docker 默认暴露：

```text
0.0.0.0:8000
```

### 裸机部署

```bash
python service_watchdog.py
```

推荐把公网端口 `22065` 反向代理到本机 API 端口 `9061`：

```text
public :22065  ->  localhost:9061
```

日志页已经挂在 API 服务本身，公网可直接访问：

```text
http://60.31.21.42:22065/logs
http://60.31.21.42:22065/logs/request-audit
```

---

## 项目结构

```text
embedding-api/
├── app/
│   ├── main.py                 # FastAPI app、CORS、OpenAPI、请求审计
│   ├── routers/                # API 路由
│   ├── schemas/                # Pydantic 请求/响应模型
│   └── services/               # 数据、模型、SAM3、训练、推理逻辑
├── data/                       # 区域 embedding、任务结果、patch 元数据
├── docs/                       # API 文档、训练工作流、实施计划
├── logs/                       # uvicorn、watchdog、请求审计日志
├── models/                     # 系统模型 checkpoint 和任务头
├── pipelines/                  # ModelScope 数据下载与整理脚本
├── scripts/                    # 审计、可视化和静态日志服务脚本
├── sam3_pkg/                   # 本地 SAM3 包
├── tests/                      # pytest 测试
├── config.yaml                 # 默认运行配置
├── docker-compose.yml
└── service_watchdog.py
```

---

## 安全与上线注意事项

- 文件路径相关接口必须保留 patch_id、month、period 校验和路径 containment 检查。
- 生产环境建议设置 `CORS_ORIGINS`，不要使用宽泛跨域。
- 如配置 `auth.users`，`/models/*` 和 `/system-models/*` 需要 `X-API-Key` 或 `Authorization: Bearer <key>`。
- Swagger/ReDoc 当前为了联调默认开启；正式公网环境可通过 `DOCS_URL=none`、`REDOC_URL=none` 关闭。
- 请求审计日志会记录业务请求 body，请不要在业务请求中传明文密钥；常见鉴权字段会自动脱敏。
- 当前在线服务是 HTTP。如果前端是 HTTPS 域名，建议通过同一 HTTPS 反向代理转发，避免浏览器混合内容拦截。

---

## 更多文档

- [`docs/API.md`](docs/API.md)：完整接口说明和 curl 清单。
- [`docs/custom-training-workflow.md`](docs/custom-training-workflow.md)：自定义训练前端接入流程。
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)：近期修复、上线检查和实施记录。
- [`AGENTS.md`](AGENTS.md)：给 AI coding agents 的项目说明。

---

## README 设计参考

README 结构参考了 GitHub 官方 README 指南、The Good Docs Project 的 README / Quickstart 模板思路，以及 API 文档平台常见的信息架构：先给入口和快速开始，再给核心用法、运维日志、配置、测试与部署。
