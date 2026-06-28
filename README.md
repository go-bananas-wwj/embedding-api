# Embedding API

遥感 embedding 与下游监测任务的统一 RESTful API。当前支持 **哈尔滨新区** 和 **海淀区** 两个区域，提供 embedding 查询、专题任务结果、自定义模型训练与推理、SAM3 交互式分割等能力。

---

## 在线服务

- **Base URL**: `http://60.31.21.42:22065`
- **Swagger UI**: `http://60.31.21.42:22065/docs`
- **ReDoc**: `http://60.31.21.42:22065/redoc`

> 默认启动（`python service_watchdog.py`）会开启 Swagger；若直接用 `uvicorn` 启动，文档默认关闭，需设置 `DOCS_URL=/docs`。

---

## 主要能力

- **多区域**: `harbin`（哈尔滨新区）、`haidian`（海淀区）。
- **Embedding**: 按 Patch/月份获取 PNG 可视化、NPY 数组、JSON 统计。
- **专题任务**: 5 类统一任务 — 变化检测、建筑物提取、土地利用分类、土地覆盖分类、水体提取。
- **自定义模型**: 前端提交 GeoJSON 标注包（支持多类别、多标注），后端训练分类/变化检测头并推理，结果图为 128×128 PNG。
- **区域马赛克大图**: 按日期/传感器拼接整区域 S2/S1/Landsat 大图，用于前端展示。
- **SAM3 交互式分割**: 基于 Sentinel-2 影像的点提示实例分割。
- **配置热重载**: 修改 `config.yaml` 后无需重启即可生效。

> 自定义训练前端接入指南: [`docs/custom-training-workflow.md`](docs/custom-training-workflow.md)  
> 完整接口文档: [`docs/API.md`](docs/API.md)

---

## 快速开始

### 环境

```bash
pip install -r requirements.txt
```

### 启动（推荐）

```bash
python service_watchdog.py
```

服务监听 `0.0.0.0:9061`，watchdog 会在服务异常时自动重启：

```bash
python service_watchdog.py stop   # 停止
```

### 本地开发

```bash
DOCS_URL=/docs uvicorn app.main:app --reload --host 0.0.0.0 --port 9061
```

### Docker

```bash
docker-compose up -d
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CONFIG_PATH` | `./config.yaml` | 配置文件路径 |
| `CORS_ORIGINS` | *(空)* | 逗号分隔的允许跨域来源，未设置时拒绝跨域 |
| `DOCS_URL` | `none` | Swagger 路径，设为 `/docs` 开启 |
| `REDOC_URL` | `none` | ReDoc 路径，设为 `/redoc` 开启 |

---

## 接口概览

### 基础
- `GET /health` — 健康检查

### 区域与图块
- `GET /regions` — 区域列表
- `GET /regions/{region_id}` — 区域详情
- `GET /regions/{region_id}/patches` — Patch 列表（支持 bbox 过滤）
- `GET /regions/{region_id}/patches/{patch_id}` — Patch 详情

### Embedding
- `GET /regions/{region_id}/patches/{patch_id}/embedding?format=png|npy|json|cache&version=v1|v2&month=YYYY-MM`

### 专题任务
- `GET /regions/{region_id}/tasks` — 任务列表
- `GET /regions/{region_id}/tasks/{task_type}/summary` — 任务汇总
- `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result?format=png|npy&version=...` — Patch 级结果

### 区域马赛克大图
- `GET /regions/{region_id}/mosaic?date=YYYY-MM&sensor_type=s2|s1|landsat&format=png` — 整区域马赛克大图，支持 Sentinel-2 / Sentinel-1 / Landsat

### 自定义模型
- `GET /models` — 模型列表
- `POST /models` — 提交 GeoJSON 标注包并启动训练
- `GET /models/{model_id}` — 模型状态
- `POST /models/{model_id}/infer` — 单 Patch 推理
- `POST /models/{model_id}/infer_batch` — 批量推理（最多 100）
- `GET /models/jobs/{job_id}` — 训练任务状态
- `GET /models/results/{filename}` — 下载推理结果图

### 系统预训练模型
- `GET /system-models?region_id={region_id}` — 列出系统模型
- `GET /system-models/{task_id}/classes?region_id={region_id}` — 获取类别
- `POST /system-models/{task_id}/infer?region_id={region_id}&patch_id={patch_id}&month={month}` — 推理

### SAM3
- `POST /regions/{region_id}/sam3/embed` — 预加载影像并计算 embedding
- `POST /regions/{region_id}/sam3/segment` — 点提示分割
- `GET /regions/{region_id}/sam3/status` — 状态与缓存

---

## 认证

`config.yaml` 中未配置 `auth` 时，默认使用 `default` 用户，无需传 key。配置 `auth.users` 后，请求需携带 API Key：

```bash
curl -H 'X-API-Key: your-api-key' http://60.31.21.42:22065/models
# 或
curl -H 'Authorization: Bearer your-api-key' http://60.31.21.42:22065/models
```

---

## 数据配置

区域、任务、embedding 版本均在 `config.yaml` 中定义。修改后配置会被自动重新加载，无需重启服务。

---

## 目录结构

```text
app/                FastAPI 应用
  routers/          路由
  services/         业务逻辑
  schemas/          Pydantic 模型
data/               区域数据（embedding、任务结果）
docs/               接口文档与工作流说明
models/             系统预训练模型
sam3_pkg/           SAM3 相关代码
service_watchdog.py 服务守护脚本
tests/              测试用例
```

---

## 测试

```bash
python -m pytest -q
```

---

## 协议

- 在线服务当前为 HTTP；若前端部署在 HTTPS 域名下，需通过 Nginx 反向代理等方式统一协议，避免浏览器拦截混合内容。
