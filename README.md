# Embedding API

面向遥感业务的 FastAPI 服务，统一提供 embedding 查询、下游监测任务、自定义模型训练、批量推理和 SAM3 交互式分割能力。

当前支持：

| 区域 | ID | Patch 数 | 说明 |
|------|----|----------|------|
| 哈尔滨新区 | `harbin` | 424 | embedding、监测任务、SAM3、系统模型 |
| 北京海淀区 | `haidian` | 320 | 最新 ModelScope 资产、建筑/道路/水体任务头、SAM3 |

---

## 在线服务

| 入口 | 地址 |
|------|------|
| Base URL | `http://60.31.21.42:22065` |
| Swagger UI | `http://60.31.21.42:22065/docs` |
| ReDoc | `http://60.31.21.42:22065/redoc` |
| 后台日志 | `http://60.31.21.42:22065/logs` |
| 前端请求审计 | `http://60.31.21.42:22065/logs/request-audit` |

`22065` 是部署机映射到公网的访问端口。部署机本机不一定能通过这个公网地址回环访问，这是网络环境限制，不代表外部用户不可访问。

---

## 主要能力

- **Embedding 查询**：按区域、patch、版本、月份获取 PNG / NPY / NPZ / JSON。
- **任务结果**：建筑物、道路、水体、施工地、土地利用/覆盖、变化检测等结果查询。
- **系统模型**：调用官方预训练任务头进行推理。
- **自定义训练**：前端提交 GeoJSON 标注包，后端训练用户自己的任务头。
- **批量推理**：支持自定义模型和系统模型对多个 patch 批量推理。
- **SAM3 分割**：输入 WGS84 点提示，返回 WGS84 GeoJSON 多边形。
- **日志审计**：浏览器里查看后端日志和真实前端 API 请求参数。

---

## 快速开始

```bash
pip install -r requirements.txt
python service_watchdog.py
```

本地入口：

```text
API:      http://localhost:9061
Swagger: http://localhost:9061/docs
日志:     http://localhost:9061/logs
```

开发模式：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 9061
```

运行测试：

```bash
python -m pytest tests/ -q
```

---

## 常用接口示例

```bash
export BASE="http://60.31.21.42:22065"

curl -s "$BASE/health"
curl -s "$BASE/regions"
curl -s "$BASE/regions/haidian/patches/patch_000000/embedding?format=json&version=v1&month=202512"
curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/road_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_road.png
```

前端联调优先看 Swagger：

```text
http://60.31.21.42:22065/docs
```

---

## 文档导航

| 文档 | 适合查看 |
|------|----------|
| [`docs/API.md`](docs/API.md) | 完整接口说明和 curl 清单 |
| [`docs/api-quickstart.md`](docs/api-quickstart.md) | 前端常用接口示例 |
| [`docs/operations.md`](docs/operations.md) | 部署、端口、日志、watchdog、请求审计 |
| [`docs/haidian-assets.md`](docs/haidian-assets.md) | 海淀最新 ModelScope 资产和替换流程 |
| [`docs/pu-query-reproduction.md`](docs/pu-query-reproduction.md) | PU + Query 原理、模型格式与跨机器复现 |
| [`docs/custom-training-workflow.md`](docs/custom-training-workflow.md) | 自定义训练和批量推理流程 |
| [`docs/model-training-integration.md`](docs/model-training-integration.md) | 变化检测、土地分类和四种训练方式的前端对接契约 |
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | 本轮生产修复与验证记录 |
| [`AGENTS.md`](AGENTS.md) | 给 AI coding agents 的项目说明 |

---

## 海淀最新资产

海淀区 `v1` 使用最新 ModelScope 数据集：

```text
https://modelscope.cn/datasets/WeijieWu/xuannv_haidian_embdding
```

当前来源前缀：

```text
artifacts/haidian-embedding-v1
```

推荐联调参数：

```text
region_id = haidian
patch_id  = patch_000000
version   = v1
month     = 202512
```

下载和替换方式见 [`docs/haidian-assets.md`](docs/haidian-assets.md)。

---

## 项目结构

```text
app/          FastAPI 应用、路由、schema、业务服务
data/         区域元数据、embedding、任务结果
docs/         人类可读文档
models/       系统模型 checkpoint 和任务头
pipelines/    ModelScope 下载与整理脚本
scripts/      审计、可视化、静态日志服务脚本
tests/        pytest 测试
```

---

## 上线注意事项

- 当前公网服务是 HTTP；如果前端是 HTTPS，建议通过同一个 HTTPS 反向代理转发，避免浏览器拦截混合内容。
- 请求审计日志会自动脱敏常见密钥字段，并过滤 `/logs`、`/docs`、`/openapi.json`、`/favicon.ico`、`/health` 等非业务请求。
- 生产环境浏览器调用需要设置 `CORS_ORIGINS`。
- 如需关闭公网文档，可设置 `DOCS_URL=none` 和 `REDOC_URL=none`。

---

## README 原则

根 README 只做项目入口，不承载完整接口手册。详细说明、操作步骤和接口 reference 都放在 [`docs/`](docs/) 下。
