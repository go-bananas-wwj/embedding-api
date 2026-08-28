# 玄女遥感 Embedding API

面向遥感影像分析的 FastAPI 服务，统一提供 **Embedding 查询、预设任务推理、
自定义模型训练、批量推理和 SAM3 交互式分割**。

> 当前稳定版本：`embedding-api-20260730-stable`
>
> 对应提交：`22921a71569c2ee6f03dc0e27e67cc51339d58ba`

| 区域 | `region_id` | Patch 数 | 默认 Embedding |
|---|---:|---:|---|
| 北京海淀区 | `haidian` | 320 | P10C 64D，`v1` |
| 哈尔滨新区 | `harbin` | 424 | 玄女 V5，API 默认版本 |

## 从这里开始

根据你的目的选择一条路径：

| 目的 | 建议阅读 |
|---|---|
| 了解项目、查看接口 | 继续阅读本页，然后打开 [Swagger UI](http://60.31.21.42:22065/docs) |
| 在已有模型和数据的服务器启动 | 参照下方“快速启动” |
| 在一台新机器完整复现 | 阅读 [完整复现手册](docs/REPRODUCTION.md) |
| 维护或恢复稳定备份 | 阅读 [备份与恢复说明](docs/BACKUP_AND_RESTORE.md) |

完整复现需要从私有 ModelScope 数据集下载模型、数据和环境，正式载荷约
**57.02 GiB**。只有 GitHub 源码不能得到完整推理效果。

## 在线服务

| 入口 | 地址 |
|---|---|
| API 根地址 | `http://60.31.21.42:22065` |
| Swagger UI | [http://60.31.21.42:22065/docs](http://60.31.21.42:22065/docs) |
| ReDoc | [http://60.31.21.42:22065/redoc](http://60.31.21.42:22065/redoc) |
| 服务日志 | [http://60.31.21.42:22065/logs](http://60.31.21.42:22065/logs) |
| 前端请求审计 | [http://60.31.21.42:22065/logs/request-audit](http://60.31.21.42:22065/logs/request-audit) |

`22065` 是部署机的公网映射端口。部署机本身可能无法通过该地址回环访问；
本机检查请使用 `http://127.0.0.1:9061`。

## 主要能力

- **Embedding**：按区域、Patch、版本和月份获取 PNG、NPY、NPZ 或 JSON。
- **预设任务**：建筑物、道路、水体、施工地、土地利用、土地覆盖和变化检测。
- **自定义模型**：提交 GeoJSON 标注训练任务头，并对一个或多个 Patch 推理。
- **多基座训练**：支持玄女、传统特征、AEF 2025 和 DINOv3-SAT493M。
- **SAM3**：输入 WGS84 点提示，返回 WGS84 GeoJSON 分割多边形。
- **运行审计**：在浏览器中查看服务日志和真实业务请求参数。

接口参数、默认值、区域差异和请求示例以 Swagger UI 为准。

## 快速启动

本节适用于**项目目录中已经存在 `models/` 和 `data/` 完整资产**的机器。
全新机器请改用 [完整复现手册](docs/REPRODUCTION.md)。

```bash
cd embedding-api
python service_watchdog.py start
python service_watchdog.py status
curl -fsS http://127.0.0.1:9061/health
```

本地入口：

```text
API      http://127.0.0.1:9061
Swagger  http://127.0.0.1:9061/docs
日志     http://127.0.0.1:9061/logs
```

开发模式：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 9061
```

运行测试：

```bash
python -m pytest tests -q
```

## 最小调用示例

```bash
export BASE_URL="http://127.0.0.1:9061"

curl -fsS "$BASE_URL/health"
curl -fsS "$BASE_URL/regions"

curl -fsS \
  "$BASE_URL/regions/haidian/patches/patch_000000/embedding?format=json&version=v1&month=202512"

curl -fsS \
  "$BASE_URL/regions/haidian/patches/patch_000000/tasks/road_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian-road.png
```

## 项目怎么读

```text
app/          API 路由、数据模型和业务服务
config.yaml   区域、Embedding、任务头和数据路径
models/       SAM3、系统模型和下游任务头
data/         区域元数据、Embedding 与任务结果
pipelines/    海淀、哈尔滨资产下载和整理流程
scripts/      训练、审计、备份校验和可视化工具
tests/        API 与业务逻辑测试
docs/         接口、训练、部署和复现文档
```

建议阅读顺序：

1. [API 快速入门](docs/api-quickstart.md)
2. [完整接口说明](docs/API.md)
3. [自定义训练流程](docs/custom-training-workflow.md)
4. [模型训练对接](docs/model-training-integration.md)
5. [运维说明](docs/operations.md)

## 文档导航

| 文档 | 内容 |
|---|---|
| [完整复现手册](docs/REPRODUCTION.md) | 新机器下载、校验、解压、启动与验收 |
| [API 快速入门](docs/api-quickstart.md) | 前端常用请求示例 |
| [完整接口说明](docs/API.md) | 接口与 curl 清单 |
| [自定义训练流程](docs/custom-training-workflow.md) | 训练、模型 ID 和批量推理 |
| [PU + Query 复现](docs/pu-query-reproduction.md) | 少样本检索原理和跨机器复现 |
| [模型训练对接](docs/model-training-integration.md) | 多种基座与前端契约 |
| [海淀资产说明](docs/haidian-assets.md) | 最新海淀 ModelScope 资产 |
| [运维说明](docs/operations.md) | 端口、日志和 Watchdog |
| [备份与恢复说明](docs/BACKUP_AND_RESTORE.md) | 稳定备份维护规则 |
| [AGENTS.md](AGENTS.md) | AI coding agent 项目指南 |

## 数据与代码来源

- 代码仓库：[go-bananas-wwj/embedding-api](https://github.com/go-bananas-wwj/embedding-api)
- 海淀最新资产：[WeijieWu/xuannv_haidian_embdding](https://modelscope.cn/datasets/WeijieWu/xuannv_haidian_embdding)
- 私有灾备仓库：[WeijieWu/xuannv_embdding_backup](https://modelscope.cn/datasets/WeijieWu/xuannv_embdding_backup)
- 玄女 Embedding 参考实现：[go-bananas-wwj/xuannv_embdding](https://github.com/go-bananas-wwj/xuannv_embdding/tree/v3-semantic-64d)

私有 ModelScope 数据集必须获得访问权限。Token 只应通过临时环境变量传入，
不要写进代码、README、配置文件或 Shell 历史。

## 生产注意事项

- 建议通过 HTTPS 反向代理暴露服务，避免浏览器混合内容限制。
- 浏览器跨域来源通过 `CORS_ORIGINS` 配置。
- 自定义模型默认保留 24 小时，并由后台定时清理；官方预设模型不受影响。
- 后台在服务启动时和每天北京时间 00:00 执行清理；可使用
  `CUSTOM_MODEL_CLEANUP_ENABLED` 和 `CUSTOM_MODEL_TTL_HOURS` 调整清理策略。
- 如需关闭在线文档，设置 `DOCS_URL=none` 和 `REDOC_URL=none`。
