# 部署与运维

本文档说明运行端口、启动方式、日志、请求审计和部署注意事项。

## 访问地址

| 入口 | 地址 |
|------|------|
| 公网 API | `http://60.31.21.42:22065` |
| 公网 Swagger | `http://60.31.21.42:22065/docs` |
| 公网日志 | `http://60.31.21.42:22065/logs` |
| 公网请求审计 | `http://60.31.21.42:22065/logs/request-audit` |
| 本地 API | `http://localhost:9061` |
| 可选本地静态日志 | `http://localhost:9091/logs` |

`22065` 是外部映射端口。部署机本机不一定能通过公网地址回环访问，但外部客户端可以使用这个地址。

## 启动 API

推荐：

```bash
python service_watchdog.py
```

停止：

```bash
python service_watchdog.py stop
```

开发模式：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 9061
```

## 可选静态日志服务

API 本身已经提供 `/logs`，只有需要单独暴露 `9091` 时才启动这个服务：

```bash
python scripts/static_logs_server.py --host 0.0.0.0 --port 9091 --root /workspace/embedding-api
```

## 日志文件

| 文件 | 用途 |
|------|------|
| `logs/uvicorn.log` | FastAPI/uvicorn 访问日志、异常堆栈、模型推理日志 |
| `logs/watchdog.log` | watchdog 生命周期、健康检查、重启记录 |
| `logs/watchdog.console.log` | watchdog stdout/stderr |
| `logs/request_audit.jsonl` | 结构化请求审计记录 |
| `logs/static_tmp_9091.log` | 可选静态日志服务访问日志 |

## 请求审计

打开：

```text
http://60.31.21.42:22065/logs/request-audit
```

审计页面展示：

- method 和 path
- query 参数
- JSON body 摘要
- 响应状态码
- 请求耗时
- client 和 user-agent

以下字段会自动脱敏：

- `Authorization`
- `X-API-Key`
- `token`
- `password`
- `secret`

以下非业务请求不会展示：

- `/logs/*`
- `/docs`
- `/redoc`
- `/openapi.json`
- `/favicon.ico`
- `/health`

默认轮转策略：

| 变量 | 默认值 |
|------|--------|
| `REQUEST_AUDIT_LOG` | `logs/request_audit.jsonl` |
| `REQUEST_AUDIT_ROTATE_BYTES` | `20971520` |
| `REQUEST_AUDIT_BACKUP_COUNT` | `5` |

## 环境变量

| 变量 | 默认值 | 用途 |
|------|--------|------|
| `CONFIG_PATH` | `./config.yaml` | 运行配置文件 |
| `CORS_ORIGINS` | 空 | 允许跨域访问的浏览器来源，逗号分隔 |
| `DOCS_URL` | `/docs` | Swagger 路径；设为 `none` 可关闭 |
| `REDOC_URL` | `/redoc` | ReDoc 路径；设为 `none` 可关闭 |
| `WATCHDOG_CHECK_INTERVAL` | `60` | watchdog 健康检查间隔 |
| `WATCHDOG_HEALTH_TIMEOUT` | `20` | watchdog 健康检查超时 |

## Docker

```bash
docker-compose up -d
```

Docker 默认暴露 `8000`。裸机部署当前使用 `9061`，公网 `22065` 由宿主机或反向代理映射。

## 验证

```bash
curl -s http://localhost:9061/health
curl -s http://localhost:9061/logs/request-audit >/tmp/request-audit.html
python -m pytest tests/ -q
```

完整在线 API 审计：

```bash
python scripts/full_api_audit.py
```
