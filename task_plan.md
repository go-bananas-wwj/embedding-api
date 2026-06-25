# Embedding API 下游任务重构计划

## 目标

按用户要求对 `embedding-api` 进行三方面的改造：

1. **专题类型统一**：哈尔滨与海淀区统一暴露为 5 类下游专题。
2. **Get Task Result 按 patch 返回**：严格返回指定区域、指定 patch 的分析结果，不再回退到整幅 mosaic/summary。
3. **自定义训练 + 批量推理 API**：参考 `xuannv_show` 的训练/推理能力，提供模型注册、异步训练、单 patch 推理、批量推理接口。

> **范围约定**：海淀区本次仅暴露任务列表，不挂载真实数据。

---

## 需求细化

### 1. 专题类型统一

统一后的 5 类专题：

| 专题 ID | 中文名 | 任务类型 | 版本策略 |
|---------|--------|----------|----------|
| `change_detection` | 变化检测 | 双期差分二分类 | v1 |
| `land_cover_classification` | 土地覆盖分类 | 单期多分类 | v1 |
| `land_use_classification` | 土地利用分类 | 单期多分类 | v1 |
| `water_extraction` | 水体提取 | 单期二分类 | v1 |
| `building_extraction` | 建筑物提取 | 单期二分类 | v1 |

#### 哈尔滨映射

将现有数据/模型映射到新专题：

| 现有任务 | 映射到 | 说明 |
|----------|--------|------|
| `change_detection` | `change_detection` | 保留，双期差分 |
| `construction` | `building_extraction` / `change_detection` | 建筑工地监测可视为建筑物提取 + 变化检测的组合；建议保留为 `building_extraction` 的训练数据来源 |
| `building_change` | `change_detection`（建筑物变化） | 可合并入 `change_detection`，或作为变化检测的一个子类 |
| `farmland` | `land_use_classification` | 耕地监测属于土地利用分类 |
| `land_conversion` | `land_use_classification` / `change_detection` | 土地转换监测涉及分类 + 变化检测 |
| `demolition` | `change_detection`（拆迁变化） | 拆迁监测属于变化检测 |

**建议**：哈尔滨保留既有物理目录和模型不变，仅在 API 层做任务别名/映射。新增统一专题的配置层，旧任务 ID 仍可向后兼容（可选）。

#### 海淀区

- `config.yaml` 中 `haidian.tasks` 配置 5 类专题，但目录指向空路径或占位路径。
- `GET /regions/haidian/tasks` 返回 5 个任务；请求具体结果时返回 404/503。
- 不复制真实数据。

### 2. Get Task Result 严格按 patch 返回

当前 `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result` 实现中：

```python
# Dynamic discovery: find first PNG instead of hardcoded names
path = DataService._find_first_file(base, "*.png")
if path:
    return path
# Fallback to common names
for fname in ["result.png"]:
    path = _resolve_path(base, fname)
    if path:
        return path
```

这段逻辑会返回任意 PNG 或整图 `result.png`，导致返回整张大图而非指定 patch。

**修改方案**：
- 移除 `_find_first_file(base, "*.png")` 和 `result.png` fallback。
- `result` 接口只按以下规则查找：
  - v1 平铺：`{base}/{patch_id}_{period}.png` 或 `{base}/{period}.png`（如果任务本身就是 per-period 汇总）
  - v2 子目录：`{base}/{period}/{patch_id}.png` 或 `{base}/{period}/{period}.png`
- 对于真正的 mosaic/summary，前端应调用 `/regions/{region_id}/tasks/{task_type}/summary` 或新增 `/mosaic` 端点，而不是 per-patch result。

### 3. 自定义训练 + 批量推理 API

参考 `xuannv_show/backend/app/routers/annotate.py`，在 embedding-api 中新增 `/models` 路由模块。

#### 3.1 模块设计

```text
app/
├── routers/
│   └── models.py          # 模型注册、训练、推理、批量推理
├── schemas/
│   └── models.py          # Pydantic 请求/响应模型（已存在，需扩展）
└── services/
    ├── model_registry.py  # 模型注册表 CRUD
    ├── training_engine.py # 分类头训练
    ├── cd_training_engine.py # 变化检测头训练
    └── inference_engine.py # 单 patch / 批量推理
```

#### 3.2 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/models` | 列出用户训练好的模型 |
| POST | `/models` | 创建模型并启动训练（异步） |
| GET | `/models/{model_id}` | 获取模型状态 |
| DELETE | `/models/{model_id}` | 删除模型 |
| POST | `/models/{model_id}/infer` | 单 patch 推理 |
| POST | `/models/{model_id}/infer_batch` | 批量推理（同步） |
| GET | `/models/jobs/{job_id}` | 查询训练/批量推理任务状态 |
| GET | `/system-models` | 列出系统预训练模型 |
| POST | `/system-models/{model_id}/infer` | 系统模型单 patch 推理 |

#### 3.3 训练流程

1. 用户通过已有标注接口（或未来新增的标注接口）提交 annotations/masks。
2. `POST /models` 创建模型记录（状态 `training`），返回 `model_id` 和 `job_id`。
3. BackgroundTasks 调用训练引擎：
   - 加载 embedding（单期或双期）。
   - 加载 mask。
   - 训练 sklearn LogisticRegression（或复用 xuannv_show 逻辑）。
   - 保存 `.pkl` 到 `users/{user_id}/models/{model_id}.pkl`。
   - 更新模型状态为 `completed` 或 `failed`。
4. 用户轮询 `GET /models/jobs/{job_id}` 获取状态。

#### 3.4 推理流程

1. `POST /models/{model_id}/infer`：
   - 加载模型 `.pkl`。
   - 加载指定 patch 的 embedding。
   - 推理并生成 PNG（256×256）。
   - 保存到 `users/{user_id}/results/{model_id}_{patch_id}_{month}.png`。
   - 返回 PNG URL 或直接返回文件。

2. `POST /models/{model_id}/infer_batch`：
   - 接收 `patch_ids` 列表。
   - 逐个推理，返回每个 patch 的结果 URL/状态。

#### 3.5 用户隔离

- 当前 embedding-api 无用户认证，先用 `user_id=default` 占位。
- 预留 `users/{user_id}/` 目录结构，后续接入认证后可按真实用户隔离。

---

## 详细执行步骤

### Phase 1: 专题类型统一配置

1. **修改 `config.yaml`**
   - 哈尔滨：新增 `task_type_mapping` 或 `task_groups` 配置，将旧任务映射到 5 类新专题。
   - 海淀：在 `regions.haidian.tasks` 下配置 5 类专题，指向空目录或占位目录。

2. **新增/修改 schema**
   - `TaskInfo` 增加 `task_type` 枚举或分组信息。
   - 确保 `TasksResponse` 返回新专题列表。

3. **修改 `app/services/data_service.py`**
   - 支持按新专题 ID 解析数据路径。
   - 对哈尔滨，通过映射表找到底层物理目录。
   - 对海淀，任务存在但数据缺失时优雅返回 404。

4. **修改 `app/routers/tasks.py`**
   - `list_tasks` 返回统一后的 5 类专题。
   - 对海淀返回 5 个任务，summary/result/label/prediction 返回 404。

5. **测试**
   - `test_list_tasks_harbin` 断言 5 类专题。
   - `test_list_tasks_haidian` 断言 5 类专题（之前为 0）。

### Phase 2: Get Task Result 严格按 patch 返回

1. **修改 `app/services/data_service.py` 的 `get_task_result_path`**
   - 移除 `format_type == "png"` 分支中的 `_find_first_file(base, "*.png")` 和 `result.png` fallback。
   - 仅保留 per-patch 路径解析：
     - v1: `{base}/{patch_id}_{period}.png`
     - v2: `{base}/{period}/{patch_id}.png`
   - 如果确实需要整图 mosaic，返回 None，由调用方处理。

2. **新增 mosaic 端点（可选）**
   - `GET /regions/{region_id}/tasks/{task_type}/mosaic?period=...&version=...`
   - 读取 `{base}/{period}/{period}.png` 或 `{base}/mosaic_{period}.png`。

3. **修改 `app/routers/tasks.py` 的 `get_task_result`**
   - 当 `get_task_result_path` 返回 None 时，不再尝试 tile fallback 之外的其他回退。
   - 保持 tile fallback 用于兼容已有的 per-patch tiles。

4. **测试**
   - 新增测试：请求 patch_000000 的 result，不应返回整图 result.png。
   - 验证现有测试仍通过。

### Phase 3: 自定义训练 + 批量推理 API

1. **新增 `app/services/model_registry.py`**
   - 模型 CRUD、索引 JSON 持久化。
   - 用户隔离（默认 `default`）。

2. **新增 `app/services/training_engine.py`**
   - 移植 `xuannv_show/backend/app/services/annotate/training_engine.py`。
   - 适配 embedding-api 的 embedding 路径格式（harbin 按月子目录）。

3. **新增 `app/services/cd_training_engine.py`**
   - 移植 `xuannv_show/backend/app/services/annotate/cd_training_engine.py`。
   - 适配 harbin 双期 embedding 路径。

4. **新增 `app/services/inference_engine.py`**
   - 移植 `xuannv_show/backend/app/services/annotate/inference_engine.py`。
   - 支持分类头和变化检测头。
   - 支持批量推理。

5. **新增 `app/routers/models.py`**
   - 实现 3.2 中的端点。
   - 使用 `BackgroundTasks` 做异步训练。

6. **扩展 `app/schemas/models.py`**
   - 增加 `ModelCreate`、`ModelOut`、`InferRequest`、`BatchInferRequest` 等 schema。

7. **在 `app/main.py` 中注册 router**
   - `app.include_router(models.router, prefix="/models", tags=["models"])`

8. **测试**
   - 单元测试：mock 训练/推理，验证接口返回。
   - 集成测试：使用小量真实数据训练并推理（标记 `@pytest.mark.slow`）。

### Phase 4: 文档更新

1. 更新 `README.md`：
   - 5 类专题说明。
   - 新增 `/models` API 使用示例。
2. 更新 `docs/API.md`：
   - 新端点详细说明。
   - 训练/推理流程。
3. 更新 `.gitignore`：
   - 忽略 `users/` 目录下的模型和结果文件。

### Phase 5: 验证与部署

1. 本地跑 `python -m pytest tests/ -v -m "not slow"`。
2. 启动服务，curl 验证：
   - `/regions/harbin/tasks` 返回 5 类专题。
   - `/regions/haidian/tasks` 返回 5 类专题。
   - `/regions/harbin/patches/patch_000000/tasks/change_detection/result` 返回该 patch 结果。
3. 提交并 push 到 GitHub。
4. 服务器上 pull 最新代码并重启服务。

---

## 待用户决策

1. **旧任务 ID 是否保留？**
   - 选项 A：完全替换为 5 类新专题（推荐）。
   - 选项 B：保留旧任务 ID 作为别名，同时新增 5 类专题。

2. **哈尔滨现有任务如何精确映射？**
   - `construction` 是映射到 `building_extraction` 还是 `change_detection`？
   - `land_conversion` 是映射到 `land_use_classification` 还是 `change_detection`？

3. **训练数据来源？**
   - 是否复用 `xuannv_show` 中的用户标注数据（`users/{user_id}/masks/`）？
   - 还是需要新建一套标注存储？

4. **训练是同步还是异步？**
   - 推荐异步（BackgroundTasks），训练快时也可同步等待。

5. **批量推理上限？**
   - 同步批量建议限制 100 个 patch，避免 HTTP 超时。
   - 超过 100 个是否改为异步 Job？

6. **用户认证？**
   - 本次是否接入用户认证？
   - 如果不接入，所有用户共享 `default` 用户空间。

---

## 风险与注意事项

1. **向后兼容性**：任务 ID 变更会影响前端调用，需同步通知前端团队。
2. **数据路径**：海淀区任务暴露但无数据，需确保 404 信息清晰。
3. **模型存储**：训练生成的 `.pkl` 文件不应提交 Git，需更新 `.gitignore`。
4. **GPU/CPU**：当前训练使用 sklearn LogisticRegression，纯 CPU；后续如加入深度学习头，需考虑 GPU。
5. **并发训练**：BackgroundTasks 在单进程内运行，多个同时训练会串行；后续如需并发可考虑 Celery/RQ。

---

## 建议执行顺序

1. Phase 1（专题统一）+ Phase 2（result 按 patch）先做，改动相对独立。
2. Phase 3（训练/推理 API）后做，依赖前两个阶段的专题定义。
3. Phase 4 文档与 Phase 5 验证紧随。

请确认方案后执行。
