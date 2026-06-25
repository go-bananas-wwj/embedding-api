# 调研发现

## 1. 用户新需求整理（2026-06-25）

### 1.1 专题类型统一

哈尔滨新区与海淀区下游专题应统一暴露为 5 类：

| 专题中文名 | 建议 ID | 任务性质 | 输入 |
|------------|---------|----------|------|
| 变化检测 | `change_detection` | 像素级二分类（变化/未变化） | 双期 embedding |
| 土地覆盖分类 | `land_cover_classification` | 像素级多分类 | 单期 embedding |
| 土地利用分类 | `land_use_classification` | 像素级多分类 | 单期 embedding |
| 水体提取 | `water_extraction` | 像素级二分类（水体/背景） | 单期 embedding |
| 建筑物提取 | `building_extraction` | 像素级二分类（建筑/背景） | 单期 embedding |

> 海淀区当前仅做任务列表暴露，不挂载真实数据，后续再补齐。

### 1.2 Get Task Result 必须按 patch 返回

当前 embedding-api 的 `/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result` 已经以 `patch_id` 为路径参数，但实现中会回退到查找整幅 mosaic/summary 图片（见 `data_service.py` `_find_first_file(base, "*.png")` 与 `result.png` fallback）。需要移除整图回退，严格限定只返回该 patch 对应文件。

### 1.3 自定义训练 + 批量推理 API

参考 `/workspace/xuannv_show` 现有实现：

- **训练引擎**：
  - `training_engine.py`：基于用户标注 mask + embedding，训练 sklearn LogisticRegression 分类头，保存 `.pkl`。
  - `cd_training_engine.py`：基于双期 embedding 差值 + 变化 mask，训练变化检测头。
- **推理引擎**：
  - `inference_engine.py`：单 patch 推理，输出 256×256 PNG。
  - `system_models.py`：系统预训练模型（worldcover / dynamic_world / jrc_water / building_extraction）。
- **API 形态**：
  - `POST /annotate/models` 创建模型并后台训练，返回 `job_id`。
  - `GET /annotate/train/{job_id}` 查询训练状态。
  - `POST /annotate/models/{model_id}/infer` 单 patch 推理。
  - `POST /annotate/models/{model_id}/infer_batch` 批量推理。

需要在 embedding-api 中复用或移植上述能力，设计为面向 region/patch 的 REST API。

---

## 2. 行业最佳实践

### 2.1 遥感 AI 服务化

- **REST on dedicated server** 是遥感深度学习模型最常见的服务化方式（FastAPI/Flask + EC2/裸金属），参考 `satellite-image-deep-learning/model-training-and-deployment`。
- 对超大影像推荐按 patch/tile 切分后逐个推理，而非整图载入，避免内存爆炸。
- 推理结果通常以 PNG/GeoTIFF 形式返回，前端叠加到地图。

### 2.2 自定义训练 / Fine-tuning API

- 主流平台（OpenAI、Databricks、SageMaker、阿里云 Model Studio）均采用 **异步 Job 模式**：
  - `POST /jobs` 提交训练任务，立即返回 `job_id`。
  - `GET /jobs/{job_id}` 轮询状态（`queued` / `running` / `succeeded` / `failed`）。
  - 训练完成后通过状态端点或回调获取模型 ID/路径。
- 优点：训练可能耗时数秒到数分钟，同步 HTTP 会超时；异步 Job 模式更稳定，且支持失败重试。

### 2.3 批量推理 API

- 行业常见两种形态：
  1. **同步批量**：`POST /infer_batch` 传入 `patch_ids` 列表，服务端逐个推理后统一返回 URL 列表（适合 10~100 个 patch）。
  2. **异步批量**：`POST /batch_jobs` 上传 JSONL 或 patch 列表，返回 `job_id`；`GET /batch_jobs/{job_id}` 查询进度并下载结果（适合 1000+ patch）。
- 本项目 patch 数量在数百量级，先采用 **同步批量** 即可，后续可扩展为异步。

### 2.4 模型注册表

- 为每个用户/租户维护独立的模型目录和索引 JSON：
  - `users/{user_id}/models/{model_id}.pkl`
  - `users/{user_id}/models_index.json`
- 支持 CRUD：列出、创建、重命名、删除模型。

---

## 3. 当前代码现状

### 3.1 任务配置

`config.yaml` 哈尔滨当前任务：
- `construction`, `building_change`, `farmland`, `land_conversion`, `demolition`, `change_detection`

海淀区当前任务：`{}`（空）。

### 3.2 任务接口

- `GET /regions/{region_id}/tasks`：列出任务
- `GET /regions/{region_id}/tasks/{task_type}/summary`：任务汇总
- `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result?format=png|npy`：per-patch 结果
- `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/prediction`：per-patch 预测
- `GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/label`：per-patch 标签

问题：`result` 接口存在 summary/mosaic 回退，不严格限定 patch。

### 3.3 训练/推理

当前 embedding-api 无自定义训练/推理 API，需新增模块。

