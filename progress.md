> **注意**：本文件不是 `git push` 目标，只是当前工作进度记录。

# 进度日志

## 2026-06-25

### 已完成的修复（demolition + v2 available_tasks）

- 修复 `config.yaml` 缺失 `demolition` 任务。
- 修复 `get_available_tasks` 无法识别 v2 period 子目录的问题。
- 补充测试用例，覆盖任务列表和 `available_tasks`。
- 更新 `README.md` 文档。
- 已推送 GitHub：`6aacd99..c9e2e93`。
- 公网 API 验证通过：`/regions/harbin/tasks` 返回 6 个任务。

### 新需求：下游任务重构

用户提出三方面改造：
1. 哈尔滨与海淀区专题类型统一为 5 类：变化检测、土地覆盖分类、土地利用分类、水体提取、建筑物提取。
2. `Get Task Result` 必须严格返回对应区域、对应 patch 的结果，不能返回整张大图。
3. 参考 `xuannv_show` 提供自定义训练分类头和批量推理 API。

### 已完成调研

- 梳理 `xuannv_show/backend/app/routers/annotate.py` 及相关训练/推理引擎。
- 搜索行业最佳实践：遥感 AI REST API、异步训练 Job、批量推理 API、模型注册表。
- 编写 `findings.md` 记录调研结果。
- 编写 `task_plan.md` 详细实施计划。

: 用户已确认关键决策；开始执行重构。
- 完成 Phase 1-7：专题统一、per-patch result、认证、标注、训练/推理 API、系统模型。
- 更新 README.md 和 docs/API.md 文档。
- 全部非 slow 测试通过（80 passed）。

### 下一步

提交 Phase 8 文档更新，完成最终验证并推送。


## 2026-06-26 前端反馈处理

- 收到前端同事三个问题：POST /models 参数冗余、infer_batch 月份参数混淆、新增大图接口。
- 已分析当前代码与数据现状，整理到 `findings.md`。
- 已编写 `task_plan.md` 前端反馈处理计划，包含三个待决策问题及可选方案。
- 等待用户确认方案后执行。


## 2026-06-28 前端反馈执行完成

- 已按用户确认的决策完成三项前端反馈处理：
  1. 保留 `model_type` + `task_type`，优化 Swagger 示例和字段描述。
  2. `infer_batch` 保持 XOR 校验，Swagger 提供分类/变化检测两套示例。
  3. 新增 `GET /regions/{region_id}/mosaic` 整区域马赛克大图接口（当前仅支持 Sentinel-2）。
- 新增 `app/services/mosaic_service.py`、`tests/test_mosaic.py`。
- 更新 `README.md`、`docs/API.md`。
- 全量测试通过：`99 passed, 1 skipped`。
- 已推送 GitHub：`6e6e4c2`。
- 服务已重启，`/regions/harbin/mosaic?date=2025-04&sensor_type=s2&version=v2&format=png` 返回 4.9MB PNG。


## 2026-06-28 mosaic 接口扩展为 S2/S1/Landsat

- 发现 `/workspace/raw/harbin/` 下已有 S2、S1、Landsat 的原始 per-patch TIFF 数据。
- 重写 `app/services/mosaic_service.py`：
  - 从 `/workspace/raw/{region}/{sensor}/{patch_id}/{period}.tif` 读取原始数据。
  - `date=YYYY-MM` 自动映射到季度文件名（如 `2025-04` → `2025Q2`）。
  - 支持 `sensor_type=s2|s1|landsat`。
  - PNG 输出使用标准波段合成：S2/Landsat 真彩色（B4/B3/B2），S1 伪彩色（R=VV, G=VH, B=VH/VV）。
  - GeoTIFF 输出保留原始多波段和 UTM 坐标。
  - 新增 `patch_ids` 参数，可只拼指定 patch。
- 更新 `app/routers/regions.py` 的 `/regions/{region_id}/mosaic` 端点。
- 更新 `tests/test_mosaic.py`，用 5 个 patch 子集提速；GeoTIFF 测试标记为 `slow`。
- 更新 `README.md`、`docs/API.md` 说明多传感器支持和波段合成。
- 全量非 slow 测试通过：`96 passed, 5 deselected`。
- 已推送 GitHub：`38f0792`。
- 服务已重启，验证通过：
  - S2 全区域 `15.0 MB` PNG
  - S1 两 patch 预览 `90 KB` PNG
  - Landsat 两 patch 预览 `11 KB` PNG


## 2026-06-28 前端文档与 API 测试报告更新

- 同步 `test_output_agent/test_api.py`：
  - 任务配置改为 5 类新专题（change_detection/building_extraction/land_use_classification/land_cover_classification/water_extraction）。
  - 新增 `test_mosaic_endpoints()` 覆盖 s2/s1/landsat。
  - 新增 `test_model_endpoints()` 覆盖 `/models` 列表。
  - 报告结论更新为当前接口现状。
- 重新运行 `test_output_agent/test_api.py`，生成 `test_output_agent/report.md`：
  - 总测试 101 项，通过 97 项，预期内失败 4 项（change_detection 部分 result/prediction 数据缺失），非预期异常 0 项。
- 单元测试：`pytest -q -m "not slow"` → `96 passed, 5 deselected`。
- 确认 `docs/API.md` 已包含 `/regions/{region_id}/mosaic`、自定义模型、训练工作流等完整中文说明与 curl 示例。
- 提交并推送。


## 2026-06-28 mosaic 接口补充具体示例与参数取值表

- 更新 `app/routers/regions.py`：
  - 为 `region_id`、`date`、`sensor_type`、`format`、`patch_ids` 添加中文描述与 `openapi_examples`。
  - 明确 `sensor_type` 可取 `s2/s1/landsat`，`format` 可取 `png/tif`，`version` 可留空。
- 更新 `docs/API.md`：
  - 新增「参数取值表」和「`date` 与季度文件映射表」。
  - 给出 4 条完整 curl 示例（全区域 S2 PNG、S1 两 patch 预览、Landsat GeoTIFF、本地调试）。
- 更新 `test_output_agent/test_api.py`：
  - 在 `report.md` 中新增「Mosaic 接口调用示例」与参数取值说明。
- 将 `test_output_agent/test_api.py` 与 `report.md` 加入 Git 跟踪（原 `.gitignore` 忽略整个目录）。
- 重新运行测试：
  - `test_api.py` → `97 passed, 4 failed (expected-ish), 0 unexpected`。
  - `pytest -q -m "not slow"` → `96 passed, 5 deselected`。
- 重启 watchdog 服务，Swagger 已刷新。
- 已推送 GitHub：`c213a82`。


## 2026-06-28 支持系统预设模型走统一推理入口

- 前端同事反馈：想把系统预训练模型 ID 传给 `/models/{model_id}/infer` 做推理。
- 问题总结：系统模型原来只有独立的 `/system-models/{task_id}/infer`，前端需要维护两套调用逻辑。
- 实施方案（统一推理入口）：
  1. `app/schemas/models.py`：
     - 给 `ModelOut` 增加 `source`（`custom`/`system`）和 `versions` 字段。
     - 给 `InferRequest` / `BatchInferRequest` 增加 `version` 字段（默认 `v2`）。
  2. `app/services/system_model_service.py`：
     - 新增 `is_system_task()`、`get_system_model_info()` 辅助函数。
     - 将系统模型输出尺寸从 `256×256` 统一为 `128×128`。
  3. `app/routers/models.py`：
     - `/models` 列表传入 `region_id` 时合并系统模型。
     - `/models/{model_id}` 支持系统任务 ID。
     - `/models/{model_id}/infer` 与 `/models/{model_id}/infer_batch` 优先查找自定义模型，找不到则按系统任务 ID 调用 `infer_system_model`，返回 `/system-models/results/{filename}`。
     - 更新 `model_id` 参数描述与推理请求示例，加入系统模型示例。
  4. `docs/API.md`：更新“列出模型”“获取模型详情”“单 Patch 推理”“批量推理”章节，说明系统模型用法与 `result_url` 路径差异。
  5. `tests/test_models.py`：新增 4 个用例覆盖系统模型列表、详情、单 Patch 推理、批量推理。
  6. `test_output_agent/test_api.py`：新增系统模型统一推理测试，并支持 POST JSON body。
- 验证结果：
  - `test_api.py` → `101 passed, 4 failed (expected-ish), 0 unexpected`。
  - `pytest -q -m "not slow"` → `100 passed, 5 deselected`。
- 重启 watchdog 服务，接口已生效。
- 已推送 GitHub：`12f2e1a`。


## 2026-06-28 合并 Haidian V1 PR 并准备哈尔滨 ModelScope 资产上传

- 合并 GitHub PR #1（Haidian V1 P2A 支持 + ModelScope 下载流程）：
  - 使用 `git fetch origin pull/1/head:pr-haidian-v1` + `git merge --no-ff` 安全合并。
  - 合并后未重启服务，当前后端继续运行。
- 新增哈尔滨资产上传/下载流水线：
  - `pipelines/harbin/paths.py`：集中定义哈尔滨本地源路径与 ModelScope 默认仓库/前缀。
  - `pipelines/harbin/prepare_harbin_api_assets.py`：把 `data/harbin`、`models/harbin`、`models/sam3`、`/workspace/raw/harbin`、`/workspace/raw/harbin_scenes` 打包到 `api_ready`，自动生成 `manifest.json` 与 `checksums.sha256`。
  - `pipelines/harbin/download_modelscope_assets.py`：从 ModelScope 下载并按真实文件系统路径展开（含 `/workspace/raw/...`）。
- 上传范围（约 31 GB，149,498 个文件）：
  - `data/harbin`：12.59 GB
  - `models/harbin`：422 MB
  - `models/sam3`：3.22 GB
  - `/workspace/raw/harbin`：2.80 GB
  - `/workspace/raw/harbin_scenes`：11.94 GB
- 已用 Token 对数据集 `WeijieWu/xuannv_embdding_api` 做写入测试，权限正常。
- 文档更新：
  - `README.md` 新增「下载哈尔滨 V1/V2 资产」小节。
  - 新增 `pipelines/harbin/README.md` 说明打包/下载/校验流程。
- 海淀 V1 数据**不下载到本机**，避免磁盘/内存不足。
- 第一次直接上传目录树失败：ModelScope 单目录文件数超过 50,000 限制。
- 改用 tar 归档方案：
  - `data_harbin.tar` (12.59 GB)
  - `models_harbin.tar` (422 MB)
  - `models_sam3.tar` (3.22 GB)
  - `raw_harbin.tar` (2.80 GB)
  - `raw_harbin_scenes.tar` (12.12 GB)
  - 共 5 个归档 + `manifest.json` + `checksums.sha256`。
- 正式上传到 `WeijieWu/xuannv_embdding_api` 的 `harbin/v1/api_ready`：
  - 7 个文件，总计 31.14 GB，耗时约 6 分钟，全部成功。
- 上传后校验：
  - 从 ModelScope 下载 `manifest.json` 与 `checksums.sha256`。
  - 与本地生成的文件对比，内容一致。
- 代码变更已推送 GitHub：`fe9ae22`。
- 当前后端服务未重启、未受影响。


## 2026-06-29 在本机部署海淀区 V1 embedding 服务

- 目标：在不影响哈尔滨新区服务的前提下，使 `GET /regions/haidian/patches/{patch_id}/embedding` 可用。
- 保留哈尔滨 staging `/workspace/modelscope_upload/harbin/v1` 不清理（磁盘空间足够）。
- 新增 `pipelines/haidian/download_embeddings.py`：
  - 仅从 ModelScope 下载 `data/haidian/embeddings/v1/**` 与 `data/haidian/patches_meta_v1.json`。
  - Token 从 `MODELSCOPE_TOKEN` 环境变量读取。
- 下载结果：
  - 5,760 个文件，约 7.57 GB。
  - 月份覆盖 `202512` ~ `202605`。
- 重启 watchdog 服务加载 PR #1 中新增的 Haidian V1 配置：
  - 重启后 `/health` 正常返回，regions 包含 `harbin` 和 `haidian`。
  - 哈尔滨接口验证正常：`/regions/harbin/patches/patch_000000/embedding` 返回 200。
  - 海淀接口验证正常：
    - JSON：`/regions/haidian/patches/patch_000000/embedding?format=json&version=v1&month=202512` 返回 shape `[64,128,128]`。
    - PNG：`/regions/haidian/patches/patch_000000/embedding?format=png&version=v1&month=202512` 返回 `image/png`。
- 限制：
  - 仅启用海淀区 embedding 查询接口。
  - 专题任务结果与 SAM3 分割需要额外下载完整资产，本次因空间/范围限制未下载。
- 文档更新：`README.md` 增加「仅部署 embedding 接口」说明。
- 已推送 GitHub：`89944e0`。


## 2026-06-29 更新 ModelScope 数据集 README

- 参考数据集原有海淀区 V1 说明结构，补齐哈尔滨新区 V1/V2 资产说明。
- 更新后的 README 已上传至 `WeijieWu/xuannv_embdding_api/README.md`。
- 新增内容包括：
  - 哈尔滨新区 V1/V2 版本信息、时间范围、ModelScope 路径。
  - 哈尔滨新区 tar 归档结构说明（`data_harbin.tar`、`models_harbin.tar`、`models_sam3.tar`、`raw_harbin.tar`、`raw_harbin_scenes.tar`）。
  - 哈尔滨新区支持的接口与任务（embedding、变化检测、建筑物提取、土地利用分类、土地覆盖分类、水体提取）。
  - 哈尔滨新区快速部署命令（完整资产 + embedding-only）。
  - 使用 ModelScope CLI 直接下载哈尔滨资产的示例。
  - API 快速检查命令（包含哈尔滨示例）。
- 验证：重新下载 README.md，确认哈尔滨新区相关内容已写入。
