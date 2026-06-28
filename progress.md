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
