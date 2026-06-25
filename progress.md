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

### 下一步

等待用户确认 `task_plan.md` 中的方案与待决策问题，再开始执行。
