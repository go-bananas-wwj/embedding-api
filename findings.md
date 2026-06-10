# 数据路径调研发现

## 关键发现

### 1. 数据来源高度分散
当前 API 依赖的数据分布在 3 个完全不同的项目路径下：
- `/workspace/xuannv_show/data/harbin/` — 哈尔滨任务数据
- `/workspace/olmo/data/haidian/` — 海淀 embedding 数据
- `/workspace/raw/harbin_scenes/s2/` — S2 原始影像

### 2. 数据量级
- harbin 任务数据：~508MB
- haidian embedding：~1.3GB
- S2 原始影像：~6.8GB
- **总计约 8.6GB**（不含 haidian 原始影像）

### 3. 目录结构不一致
- v1 数据多为 flat 结构：`{task}/{patch_id}.{ext}`
- v2 数据多为 period 子目录：`{task}/{period}/{patch_id}.{ext}`
- label_vis 生成后按 v1/v2 分开存放

### 4. 模型文件也存在
- `/workspace/xuannv_show/data/harbin/models/` 242MB
- `/workspace/xuannv_show/data/harbin/models_v2/` 161MB
- 当前 API 不直接服务模型文件，但未来可能用于在线推理

### 5. Git 仓库现状
- embedding-api 当前仅 12MB
- 若复制全部数据，将增长至约 8.6GB
- 数据文件不应提交 Git，需更新 `.gitignore`
