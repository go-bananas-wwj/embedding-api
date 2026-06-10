# Embedding 数据整理与 API 升级计划

## 背景

当前 API 的 embedding 接口存在以下问题：
1. **没有时间信息**：只返回单个 64×64 PNG，无法按月获取 embedding
2. **格式单一**：harbin 只有 PNG，haidian 只有 NPY，不统一
3. **数据来源**：原始 NPY embedding 分散在外部路径，未纳入项目管理

用户要求：将 V4 版（→ API v1）和 V5 版（→ API v2）按月组织的 embedding 纳入项目，支持按月获取。

---

## 调研发现

### 磁盘上已存在完整数据（无需重新下载）

| 版本 | 位置 | 文件数 | 月份覆盖 | shape | dtype |
|------|------|--------|----------|-------|-------|
| **V4** | `/workspace/raw/xuannv_modelscope_upload/embeddings/v4_official/monthly_embeddings_2025/` | 2,121 | 2025-04, 06, 08, 09, 10 | (128, 64, 64) | float32 |
| **V5-2025** | `/workspace/raw/xuannv_modelscope_upload/embeddings/v5_mixed_scale/monthly_embeddings_2025/` | 2,121 | 2025-04, 06, 08, 09, 10 | (128, 64, 64) | float32 |
| **V5-2026** | `/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2026/` | 2,121 | 2026-01, 02, 03, 04, 05 | (128, 64, 64) | float32 |

> **结论**：磁盘上已同时存在 V4 和 V5 的完整月频 embedding，**不需要从 ModelScope 重新下载**。如用户要求强制重新下载，可通过 `git clone` 或 `modelscope` SDK 获取（当前 `datasets` 库有版本冲突，需额外处理）。

### 当前 API embedding 配置

```yaml
# 当前（只有 PNG，无时间信息）
embeddings:
  v2:
    path: "data/harbin/embeddings"
    template: "{patch_id}.png"
    formats: ["png"]
```

---

## 方案选项

### 方案 A：直接使用已有数据（推荐，快速）
- **操作**：用 `cp` 或 `rsync` 将磁盘上已有的 V4/V5 NPY 复制到 `embedding-api/data/harbin/embeddings/v1/` 和 `v2/`
- **优点**：无需下载，速度快（~5-10 分钟），数据完整
- **缺点**：数据版本与磁盘上当前版本一致，如需 ModelScope 最新版需先更新磁盘数据

### 方案 B：从 ModelScope 重新下载
- **操作**：使用 ModelScope SDK 或 `git lfs clone` 从用户提供的链接下载
- **数据集 ID**：`WeijieWu/xuannv_embddings`
- **子目录**：`outputs/aef_qwen_v4_official` / `outputs/aef_qwen_v5_mixed_scale`
- **Token**：`ms-399d1804-1cb3-446a-a3f7-dfc4dc70d977`
- **优点**：确保数据最新
- **缺点**：
  - 当前环境 `modelscope` 的 `datasets` 依赖有版本冲突（`ImportError: cannot import name 'Dataset'`）
  - 数据集总量约 1GB+，下载耗时取决于网络
  - 需额外时间修复依赖冲突

---

## 推荐执行方案：A + 目录重构 + API 升级

### 新目录结构

```
data/harbin/
├── patches_meta.json
├── embeddings/
│   ├── v1/                          # V4 版 → API v1
│   │   ├── 2025-04/
│   │   │   ├── patch_000000.npy
│   │   │   └── ... (424 patches)
│   │   ├── 2025-06/
│   │   ├── 2025-08/
│   │   ├── 2025-09/
│   │   └── 2025-10/
│   └── v2/                          # V5 版 → API v2
│       ├── 2025-04/
│       ├── 2025-06/
│       ├── 2025-08/
│       ├── 2025-09/
│       ├── 2025-10/
│       ├── 2026-01/
│       ├── 2026-02/
│       ├── 2026-03/
│       ├── 2026-04/
│       └── 2026-05/
└── tasks/...
```

> **文件名处理**：原始文件名为 `patch_{id}_{YYYY-MM}.npy`，复制到按月子目录后简化为 `patch_{id}.npy`，月份信息由目录层级体现。

### 数据量估算

| 版本 | 月份数 | patches/月 | 单文件大小 | 总计 |
|------|--------|------------|-----------|------|
| V4 (v1) | 5 | 424 | ~128KB | ~270MB |
| V5-2025 (v2) | 5 | 424 | ~128KB | ~270MB |
| V5-2026 (v2) | 5 | 424 | ~128KB | ~270MB |
| **合计** | | | | **~810MB** |

### API 接口升级

**1. 新增 `month` 查询参数**

```
GET /regions/{region_id}/patches/{patch_id}/embedding?format=npy&version=v1&month=2025-04
```

| 参数 | 说明 |
|------|------|
| `version` | `v1` = V4 版, `v2` = V5 版 |
| `month` | 月份，格式 `YYYY-MM`，如 `2025-04` |
| `format` | `npy` = 原始 embedding, `png` = PCA-RGB 预览图, `json` = 统计信息 |

**2. config.yaml 更新**

```yaml
embeddings:
  v1:
    path: "data/harbin/embeddings/v1"
    template: "{month}/{patch_id}.npy"
    formats: ["npy", "png", "json"]
    available_months: ["2025-04", "2025-06", "2025-08", "2025-09", "2025-10"]
  v2:
    path: "data/harbin/embeddings/v2"
    template: "{month}/{patch_id}.npy"
    formats: ["npy", "png", "json"]
    available_months: ["2025-04", "2025-06", "2025-08", "2025-09", "2025-10",
                       "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
```

**3. `format=png` 动态生成**

当请求 `format=png` 时，从 NPY 实时生成 PCA-RGB 预览图（复用 `generate_embedding_tiles.py` 的逻辑）：
- 加载 NPY → PCA 降维到 3 通道 → 归一化 → 保存为 PNG
- 或预生成 PNG 缓存到 `data/harbin/embeddings/v1/{month}/{patch_id}.png`

**4. Patch 列表扩展**

在 `GET /patches` 响应中，为每个 patch 添加 `available_months` 字段：

```json
{
  "patch_id": "patch_000000",
  "has_embedding": true,
  "available_months": ["2025-04", "2025-06", "2025-08", "2025-09", "2025-10"]
}
```

---

## 执行步骤

### Phase 1: 数据复制（使用已有数据）
1. 创建 `data/harbin/embeddings/v1/` 和 `v2/` 目录结构
2. 复制 V4 NPY → `v1/{month}/{patch_id}.npy`
3. 复制 V5-2025 NPY → `v2/{month}/{patch_id}.npy`
4. 复制 V5-2026 NPY → `v2/{month}/{patch_id}.npy`

### Phase 2: API 代码修改
1. 修改 `data_service.py` 的 `get_embedding_path`
   - 支持 `month` 参数
   - 支持 `version` 参数（v1/v2）
   - 支持动态 PNG 生成
2. 修改 `embeddings.py` 路由
   - 添加 `month` Query 参数
   - 添加 `version` Query 参数
   - `format=png` 时动态生成或返回预生成 PNG
3. 修改 `patches.py` 路由
   - 响应中添加 `available_months`

### Phase 3: config.yaml 更新
- 更新 embedding 配置，添加 v1/v2 和 available_months

### Phase 4: 验证
- pytest 全部通过
- 调用各接口确认返回正确
- 验证 `/docs` 和 `/openapi.json` 正常

### Phase 5: 文档更新
- 更新 API.md，添加 `month` 参数和 `version` 参数说明
- 添加 embedding 按月获取的示例

---

## 待用户决策

1. **数据来源**：
   - [ ] **方案 A**：直接使用磁盘上已有数据（推荐，快速）
   - [ ] **方案 B**：从 ModelScope 重新下载（需处理依赖冲突，耗时更长）

2. **PNG 生成策略**：
   - [ ] **动态生成**：请求时实时从 NPY 生成 PNG（CPU 开销，无需额外空间）
   - [ ] **预生成**：批量生成所有 PNG 缓存（占用额外 ~100MB 空间，响应更快）

3. **是否删除旧 embedding 数据**：
   - [ ] 删除 `data/harbin/embeddings/` 下旧的 64×64 PNG（已无用）
   - [ ] 保留作为备份

请确认后执行。
