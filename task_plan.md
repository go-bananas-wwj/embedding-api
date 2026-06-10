# 数据路径整理与统一迁移计划

## 目标
将分散在多个项目路径下的模型文件、静态数据（embeddings、results、predictions、labels、label_vis）统一整理并复制到 `embedding-api` 项目目录下，便于独立部署和版本管理。

## 当前数据分布调研

### 哈尔滨新区 (harbin)
| 数据类型 | 当前路径 | 大小 | 说明 |
|----------|----------|------|------|
| patches_meta | `/workspace/xuannv_show/data/harbin/patches_meta.json` | 256K | patch 元数据 |
| embeddings | `/workspace/xuannv_show/data/harbin/embeddings/v2` | 1.6M | PNG 可视化图 (64×64) |
| labels | `/workspace/xuannv_show/data/harbin/labels/{task}` | 900K | v1 二值标签 |
| labels_v2 | `/workspace/xuannv_show/data/harbin/labels_v2/{task}/{period}` | 828K | v2 二值标签 |
| label_vis | `/workspace/xuannv_show/data/harbin/label_vis/{task}` | 2.3M | v1 标签可视化图 |
| label_vis_v2 | `/workspace/xuannv_show/data/harbin/label_vis_v2/{task}/{period}` | 5.7M | v2 标签可视化图 |
| results | `/workspace/xuannv_show/data/harbin/results/{task}` | 8.6M | v1 结果图 + tiles |
| results_v2 | `/workspace/xuannv_show/data/harbin/results_v2/{task}/{period}` | 12M | v2 结果图 + tiles |
| predictions | `/workspace/xuannv_show/data/harbin/predictions/{task}` | 25M | v1 预测 NPY |
| predictions_v2 | `/workspace/xuannv_show/data/harbin/predictions_v2/{task}/{period}` | 50M | v2 预测 NPY |
| models | `/workspace/xuannv_show/data/harbin/models/{task}` | 242M | v1 任务头模型 |
| models_v2 | `/workspace/xuannv_show/data/harbin/models_v2/{task}` | 161M | v2 任务头模型 |
| **harbin 小计** | | **~508MB** | |

### 海淀区 (haidian)
| 数据类型 | 当前路径 | 大小 | 说明 |
|----------|----------|------|------|
| patches_meta_v2 | `/workspace/olmo/data/haidian/patches_meta_v2.json` | 待确认 | patch 元数据 |
| embeddings | `/workspace/olmo/data/haidian/aef_embeddings/haidian_2025_patches` | 1.3G | NPY embedding (64,128,128) |
| olmoearth | `/workspace/olmo/data/haidian/olmoearth` | 待确认 | 原始多源影像数据 |
| **haidian 小计** | | **~1.3GB+** | |

### 原始影像（可选）
| 数据类型 | 当前路径 | 大小 | 说明 |
|----------|----------|------|------|
| S2 影像 | `/workspace/raw/harbin_scenes/s2` | 6.8G | 424 个 patch 的 Sentinel-2 时间序列 |

## 拟议的统一目录结构

```
embedding-api/
├── .gitignore                    # 排除 data/ 和 models/
├── config.yaml                   # 更新为相对路径
├── data/                         # 所有静态数据（不提交 Git）
│   ├── harbin/
│   │   ├── patches_meta.json
│   │   ├── embeddings/
│   │   └── tasks/
│   │       ├── construction/
│   │       │   ├── v1/
│   │       │   │   ├── results/
│   │       │   │   ├── predictions/
│   │       │   │   ├── labels/
│   │       │   │   └── label_vis/
│   │       │   └── v2/
│   │       │       ├── results/
│   │       │       ├── predictions/
│   │       │       ├── labels/
│   │       │       └── label_vis/
│   │       ├── building_change/v1/
│   │       ├── farmland/v1/
│   │       └── land_conversion/v2/
│   └── haidian/
│       ├── patches_meta_v2.json
│       └── embeddings/
├── models/                       # 模型文件（不提交 Git）
│   └── harbin/
│       ├── v1/...
│       └── v2/...
└── docs/...
```

## 迁移方案选项

### 方案 A：完整迁移（推荐用于独立部署）
- **复制内容**：harbin 全部数据 + haidian embeddings + patches_meta
- **不复制**：S2 原始影像（6.8GB，仅用于 label_vis 生成，非 API 运行时必需）
- **磁盘占用**：约 2GB
- **优点**：项目可独立部署，不依赖外部路径
- **缺点**：占用更多磁盘，需手动同步后续更新

### 方案 B：仅迁移 API 依赖数据
- **复制内容**：embeddings + results + predictions + labels + label_vis + patches_meta
- **不复制**：models（除非 API 需要在线推理）、S2 影像
- **磁盘占用**：约 350MB
- **优点**：最小化占用，满足当前 API 需求
- **缺点**：模型文件仍分散，如需模型推理需单独处理

### 方案 C：使用符号链接（快速但依赖外部）
- **操作**：在 embedding-api 下创建指向外部数据的软链接
- **优点**：不占用额外空间，数据实时同步
- **缺点**：未解决"统一管理"问题，部署时仍需外部路径

## 统一目录结构设计

```
embedding-api/
├── data/                    # 静态数据（不提交 Git）
│   ├── harbin/
│   │   ├── patches_meta.json
│   │   ├── embeddings/
│   │   └── tasks/
│   │       ├── construction/v1, v2
│   │       ├── building_change/v1
│   │       ├── farmland/v1
│   │       └── land_conversion/v2
│   └── haidian/
│       ├── patches_meta_v2.json
│       └── embeddings/
├── models/                  # 模型权重（不提交 Git）
│   └── harbin/
│       ├── v1/{task}/
│       └── v2/{task}/
└── pipelines/               # 复现/运行程序（提交 Git）
    ├── harbin/
    │   ├── README.md
    │   ├── _paths.py
    │   ├── generate_embedding_tiles.py
    │   ├── inference_task_head.py / v2
    │   ├── generate_prediction_tiles.py / v2
    │   ├── shp_to_patch_masks.py / v2
    │   ├── visualize_labels_v2.py
    │   ├── train_task_head.py / v2
    │   └── requirements.txt
    └── haidian/
        ├── README.md
        ├── batch_produce_embeddings.py
        ├── extract_olmoearth_embeddings.py
        └── requirements.txt
```

## 推荐执行步骤

1. **Phase 1: 准备**
   - 更新 `.gitignore` 排除 `data/`、`models/`
   - 创建目标目录结构

2. **Phase 2: 复制 harbin 数据**
   - 复制 patches_meta.json
   - 复制 embeddings
   - 按任务/版本复制 results、predictions、labels、label_vis

3. **Phase 3: 复制 haidian 数据**
   - 复制 patches_meta_v2.json
   - 复制 embeddings

4. **Phase 4: 复制模型**
   - 复制 harbin models v1/v2

5. **Phase 5: 复制复现程序**
   - 复制 harbin 关键脚本（embedding 生成、推理、瓦片生成、标签可视化）
   - 复制 haidian 关键脚本（embedding 生成、预处理）
   - 为每个区域创建 README.md 说明运行流程

6. **Phase 6: 更新 config.yaml**
   - 所有路径改为 embedding-api 内部相对路径

7. **Phase 7: 验证**
   - 运行 pytest
   - 调用各接口确认返回正确
   - 重启服务

8. **Phase 8: 文档更新**
   - 更新 API.md 中关于数据路径的说明
   - 添加 pipelines README 说明复现流程

## 待用户决策

1. **选择方案**：A（完整迁移）、B（仅 API 数据）、C（软链接）
2. **是否复制 S2 原始影像**（6.8GB）
3. **是否复制模型文件**（~400MB）
4. **是否保留原位置数据**（推荐保留，作为备份）

## 风险与注意事项

- 复制过程可能耗时（2GB 数据约 5-15 分钟）
- 数据不提交 Git，部署时需要在目标环境重新准备
- 如果后续 xuannv_show/olmo 数据更新，需要手动同步到 embedding-api
