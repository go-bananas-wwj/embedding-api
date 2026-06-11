# Harbin 区域复现程序

本目录包含哈尔滨新区遥感监测任务的完整复现程序，覆盖 embedding 可视化、任务头推理、标签生成和结果可视化。

## 目录结构

```
pipelines/harbin/
├── paths.py                    # 统一路径管理（基于 embedding-api 项目根目录）
├── generate_embedding_tiles.py # 将 NPY embedding 转为 PCA-RGB PNG 预览图
├── train_task_head.py          # v1 任务头训练（单期 embedding）
├── train_task_head_v2.py       # v2 任务头训练（两期 embedding 差分）
├── inference_task_head.py      # v1 任务头推理 → predictions
├── inference_task_head_v2.py   # v2 任务头推理 → predictions
├── generate_prediction_tiles.py    # v1 从 predictions 生成 result tiles
├── generate_prediction_tiles_v2.py # v2 从 predictions 生成 result tiles
├── shp_to_patch_masks.py       # v1 从 shapefile 生成 labels
├── shp_to_patch_masks_v2.py    # v2 从 shapefile + Excel 生成 labels
└── visualize_labels_v2.py      # 生成 label_vis_v2 可视化图
```

## 数据依赖

运行本目录脚本需要以下外部数据（未复制到 embedding-api）：

| 数据 | 默认路径 | 环境变量覆盖 |
|------|----------|--------------|
| 原始 Embedding NPY | `/workspace/raw/xuannv_modelscope_upload/embeddings/v5_mixed_scale/monthly_embeddings_2025` | `RAW_EMBEDDINGS_DIR` |
| S2 原始影像 | `/workspace/raw/harbin_scenes/s2` | `S2_DIR` |
| Shapefile | `/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件` | `SHP_DIR` |
| Excel 清单 | `/workspace/哈尔滨松北新区变化检测汇总文件/变化检测清单` | `EXCEL_DIR` |

## 复现流程

### 1. 生成 Embedding 可视化 PNG

```bash
cd /workspace/embedding-api
python pipelines/harbin/generate_embedding_tiles.py \
  --embeddings-dir /path/to/embeddings \
  --output-dir data/harbin/embeddings \
  --max-patches 500
```

### 2. 训练任务头（可选，已有模型在 `models/harbin/`）

```bash
# v1
python pipelines/harbin/train_task_head.py --task construction --device cuda

# v2
python pipelines/harbin/train_task_head_v2.py --task construction --device cuda
```

### 3. 任务头推理 → Predictions

```bash
# v1
python pipelines/harbin/inference_task_head.py --task construction --month 2025-10 --device cuda

# v2
python pipelines/harbin/inference_task_head_v2.py --task construction --device cuda
```

输出到 `data/harbin/tasks/{task}/{version}/predictions/`。

### 4. 生成 Result Tiles

```bash
# v1
python pipelines/harbin/generate_prediction_tiles.py --task construction --month 2025-10

# v2
python pipelines/harbin/generate_prediction_tiles_v2.py --task construction
```

### 5. 生成 Labels（从 Shapefile）

```bash
# v1
python pipelines/harbin/shp_to_patch_masks.py

# v2
python pipelines/harbin/shp_to_patch_masks_v2.py
```

### 6. 生成 Label 可视化

```bash
python pipelines/harbin/visualize_labels_v2.py
```

输出到 `data/harbin/tasks/{task}/v2/label_vis/{period}/`。

## 路径说明

- 所有**数据输出路径**已指向 `embedding-api/data/harbin/` 下的统一结构
- 所有**模型路径**已指向 `embedding-api/models/harbin/v{version}/`，其中 `v1` 对应单期 embedding 任务头（V4 架构），`v2` 对应两期 embedding 差分任务头（V5 架构）
- 外部大文件（原始 embedding、S2 影像）通过环境变量配置，保持在外部存储
