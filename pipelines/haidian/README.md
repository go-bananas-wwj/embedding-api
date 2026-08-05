# Haidian V1 P2A Deployment Pipeline

This directory contains Haidian deployment utilities.  API version `v1` for
Haidian is backed by the xuannv P2A embedding model trained on December 2025
through May 2026 imagery.

## Available API Tasks

The Haidian V1 package follows the same `data/<region>` and `models/<region>`
layout used by Harbin.

| API task | Source task | Default head | Status |
|---|---|---|---|
| `building_extraction` | latest Haidian embedding | MLP | ready |
| `road_extraction` | latest Haidian embedding | MLP | ready |
| `water_extraction` | latest Haidian embedding | MLP | ready |

The API version name is intentionally `v1` even though the underlying research
experiment is called P2A.

## Directory Layout

```text
data/haidian/
  patches_meta_v1.json
  embeddings/v1/{month}/{patch_id}.npy
  embeddings/v1/{month}/{patch_id}.png
  tasks/{task}/v1/predictions/{patch_id}.npy
  tasks/{task}/v1/results/tiles/{patch_id}.png
  tasks/{task}/v1/labels/...

models/haidian/v1/
  embedding/haidian_embedding_v1_p10c_epoch800.pt
  task_heads/building_mlp_fold0_best.pt
  task_heads/road_mlp_fold0_best.pt
  task_heads/water_mlp_fold0_best.pt
```

Months are `202512`, `202601`, `202602`, `202603`, `202604`, and `202605`.

## Install Assets From ModelScope

The latest assets are stored in the ModelScope dataset
`WeijieWu/xuannv_haidian_embdding` under
`artifacts/haidian-embedding-v1`.

```bash
cd /workspace/projects/embedding-api
export MODELSCOPE_TOKEN="..."  # only needed for private datasets
python pipelines/haidian/download_modelscope_assets.py \
  --repo WeijieWu/xuannv_haidian_embdding \
  --prefix artifacts/haidian-embedding-v1 \
  --target .
```

下载器也会安装 `artifacts/haidian-embedding-v1/deployment` 下的可选部署归档，
包括三个 Conv3×3 系统任务头、已经生成的按月任务结果，以及 S1、S2、Landsat、
高分光学和高分 SAR GeoTIFF。补充包包含哈尔滨道路结果和可选 AEF 2025 外部
embedding。归档在解压前会进行 SHA256 校验。

海淀施工地六个月目录当前复用同一套静态结果，不代表六个月分别完成了模型
推理，也不应用于施工地月际变化分析。

目标机器已经具备 embedding 时，增加 `--deployment-only`，避免重复下载和
转换 embedding。

道路结果可从本地 embedding 和系统任务头重新生成：

```bash
python pipelines/haidian/generate_system_task_results.py \
  --task road_extraction \
  --months 202512 202601 202602 202603 202604 202605
```

只把本机已经存在的 API 资产整理成部署归档，不会重新训练或推理：

```bash
python pipelines/haidian/prepare_deployment_assets.py \
  --output-root /workspace/modelscope_upload/haidian/deployment
```

After download, start the API normally:

```bash
DOCS_URL=/docs uvicorn app.main:app --host 0.0.0.0 --port 9061
```

## Prepare Upload Package

On the xuannv training machine, create the ModelScope upload package:

```bash
cd /workspace/projects/embedding-api
python pipelines/haidian/prepare_v1_api_assets.py \
  --output-root /data/xuannv_embedding/modelscope_upload/haidian/v1
```

For a fast dry run:

```bash
python pipelines/haidian/prepare_v1_api_assets.py \
  --output-root /tmp/haidian_v1_dryrun \
  --max-patches 2 \
  --copy-mode symlink \
  --skip-raw-training-data
```

## Regenerate Task Predictions

If task results need to be regenerated from downloaded embeddings and task-head
weights:

```bash
python pipelines/haidian/inference_task_head.py \
  --task building_extraction \
  --before-month 202512 \
  --after-month 202605 \
  --device cuda
```

Supported downloaded heads are `building_extraction`, `road_extraction`, and
`water_extraction`.

## Example API Calls

```bash
curl -s "http://localhost:9061/regions/haidian/patches/patch_000000/embedding?format=json&version=v1&month=202512"
curl -s "http://localhost:9061/regions/haidian/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1" -o /tmp/haidian_building.png
curl -s "http://localhost:9061/regions/haidian/patches/patch_000000/tasks/road_extraction/prediction?version=v1" -o /tmp/haidian_road.npy
```
