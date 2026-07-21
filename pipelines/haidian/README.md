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
