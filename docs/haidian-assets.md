# Haidian Assets

Haidian `v1` uses the latest ModelScope dataset:

```text
https://modelscope.cn/datasets/WeijieWu/xuannv_haidian_embdding
```

Current source prefix:

```text
artifacts/haidian-embedding-v1
```

The old `haidian/v1/api_ready` package is no longer the source of truth for
the deployed API.

## Download and Install

```bash
export MODELSCOPE_TOKEN="..."  # required for private datasets; do not commit it

python pipelines/haidian/download_modelscope_assets.py \
  --repo WeijieWu/xuannv_haidian_embdding \
  --prefix artifacts/haidian-embedding-v1 \
  --target .
```

## Installed Layout

| Asset | Path |
|-------|------|
| Embeddings | `data/haidian/embeddings/v1/{YYYYMM}/{patch_id}.npy|png|json` |
| Embedding checkpoint | `models/haidian/v1/embedding/haidian_embedding_v1_p10c_epoch800.pt` |
| Building task head | `models/haidian/v1/task_heads/building_mlp_fold0_best.pt` |
| Road task head | `models/haidian/v1/task_heads/road_mlp_fold0_best.pt` |
| Water task head | `models/haidian/v1/task_heads/water_mlp_fold0_best.pt` |

Current available months:

```text
202512, 202601, 202602, 202603, 202604, 202605
```

## Recommended Integration Values

```text
region_id = haidian
patch_id  = patch_000000
version   = v1
month     = 202512
```

## Smoke Tests

```bash
export BASE="http://localhost:9061"

curl -s "$BASE/regions/haidian/patches/patch_000000/embedding?format=json&version=v1&month=202512"

curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_building.png

curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/road_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_road.png
```

## Notes

- Road extraction should use the latest Haidian model head, not a GT override.
- Result visualizations can be regenerated with the audit/visualization scripts
  under `scripts/`.
- Keep `config.yaml` aligned with the paths above when changing asset versions.
