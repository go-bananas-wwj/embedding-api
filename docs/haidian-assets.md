# 海淀区模型与数据资产

海淀区 `v1` 使用最新 ModelScope 数据集：

```text
https://modelscope.cn/datasets/WeijieWu/xuannv_haidian_embdding
```

当前来源前缀：

```text
artifacts/haidian-embedding-v1
```

旧的 `haidian/v1/api_ready` 不再作为当前 API 的数据来源。

## P10C 来源校验

当前部署的是 P10C epoch 800，不是 P10A/P10B 或旧版 P2A 静态资产：

| 校验项 | 当前值 |
|------|------|
| 来源实验 | `v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704` |
| 月度 embedding 目录 | `haidian_202512_202605_p10c_epoch800` |
| Checkpoint | `haidian_embedding_v1_p10c_epoch800.pt` |
| Checkpoint SHA-256 | `69dfd81c898544413a747f5c7304cc9210ad1cf420ce724864b8bd7deb6ed790` |

API 使用的 `{patch_id}.png` PCA 预览不是直接复制 ModelScope 中的整幅展示图，
而是从同批 P10C `{patch_id}.npy` embedding 重新生成。所有 patch 共用同一套
PCA 基和全局 2%–98% 色阶，并仅对共享边界最外侧像素做轻量展示混合；原始
NPY 和模型推理输入不受影响。

## 下载并安装

```bash
export MODELSCOPE_TOKEN="..."  # 私有数据集需要，不要提交到代码仓库

python pipelines/haidian/download_modelscope_assets.py \
  --repo WeijieWu/xuannv_haidian_embdding \
  --prefix artifacts/haidian-embedding-v1 \
  --target .
```

## 安装后的目录

| 资产 | 路径 |
|------|------|
| Embedding | `data/haidian/embeddings/v1/{YYYYMM}/{patch_id}.npy|png|json` |
| Embedding checkpoint | `models/haidian/v1/embedding/haidian_embedding_v1_p10c_epoch800.pt` |
| 建筑物任务头 | `models/haidian/v1/task_heads/building_mlp_fold0_best.pt` |
| 道路任务头 | `models/haidian/v1/task_heads/road_mlp_fold0_best.pt` |
| 水体任务头 | `models/haidian/v1/task_heads/water_mlp_fold0_best.pt` |

当前可用月份：

```text
202512, 202601, 202602, 202603, 202604, 202605
```

## 推荐联调参数

```text
region_id = haidian
patch_id  = patch_000000
version   = v1
month     = 202512
```

## 冒烟测试

```bash
export BASE="http://localhost:9061"

curl -s "$BASE/regions/haidian/patches/patch_000000/embedding?format=json&version=v1&month=202512"

curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_building.png

curl -s "$BASE/regions/haidian/patches/patch_000000/tasks/road_extraction/result?format=png&version=v1&month=202512" \
  -o /tmp/haidian_road.png
```

## 注意事项

- 道路提取应使用最新海淀模型任务头，不使用 GT override。
- 接口返回效果可用 `scripts/` 下的审计和可视化脚本重新生成。
- 替换资产版本时，需要同步检查 `config.yaml` 中的路径。
