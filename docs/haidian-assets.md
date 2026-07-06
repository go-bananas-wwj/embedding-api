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
