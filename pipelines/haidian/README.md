# Haidian 区域复现程序

本目录包含海淀区遥感 embedding 生成程序。海淀数据使用 OLMO Earth 模型，与哈尔滨的 xuannv_show 流程不同。

## 目录结构

```
pipelines/haidian/
├── batch_produce_embeddings.py         # 批量生产 embedding
├── extract_olmoearth_embeddings.py     # 从 OlmoEarth 数据提取 embedding
└── README.md
```

## 复现流程

### 1. 提取 OlmoEarth Embedding

```bash
cd /workspace/embedding-api
python pipelines/haidian/extract_olmoearth_embeddings.py \
  --input-dir /path/to/olmoearth/data \
  --output-dir data/haidian/embeddings
```

### 2. 批量生产 Embedding

```bash
python pipelines/haidian/batch_produce_embeddings.py \
  --input-dir /path/to/raw/data \
  --output-dir data/haidian/embeddings \
  --model-checkpoint /path/to/model.pt
```

## 路径说明

- Embedding 输出到 `embedding-api/data/haidian/embeddings/`
- 每个 patch 对应一个 `{patch_id}.npy` 文件，shape 为 `(64, 128, 128)`
- 海淀区当前只有 embedding 数据，无下游任务数据
