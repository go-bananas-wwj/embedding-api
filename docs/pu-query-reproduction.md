# PU + Query 原理与跨机器复现指南

本文说明少样本自定义模型中的 `PU + Query` 是如何实现的，以及如何在另一台
机器上复现生产 API 和历史可视化实验。

对应实现：

- 核心算法：`app/services/pu_query.py`
- 训练接入：`app/services/training_engine.py`
- 推理接入：`app/services/inference_engine.py`
- 数值测试：`tests/test_pu_query.py`
- 历史实验：`scripts/experiment_no_positive_memory.py`

## 1. 使用条件

自定义训练会根据**有效 Polygon 数量**自动选择算法：

| 有效 Polygon 数 | 后端策略 | 是否迭代训练 |
|---:|---|---:|
| `1~9` | `PU + Query` | 否 |
| `>=10` | `Binary Conv 3x3` | 是 |

一个 `Polygon` 计一个样本；`MultiPolygon` 中每个独立 Polygon 分别计数。只能
选择一个目标 `class_id`，最终输出“目标 / 非目标”二分类结果。API 请求格式不因
策略变化而变化。

PU 是 Positive-Unlabeled：Polygon 内是确定的目标正样本，Polygon 外只是未标注
区域，不能全部当成负样本。Query 指推理时根据当前待预测 Patch 做一次保守的
查询自适应，不是前端需要填写的新参数。

## 2. 输入约定

核心函数接收：

```python
polygon_samples: list[tuple[str, np.ndarray, np.ndarray]]
```

每项分别是：

| 项 | 格式 | 说明 |
|---|---|---|
| `support_key` | `str` | 支持样本唯一键，例如 `patch_000212:202603` |
| `feature` | `float32[C,H,W]` | 玄女 embedding；海淀 P10C 当前为 64 维 |
| `polygon_mask` | `bool[H,W]` | 单个 Polygon 栅格化后的正样本 mask |

同一个 Patch 上有多个 Polygon 时，`feature` 可以重复传入，但 `support_key` 必须
相同。算法会按 `support_key` 去重 embedding，并保留每个 Polygon 的独立原型。

单时间检测直接使用当月 embedding。变化检测先计算：

```text
feature = embedding_after - embedding_before
```

再执行相同的 PU + Query 流程。训练和推理必须使用相同区域、embedding 版本和
通道维度。

## 3. 训练算法

### 3.1 标准化

使用所有支持 Patch 的全部像素估计逐通道均值和标准差：

```text
z = (x - mean) / max(std, 1e-5)
pixel = z / max(||z||2, 1e-8)
```

保存 `feature_mean` 和 `feature_std`，推理时必须复用，不能在查询 Patch 上重新
估计。

### 3.2 前景原型

1. 对每个 Polygon 内的归一化像素求均值。
2. 将每个 Polygon 均值做 L2 归一化，得到 Polygon 级原型。
3. 对所有 Polygon 原型求平均，再次 L2 归一化，得到最终前景中心 `p_fg`。

这种方式让每个 Polygon 具有近似相同的投票权，避免大 Polygon 因像素多而完全
支配原型。

### 3.3 可靠背景

算法不会把 Polygon 外全部当负样本，而是对每个支持 Patch：

1. 合并该 Patch 的所有正样本 mask。
2. 向外膨胀 `3` 个 embedding 像素，排除目标边缘和邻近区域。
3. 计算剩余未标注像素与 `p_fg` 的余弦相似度。
4. 只保留相似度最低的 `30%`，作为可靠背景候选。
5. 每个支持 Patch 最多均匀抽取 `2048` 个背景像素。
6. 对全部候选求均值并 L2 归一化，得到背景中心 `p_bg`。

### 3.4 分数与阈值

每个像素的对比分数为：

```text
score(x) = cos(x, p_fg) - 0.65 * cos(x, p_bg)
```

正样本来自 Polygon 内部，弱负样本来自可靠背景。算法在分数最小值到最大值间
扫描 `180` 个阈值，以 `F0.5` 最大为目标选择阈值。`F0.5` 更重视 Precision，
用于减少少样本检索的大面积误检。

模型不保存整套正样本记忆库，只保存两个中心、标准化统计量和阈值。

## 4. Query 自适应

基础分数先经过 `sigma=0.55` 的轻量高斯平滑。随后在当前查询 Patch 上：

1. 计算基础分数的 `99.7%` 分位数。
2. 置信门槛取 `max(99.7% 分位数, threshold + 0.05)`。
3. 只有置信像素数处于 `4~128` 时才生成查询原型 `p_query`。
4. 用置信像素均值生成 `p_query`，重新计算对比得分。
5. 最终分数为 `88%` 基础分数加 `12%` 查询得分。

为防止错误自适应扩散，如果新预测面积超过下式，就放弃 Query 更新并返回基础
分数：

```text
candidate_area > max(64, base_area * 1.35)
```

因此 Query 不是每次都会启用。置信像素过少、过多或面积增长失控时，
`query_adapted=False` 是正常的安全回退。

## 5. 模型文件格式

生产模型使用 `torch.save()`，文件扩展名由模型注册表决定，内容格式标识为：

```text
pu_query_retrieval_v1
```

主要字段：

| 字段 | 含义 |
|---|---|
| `__format__` | 固定为 `pu_query_retrieval_v1` |
| `head_type` | 固定为 `pu_query_retrieval` |
| `feature_mean/std` | 支持 Patch 的通道统计量 |
| `foreground_center` | 前景原型 |
| `background_center` | 可靠背景原型 |
| `threshold` | F0.5 自动选择的阈值 |
| `training_f05` | 正样本与弱负样本上的训练参考值 |
| `region_id` | 区域 |
| `embedding_version` | embedding 版本 |
| `feature_type` | `embedding` 或 `diff` |
| `polygon_count` | 实际有效 Polygon 数 |

`training_f05` 不是独立验证集精度，不能当作上线效果指标。

## 6. 从零复现生产 API

### 6.1 获取代码和资产

```bash
git clone https://github.com/go-bananas-wwj/embedding-api.git
cd embedding-api

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export MODELSCOPE_TOKEN="你的新 Token"
# 复现海淀时执行：
python pipelines/haidian/download_modelscope_assets.py --target .

# 复现哈尔滨时执行；无需同时下载两个区域：
python pipelines/harbin/download_modelscope_assets.py \
  --target . --verify-checksums
```

海淀当前使用 `v1` P10C embedding；哈尔滨当前使用 `v2`。不要混合不同版本的
embedding、支持原型和查询数据。

### 6.2 数值自检

```bash
pytest -q tests/test_pu_query.py tests/test_training_engine.py
```

预期核心检查包括：前景区域平均分高于背景、前景中心单位化、无界面积增长会被
拒绝、少于 10 个 Polygon 的 checkpoint 格式正确。

### 6.3 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9061
```

调用 `POST /models` 提交 `1~9` 个有效 Polygon。轮询
`GET /models/jobs/{job_id}` 完成后，检查 checkpoint：

```python
import torch

model = torch.load("users/default/models/模型文件", map_location="cpu",
                   weights_only=False)
assert model["__format__"] == "pu_query_retrieval_v1"
assert model["training_strategy"] == "pu_query_retrieval"
assert model["polygon_count"] < 10
```

单张和批量推理继续使用现有 `/models/{model_id}/infer` 和批量推理接口，不需要
增加 PU 或 Query 参数。

## 7. 不启动 API 的最小复现

```python
import numpy as np
import torch

from app.services.pu_query import score_pu_query, train_pu_query

embedding = np.load("support.npy").astype(np.float32)  # [C,H,W]
mask = np.load("support_mask.npy").astype(bool)        # [H,W]

model = train_pu_query([
    ("patch_000001:202604", embedding, mask),
])
model.update({
    "__format__": "pu_query_retrieval_v1",
    "head_type": "pu_query_retrieval",
    "region_id": "haidian",
    "embedding_version": "v1",
})
torch.save(model, "pu_query_demo.pt")

query = np.load("query.npy").astype(np.float32)
score, query_adapted = score_pu_query(query, model)
prediction = score >= float(model["threshold"])
np.save("prediction.npy", prediction.astype(np.uint8))
print("threshold:", model["threshold"])
print("query adapted:", query_adapted)
```

mask 必须已经与 embedding 网格对齐。通过 API 时，这一步由 GeoJSON 栅格化与
mask resize 自动完成；直接调用核心函数时需要复现者自己保证空间对应关系。

## 8. 复现历史可视化实验

历史画廊使用：

```bash
python scripts/experiment_no_positive_memory.py \
  --output Tmp/pu_query_reproduction
```

输出包括：

- `index.html`：可点击放大的中文实验画廊。
- `results.json`：不同类别和 Polygon 数下的 Precision、Recall、F1、IoU。
- `feature_stats.npz`：实验使用的通道均值和标准差。
- 各类别独立测试 Patch 的对照图。

该脚本需要完整海淀 embedding、建筑/道路标签、土地覆盖参考标签、原始 S2 影像
和项目中文字体。完整真实标签只用于离线评估，不参与原型、背景或阈值计算。

### 历史实验与生产 API 的差异

| 项目 | 历史画廊 | 当前生产 API |
|---|---|---|
| 标准化统计 | 全海淀 `202604` embedding | 用户支持 Patch |
| 测试标签 | 用于计算离线指标 | 不需要 |
| 跨区域展示 | 从满足距离条件的候选中按标签指标精选 | 不做精选 |
| 模型产物 | 实验过程变量和 JSON | 可持久化 checkpoint |
| API 兼容 | 不涉及 | 前端请求保持不变 |

历史画廊中的“跨片区精选结果”是展示用途，不能作为无偏总体精度。上线评价应固定
独立测试集，并报告所有样本的汇总指标。

## 9. 复现验收清单

1. 支持和查询 embedding 的通道数、区域及版本一致。
2. Polygon mask 与 embedding 网格空间对齐，且至少覆盖一个像素。
3. Polygon 外部保持未标注语义，没有整体作为负样本。
4. checkpoint 的 `__format__` 为 `pu_query_retrieval_v1`。
5. 相同输入重复训练得到相同原型、阈值和预测结果。
6. `score` 全部为有限值，没有 NaN/Inf。
7. 记录 `query_adapted`，并确认拒绝自适应时仍能输出基础预测。
8. 用未参与训练和阈值选择的 Patch 单独评估 Precision、Recall、F1、IoU。

## 10. 当前固定参数

这些参数定义在 `app/services/pu_query.py`，跨机器严格复现时不要自行修改：

| 参数 | 当前值 |
|---|---:|
| 背景对比权重 | `0.65` |
| 可靠背景分位数 | `0.30` |
| 标注边缘排除距离 | `3` 像素 |
| 每支持 Patch 最大背景像素 | `2048` |
| Query 融合权重 | `0.12` |
| Query 置信分位数 | `0.997` |
| Query 最少/最多像素 | `4 / 128` |
| Query 最小阈值裕量 | `0.05` |
| 最大面积增长 | `1.35` 倍 |
| 面积保护下限 | `64` 像素 |
| 高斯平滑 sigma | `0.55` |
| 阈值候选数 | `180` |
| 阈值目标 | `F0.5` |
