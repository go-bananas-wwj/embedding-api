# 模型与训练方式前端对接

本文是变化检测、土地分类和自定义训练方式的前端契约。字段定义以
`/openapi.json` 为准，可用能力以 `GET /models/capabilities` 为准。

## 1. 前端调用顺序

1. 调用 `GET /models/capabilities?region_id=haidian` 获取真实可用方式。
2. 只展示 `available=true` 的训练方式；不可用原因可以作为禁用提示。
3. 使用现有 `POST /models` 创建模型，轮询 `/models/jobs/{job_id}`。
4. 训练完成后调用 `/models/{model_id}/infer` 或 `/infer_batch`。

旧前端不传 `training_method` 时行为不变，默认等价于：

```json
{"training_method": "xuannv_earth"}
```

## 2. 训练方式

| training_method | 中文名称 | 输入 | 下游算法 | 当前状态 |
|---|---|---|---|---|
| `xuannv_earth` | 玄女地球训练（默认） | 玄女 embedding | `<10` 个有效 Polygon 使用 PU + Query，`>=10` 使用 Binary Conv 3x3 | 可用 |
| `traditional_ml` | 传统方法训练 | 仅 Sentinel-2 L2A 光学影像 | Random Forest 像素二分类 | 可用，限单时相 |
| `aef` | AEF 下游训练 | 冻结 AEF embedding | 两层像素 MLP | 安装真实 AEF embedding 后可用 |
| `dinov3_sat493m` | DINOv3-SAT493M 下游训练 | 冻结 ViT-L/16 SAT-493M dense tokens | 两层像素 MLP | 安装权重后可用 |

### 传统方法的准确含义

传统方法不读取玄女 embedding、S1、Landsat 或高分影像。服务只读取与标注
`patch_id + month` 对应的 Sentinel-2 光学 GeoTIFF；同月存在多景时按日期倒序
选择最新一景。

统一使用 `B02/B03/B04/B08/B11/B12`，并计算 `NDVI/NDWI/MNDWI/NDBI`。
海淀 0~10000 反射率会自动缩放，哈尔滨已经归一化的反射率保持原值。
无数据像素不参与训练或预测。

Polygon 内部是正样本，外部是未标注区域，不会整体当作负样本。服务只从
光谱上明显远离正样本的未标注像素中抽取保守弱负样本。分类器固定为
`RandomForestClassifier`，避免前端再传一组难以稳定复现的算法参数。

## 3. 创建传统模型

请求体与原接口完全相同，只增加一个字段：

```json
{
  "name": "S2 水体随机森林",
  "model_type": "single_time_detection",
  "training_method": "traditional_ml",
  "region_id": "haidian",
  "class_ids": ["cls_water"],
  "annotations": {
    "type": "FeatureCollection",
    "features": []
  },
  "classes": [
    {"id": "cls_water", "name": "水体", "color": "#1D4ED8"}
  ]
}
```

`features` 仍使用现有 GeoJSON Polygon/MultiPolygon 格式。每个 Feature 必须提供
`patch_id`、`region_id`、`class_id`、`task_type` 和 `month`。

变化检测不能选择 `traditional_ml`。提交不支持的组合会在创建任务前返回 422；
已选择的训练方式缺少模型权重、S2 影像或 AEF embedding 时返回 409，
不会静默改用玄女模型。

## 4. 训练响应追溯字段

模型详情和任务状态增加以下可选字段，因此旧模型仍可正常读取：

| 字段 | 含义 |
|---|---|
| `requested_training_method` | 前端请求的方案 |
| `resolved_training_method` | 实际执行算法，例如 `random_forest`、`pu_query_retrieval` |
| `feature_source` | `sentinel2_l2a` 或 `xuannv_embedding` |

传统模型 checkpoint 同时保存波段顺序、光谱指数、阈值、Random Forest 参数、
训练区域和输入影像数量。推理时禁止跨区域使用。

## 5. 变化检测接口

变化检测是双时相二分类任务：

```json
{
  "model_type": "change_detection",
  "training_method": "xuannv_earth",
  "region_id": "harbin"
}
```

标注 Feature 和推理请求都必须提供：

```json
{"before_month": "2025-04", "after_month": "2025-06"}
```

不得同时传 `month`。系统预生成结果使用：

```text
GET /regions/{region_id}/patches/{patch_id}/tasks/change_detection/result
```

## 6. 土地利用分类接口

`land_use_classification` 表示单时相类别图，只传 `month`：

```text
GET /regions/{region_id}/patches/{patch_id}/tasks/land_use_classification/result?version=v1&month=2026-05
```

历史哈尔滨 `land_use_classification v2` 实际承载 `land_conversion` 双时相结果。
为兼容现有前端暂不删除，但新页面不得把它当成单月土地利用分类。后续规范任务为：

- `land_use_classification`：单时相类别图；
- `change_detection`：是否变化的双时相二分类；
- `land_use_transition`：从一种土地类别转移到另一种类别，目前未开放。

Dynamic World 和 ESA WorldCover 是不同标签来源。前端必须使用对应 classes 接口
返回的类别 ID、名称和颜色，不能在页面中硬编码一套颜色表。

## 7. AEF 与 DINOv3

两种方式都冻结底座特征，不微调 foundation model。下游头统一为普通两层 MLP：
`Linear(C,128) -> ReLU -> Dropout -> Linear(128,1)`。单时相直接输入 embedding；
变化检测拼接 `before`、`after`、绝对差值和逐元素乘积。

## 模型 ID 与基座模型绑定

每个自定义模型 ID 表示一条完整、不可拆分的推理管线，而不只是下游头权重。
训练完成后，模型注册表和 checkpoint 会同时保存：

- `foundation_model_id`、`foundation_model_version`
- `feature_source`、`feature_dimension`
- `preprocessing_version`
- `head_type`、`checkpoint_format`
- `compatible_regions`

单张和批量推理会根据模型 ID 自动加载训练时绑定的玄女、AEF、DINOv3 或
Sentinel-2 特征流程，并在读取特征前校验区域、基座版本和输入维度。前端不能
在推理请求中覆盖这些字段，也不需要按训练方式维护多套推理参数；单时间模型
统一传 `month`，双时相模型统一传 `before_month` 和 `after_month`。

AEF 从 `AEF_EMBEDDING_DIR` 读取真实 `[C,H,W]` NPY。当前部署使用
[Source Cooperative AlphaEarth Foundations](https://source.coop/tge-labs/aef) `v1/annual`
2025 年 64 维年度 embedding：哈尔滨 424 个 Patch、海淀 320 个 Patch，均按 API
Patch 边界裁成 `64x128x128`，并使用官方公式反量化。AEF 是年度产品，因此
前端无论提交哪个月份，当前部署都读取同一份 2025 年特征，不应解释为月度
embedding。训练完成后模型绑定中的 `foundation_model_version` 和
`preprocessing_version` 均为 `aef_annual_2025`。只有区域内完全没有 AEF 年度
资产时，能力接口才返回 `available=false`，创建模型返回 409。

可用以下命令从公开 COG 重新安装，脚本会按配置中的 Patch 网格裁切并校验 nodata：

```bash
python scripts/install_sourcecoop_aef.py --region harbin --year 2025
python scripts/install_sourcecoop_aef.py --region haidian --year 2025
```

DINOv3 使用 Meta ViT-L/16 SAT-493M（300M 参数、1024 维 dense token），从
Sentinel-2 RGB 生成 dense feature 并缓存。权重目录由
`DINOV3_SAT493M_MODEL_DIR` 配置，默认 `models/dinov3_sat493m`。服务会严格校验
checkpoint，出现 missing/unexpected keys 时拒绝启动特征提取，不能使用随机权重。

## 8. 错误处理

| 状态码 | 场景 |
|---|---|
| 404 | 区域、S2 影像、模型或任务不存在 |
| 409 | 训练方式所需的模型权重、S2 影像或 AEF embedding 未安装 |
| 422 | 参数不合法，或训练方式与单/双时相不兼容 |
| 400 | 已创建模型但训练数据不能形成有效正样本和弱负样本 |

前端应显示后端 `detail`，但不要自动换训练方式后重试。
