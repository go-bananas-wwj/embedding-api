# 自定义模型训练工作流（前端接入指南）

> 本文档面向前端开发人员，说明如何调用 `embedding-api` 的自定义训练相关接口，完成从用户标注到模型训练、推理的完整流程。
>
> 生产环境 Base URL：`http://60.31.21.42:22065`

---

## 目录

1. [整体流程概览](#1-整体流程概览)
2. [前置准备](#2-前置准备)
3. [Step 1：前端本地管理分类与标注](#step-1前端本地管理分类与标注)
4. [Step 2：创建模型并启动训练](#step-2创建模型并启动训练)
5. [Step 3：轮询训练进度](#step-3轮询训练进度)
6. [Step 4：单张推理](#step-4单张推理)
7. [Step 5：批量推理](#step-5批量推理)
8. [Step 6：展示结果图](#step-6展示结果图)
9. [状态流转说明](#状态流转说明)
10. [错误处理](#错误处理)
11. [完整前端代码示例](#完整前端代码示例)

---

## 1. 整体流程概览

```text
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ 1. 前端本地管理分类  │     │ 2. 前端本地管理标注  │     │ 3. 前端提交标注包    │
│    与标注 (localStorage) │    (GeoJSON)         │     │    POST /models      │
└─────────────────────┘     └─────────────────────┘     └──────────┬──────────┘
                                                                   │
                                                                   ▼
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ 6. 展示结果图        │ ◀── │ 4/5. 单张/批量推理   │ ◀── │ 后端解析标注包、     │
│                     │     │    POST /models/{id}/infer  │    训练下游任务头    │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

### 关键设计

- **前端自治**：分类和标注完全由前端在浏览器 `localStorage` / `IndexedDB` 中管理，后端不再提供 `/annotations` 接口。
- **训练包**：前端在调用 `POST /models` 时，把完整的标注包（GeoJSON FeatureCollection + classes 数组）一次性传给后端。
- **后端训练**：后端解析 GeoJSON，把 WGS84 多边形栅格化为 128×128 mask 并提取 embedding。有效 Polygon 少于 10 个时自动使用 `PU + Query` 向量检索；达到 10 个时使用 `binary_conv3x3` few-shot 二分类下游头。
- **模型名称用户定义**：`name` 字段由用户输入，后端只负责生成 `model_id`。
- **无演示兜底**：`POST /models` 必须提交完整 `annotations` 和
  `classes`。空请求、缺字段或格式错误会返回 `422`，不会自动创建 demo
  模型。

---

## 2. 前置准备

### 2.1 认证

接口需要携带 API Key：

```bash
curl -H "X-API-Key: your_api_key" http://60.31.21.42:22065/health
```

或 Bearer Token：

```bash
curl -H "Authorization: Bearer your_api_key" http://60.31.21.42:22065/health
```

如果 `config.yaml` 中未配置 `auth`，系统会回退到 `default` 用户。

### 2.2 常用概念

| 字段 | 含义 | 示例 |
|------|------|------|
| `region_id` | 区域 ID | `harbin`、`haidian` |
| `patch_id` | 图块 ID | `patch_000000` |
| `month` | 影像月份 | `2025-04` |
| `before_month` / `after_month` | 变化检测两期 | `2025-04` / `2025-06` |
| `task_type` | 任务类型 | `change_detection`、`building_extraction`、`land_use_classification` |
| `model_type` | 模型类型 | `single_time_detection`、`change_detection`；旧值 `classification` 仍兼容 |

---

## Step 1：前端本地管理分类与标注

### 1.1 分类数组（classes）

前端在本地维护分类列表：

```json
[
  { "id": "cls_001", "name": "建筑用地", "color": "#FF0000" },
  { "id": "cls_002", "name": "水体", "color": "#0000FF" }
]
```

### 1.2 标注 GeoJSON（annotations）

前端把用户在地图上绘制的标注保存为标准 GeoJSON FeatureCollection：

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "patch_id": "patch_000000",
        "region_id": "harbin",
        "class_id": "cls_001",
        "class_name": "建筑用地",
        "color": "#FF0000",
        "task_type": "building_extraction",
        "month": "2025-04"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [126.51631, 45.743707],
            [126.532242, 45.743707],
            [126.532242, 45.755574],
            [126.51631, 45.755574],
            [126.51631, 45.743707]
          ]
        ]
      }
    }
  ]
}
```

### 坐标系说明

- 所有 `geometry` 坐标必须是 **WGS84 (EPSG:4326)**，顺序为 `[经度, 纬度]`。
- 支持的 `geometry.type`：**`Polygon`**、**`MultiPolygon`**。
- 不支持 `Point`、`LineString`（如需点/线，请前端转成很小的 Polygon）。

### GeoJSON Feature 的 properties

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `patch_id` | string | 是 | 标注所在 Patch |
| `region_id` | string | 是 | 区域 ID |
| `class_id` | string | 是 | 分类 ID |
| `class_name` | string | 否 | 分类名称（展示用） |
| `color` | string | 否 | 分类颜色 |
| `task_type` | string | 是 | 任务类型 |
| `month` | string | 条件 | 单期任务必填 |
| `before_month` | string | 条件 | 变化检测必填 |
| `after_month` | string | 条件 | 变化检测必填 |

### 多分类与多标注支持

- 一个 `FeatureCollection` 可以包含**多个 Feature**。每个 Polygon 都单独计入有效样本数；MultiPolygon 的每个独立 Polygon 也分别计数。使用卷积头时，同一 `class_id` 在同一 Patch/时间下的多个 Polygon 会合并成训练 mask。
- 一个训练请求当前只能选择**一个目标类别**参与训练；输出语义是“目标 / 非目标”。`classes` 可以包含完整类别列表，但 `class_ids` 只能传 1 个目标类别。
- few-shot 训练不会把 Polygon 外部全部当成负样本。Polygon 内部是目标正样本，Polygon 外部默认是“未标注/忽略”。如果请求中没有显式负样本，后端只从与正样本 embedding 相似度较低的区域抽取少量弱负样本，避免模型把整张图外部都学成背景。
- `class_ids` 可选。传入时表示“本次训练的目标类别”，当前只能传 1 个；完整标注包里可以包含未选中的类别，后端会忽略它们。`class_ids` 中的 ID 必须在 `classes` 中定义。

---

## Step 2：创建模型并启动训练

### 请求

```http
POST /models
Content-Type: application/json
```

### 请求体（分类任务示例）

```json
{
  "name": "哈尔滨建筑提取模型",
  "model_type": "single_time_detection",
  "region_id": "harbin",
  "embedding_version": "v2",
  "epochs": 20,
  "annotations": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "properties": {
          "patch_id": "patch_000000",
          "region_id": "harbin",
          "class_id": "cls_001",
          "class_name": "建筑用地",
          "color": "#FF0000",
          "task_type": "building_extraction",
          "month": "2025-04"
        },
        "geometry": {
          "type": "Polygon",
          "coordinates": [
            [
              [126.51631, 45.743707],
              [126.532242, 45.743707],
              [126.532242, 45.755574],
              [126.51631, 45.755574],
              [126.51631, 45.743707]
            ]
          ]
        }
      }
    ]
  },
  "classes": [
    { "id": "cls_001", "name": "建筑用地", "color": "#FF0000" }
  ]
}
```

### 请求体（变化检测任务示例）

```json
{
  "name": "哈尔滨变化检测模型",
  "model_type": "change_detection",
  "region_id": "harbin",
  "embedding_version": "v2",
  "epochs": 20,
  "annotations": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "properties": {
          "patch_id": "patch_000000",
          "region_id": "harbin",
          "class_id": "cls_001",
          "class_name": "变化区域",
          "color": "#FF0000",
          "task_type": "change_detection",
          "before_month": "2025-04",
          "after_month": "2025-06"
        },
        "geometry": {
          "type": "Polygon",
          "coordinates": [
            [
              [126.51631, 45.743707],
              [126.532242, 45.743707],
              [126.532242, 45.755574],
              [126.51631, 45.755574],
              [126.51631, 45.743707]
            ]
          ]
        }
      }
    ]
  },
  "classes": [
    { "id": "cls_001", "name": "变化区域", "color": "#FF0000" }
  ]
}
```

### 响应

```json
{
  "id": "model_xyz789",
  "name": "哈尔滨建筑提取模型",
  "type": "single_time_detection",
  "task_type": "building_extraction",
  "status": "training",
  "created_at": "2025-06-26T10:00:00",
  "completed_at": null,
  "classes": [
    { "id": "cls_001", "name": "建筑用地", "color": "#FF0000" }
  ],
  "accuracy": null,
  "n_samples": null,
  "model_path": "users/default/models/model_xyz789.pkl",
  "description": "哈尔滨建筑提取模型",
  "message": null,
  "job_id": "job_def456"
}
```

### 后端处理流程

1. 校验 `annotations` 和 `classes` 格式。
2. 按 `patch_id` 分组 GeoJSON features。
3. 读取每个 Patch 的 WGS84 bbox。
4. 使用 `shapely` + `rasterio` 把 Polygon 栅格化为 128×128 mask。
5. 推理结果图统一输出为 128×128 PNG。
6. 加载对应月份的 embedding；变化检测会加载前后两期并计算 embedding 差分。
7. 将 Polygon 内部作为目标正样本；Polygon 外部保持“未标注”语义，不整体视为负样本。
8. 有效 Polygon 少于 10 个时，从远离标注且前景相似度最低的 30% 未标注像素中估计可靠背景，训练 PU 前景/背景原型并用 F0.5 自动选择阈值；推理时执行一次受限 Query 自适应。
9. 有效 Polygon 大于等于 10 个时，训练 `binary_conv3x3` few-shot 二分类下游头并自动选择推理阈值。
10. 保存带格式版本的模型 checkpoint，更新模型状态。

### 校验与限制

- `model_type` 与 `task_type` 必须匹配：
  - `single_time_detection` 仅支持 `building_extraction`、`road_extraction`、`construction`、`land_use_classification`、`land_cover_classification`、`water_extraction`。
  - `change_detection` 必须搭配 `task_type: "change_detection"`。
- 所有 Feature 的 `region_id` 必须与顶层 `region_id` 一致。
- 所有 Feature 的 `class_id` 必须在 `classes` 中定义；如果传入 `class_ids`，只能传 1 个目标类别，未选中的类别会在训练时被忽略。
- 标注包限制：最多 `10000` 个 Feature，总顶点数不超过 `100000`。
- `epochs` 默认 `100`，请求范围 `1~1000`；服务端最多执行 `100` 轮以控制训练耗时。

---

## Step 3：轮询训练进度

### 请求

```http
GET /models/jobs/{job_id}
```

### 响应（训练中）

```json
{
  "job_id": "job_def456",
  "status": "running",
  "model_id": "model_xyz789",
  "accuracy": null,
  "n_samples": null,
  "model_path": null,
  "message": null
}
```

### 响应（完成）

```json
{
  "job_id": "job_def456",
  "status": "completed",
  "model_id": "model_xyz789",
  "accuracy": 0.95,
  "n_samples": 1200,
  "model_path": "users/default/models/model_xyz789.pkl",
  "message": null
}
```

### 响应（失败）

```json
{
  "job_id": "job_def456",
  "status": "failed",
  "model_id": "model_xyz789",
  "accuracy": null,
  "n_samples": null,
  "model_path": null,
  "message": "No valid training samples after filtering"
}
```

### 前端提示

- 建议每隔 2-3 秒轮询一次。
- 状态为 `completed` 后，可以调用推理接口。
- 状态为 `failed` 时，展示 `message` 给用户。

---

## Step 4：单张推理

### 请求

```http
POST /models/{model_id}/infer
Content-Type: application/json
```

分类模型传 `month`；变化检测模型传 `before_month` 和 `after_month`。

```json
{
  "region_id": "harbin",
  "patch_id": "patch_000001",
  "month": "2025-04"
}
```

变化检测示例：

```json
{
  "region_id": "harbin",
  "patch_id": "patch_000001",
  "before_month": "2025-04",
  "after_month": "2025-06"
}
```

### 响应

```json
{
  "result_url": "/models/results/infer_model_xyz789_harbin_patch_000001_2025-04.png"
}
```

---

## Step 5：批量推理

### 请求

```http
POST /models/{model_id}/infer_batch
Content-Type: application/json
```

```json
{
  "region_id": "harbin",
  "patch_ids": [
    "patch_000000",
    "patch_000001",
    "patch_000002"
  ],
  "month": "2025-04"
}
```

变化检测批量推理示例：

```json
{
  "region_id": "harbin",
  "patch_ids": [
    "patch_000000",
    "patch_000001",
    "patch_000002"
  ],
  "before_month": "2025-04",
  "after_month": "2025-06"
}
```

### 响应

```json
{
  "total": 2,
  "success_count": 2,
  "error_count": 0,
  "results": [
    {
      "patch_id": "patch_000000",
      "status": "success",
      "result_url": "/models/results/infer_model_xyz789_harbin_patch_000000_2025-04.png",
      "error": null
    },
    {
      "patch_id": "patch_000001",
      "status": "success",
      "result_url": "/models/results/infer_model_xyz789_harbin_patch_000001_2025-04.png",
      "error": null
    }
  ]
}
```

---

## Step 6：展示结果图

### 请求

```http
GET /models/results/infer_model_xyz789_harbin_patch_000001_2025-04.png
```

### 响应

直接返回 128×128 PNG 图片。

### 前端展示方式

```html
<img src="http://60.31.21.42:22065/models/results/infer_model_xyz789_harbin_patch_000001_2025-04.png" alt="推理结果" />
```

或在地图上叠加为图层：

```javascript
L.imageOverlay(
  'http://60.31.21.42:22065/models/results/infer_model_xyz789_harbin_patch_000001_2025-04.png',
  [[45.743707, 126.51631], [45.755574, 126.532242]]
).addTo(map);
```

---

## 状态流转说明

### 模型状态

```text
training ──▶ completed
    │
    ▼
 failed
```

| 状态 | 含义 | 可执行操作 |
|------|------|------------|
| `training` | 训练中 | 查看进度 |
| `completed` | 训练完成 | 推理、删除 |
| `failed` | 训练失败 | 查看错误、删除 |

---

## 错误处理

### 常见错误码

| HTTP 状态码 | 场景 | 前端处理 |
|-------------|------|----------|
| 400 | 请求参数错误 | 展示具体错误信息 |
| 401 | API Key 无效或缺失 | 提示用户登录或检查 API Key |
| 404 | 模型/任务不存在 | 检查 ID 是否正确 |
| 422 | Pydantic 校验失败 | 根据返回 detail 修正字段 |
| 500 | 服务器内部错误 | 联系后端排查 |

### 典型错误示例

**GeoJSON geometry 类型不支持**

```json
{
  "detail": "Unsupported geometry type: Point. Only Polygon and MultiPolygon are allowed."
}
```

**分类任务缺少 month**

```json
{
  "detail": "classification model requires 'month' for patch patch_000000"
}
```

**没有有效训练样本**

```json
{
  "detail": "No valid training samples after filtering"
}
```

---

## 完整前端代码示例

```javascript
const BASE_URL = 'http://60.31.21.42:22065';
const API_KEY = 'your_api_key';

async function request(path, method = 'GET', body = null) {
  const options = {
    method,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
    },
  };
  if (body) options.body = JSON.stringify(body);
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// 1. 前端本地存储分类和标注（示例）
const classes = [
  { id: 'cls_001', name: '建筑用地', color: '#FF0000' }
];

const annotations = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      properties: {
        patch_id: 'patch_000000',
        region_id: 'harbin',
        class_id: 'cls_001',
        task_type: 'building_extraction',
        month: '2025-04'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [126.51631, 45.743707],
          [126.532242, 45.743707],
          [126.532242, 45.755574],
          [126.51631, 45.755574],
          [126.51631, 45.743707]
        ]]
      }
    }
  ]
};

// 2. 创建模型并启动训练
const model = await request('/models', 'POST', {
  name: '我的建筑提取模型',  // 用户自定义名称
  model_type: 'classification',
  task_type: 'building_extraction',
  region_id: 'harbin',
  embedding_version: 'v2',
  epochs: 100,
  annotations: annotations,
  classes: classes
});

// 3. 轮询训练进度
const pollTraining = setInterval(async () => {
  const job = await request(`/models/jobs/${model.job_id}`);
  updateProgress(job.status, job.accuracy, job.n_samples);
  if (job.status === 'completed') {
    clearInterval(pollTraining);
    runInference(model.id);
  } else if (job.status === 'failed') {
    clearInterval(pollTraining);
    alert('训练失败：' + job.message);
  }
}, 2000);

// 4. 单张推理
async function runInference(modelId) {
  const result = await request(`/models/${modelId}/infer`, 'POST', {
    region_id: 'harbin',
    patch_id: 'patch_000001',
    month: '2025-04'
  });
  const imgUrl = `${BASE_URL}${result.result_url}`;
  document.getElementById('result-image').src = imgUrl;
}
```

---

## 附：接口速查表

| 步骤 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 创建模型 | POST | `/models` | 提交名称 + GeoJSON 标注包 + classes |
| 查看进度 | GET | `/models/jobs/{job_id}` | 轮询训练状态 |
| 查看模型 | GET | `/models/{model_id}` | 获取模型信息 |
| 单张推理 | POST | `/models/{model_id}/infer` | 推理一个 Patch |
| 批量推理 | POST | `/models/{model_id}/infer_batch` | 批量推理（最多 100） |
| 获取结果图 | GET | `/models/results/{filename}` | 下载 PNG 结果 |

---

## 版本记录

| 版本 | 日期 | 说明 |
|------|------|------|
| v2.0 | 2025-06-26 | 改为 GeoJSON 标注包模式，删除后端 annotation 接口 |
| v1.0 | 2025-06-26 | 初稿，基于后端 AnnotationStore |
