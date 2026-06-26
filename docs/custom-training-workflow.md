# 自定义模型训练工作流（前端接入指南）

> 本文档面向前端开发人员，说明如何调用 `embedding-api` 的自定义训练相关接口，完成从标注 → 训练 → 推理的完整流程。
>
> 生产环境 Base URL：`http://60.31.21.42:22065`

---

## 目录

1. [整体流程概览](#1-整体流程概览)
2. [前置准备](#2-前置准备)
3. [Step 1：创建分类（Class）](#step-1创建分类class)
4. [Step 2：创建标注（Annotation）](#step-2创建标注annotation)
5. [Step 3：创建模型并启动训练](#step-3创建模型并启动训练)
6. [Step 4：轮询训练进度](#step-4轮询训练进度)
7. [Step 5：单张推理](#step-5单张推理)
8. [Step 6：批量推理](#step-6批量推理)
9. [Step 7：展示结果图](#step-7展示结果图)
10. [状态流转说明](#状态流转说明)
11. [错误处理](#错误处理)
12. [完整前端代码示例](#完整前端代码示例)

---

## 1. 整体流程概览

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  1.创建分类  │ ──▶ │  2.创建标注  │ ──▶ │ 3.创建并训练 │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                                                ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 7.展示结果图 │ ◀── │ 5/6.单张/批量│ ◀── │ 4.轮询训练  │
└─────────────┘     │    推理      │     │    进度     │
                    └─────────────┘     └─────────────┘
```

### 关键设计

- **用户隔离**：每个用户的数据独立存储在 `users/{user_id}/` 下，通过 `X-API-Key` 或 `Authorization: Bearer <key>` 识别用户。
- **异步训练**：训练在后台线程执行，创建模型接口立即返回 `job_id`，前端通过 `/models/jobs/{job_id}` 轮询进度。
- **基于 Embedding 训练**：模型训练不直接读取原始影像，而是读取预生成的 embedding 特征，训练速度快、资源占用低。
- **支持两类模型**：
  - `classification`：单期影像分类，适用于 `building_extraction`、`land_use_classification`、`land_cover_classification`、`water_extraction`。
  - `change_detection`：两期影像变化检测，适用于 `change_detection`。

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
| `period` | 对比期 | `2025-04_vs_2025-06` |
| `task_type` | 任务类型 | `change_detection`、`building_extraction`、`land_use_classification` |
| `model_type` | 模型类型 | `classification`、`change_detection` |

---

## Step 1：创建分类（Class）

### 用途

为标注定义类别。例如「建筑用地」、「水体」、「变化区域」等。

### 请求

```http
POST /annotations/classes
Content-Type: application/json
```

```json
{
  "name": "建筑用地",
  "color": "#FF0000"
}
```

### 响应

```json
{
  "id": "cls_abc123",
  "name": "建筑用地",
  "color": "#FF0000",
  "created_at": "2025-06-26T10:00:00"
}
```

### 前端提示

- 建议在页面上提供「类别管理」模块，让用户创建/修改/删除类别。
- 创建标注前，必须先有至少一个类别。

---

## Step 2：创建标注（Annotation）

### 用途

用户在 Patch 上绘制掩膜、多边形或折线，保存为训练样本。

### 请求

```http
POST /annotations
Content-Type: application/json
```

#### 2.1 掩膜标注（Mask）

```json
{
  "region_id": "harbin",
  "patch_id": "patch_000000",
  "month": "2025-04",
  "class_id": "cls_abc123",
  "task_type": "building_extraction",
  "geometry": {
    "type": "mask",
    "mask_b64": "iVBORw0KGgoAAAANSUhEUgAA..."
  }
}
```

#### 2.2 多边形标注（Polygon）

```json
{
  "region_id": "harbin",
  "patch_id": "patch_000000",
  "month": "2025-04",
  "class_id": "cls_abc123",
  "task_type": "building_extraction",
  "geometry": {
    "type": "polygon",
    "points": [
      [50, 50],
      [200, 50],
      [200, 200],
      [50, 200]
    ]
  }
}
```

#### 2.3 折线标注（Polyline）

```json
{
  "region_id": "harbin",
  "patch_id": "patch_000000",
  "month": "2025-04",
  "class_id": "cls_abc123",
  "task_type": "building_extraction",
  "geometry": {
    "type": "polyline",
    "points": [
      [50, 50],
      [150, 100],
      [200, 200]
    ]
  }
}
```

### 响应

```json
{
  "id": "ann_def456",
  "region_id": "harbin",
  "patch_id": "patch_000000",
  "month": "2025-04",
  "class_id": "cls_abc123",
  "task_type": "building_extraction",
  "geometry": {
    "type": "mask",
    "mask_b64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "mask_path": "users/default/annotations/masks/ann_def456.npz"
  },
  "created_at": "2025-06-26T10:05:00"
}
```

### 前端提示

- 标注绘制完成后，把 256×256 的 mask 转成 base64 PNG 字符串传给后端。
- 多边形/折线的 `points` 是像素坐标，范围 `[0, 255]`。
- 建议给用户提供「撤销/删除」功能，调用 `DELETE /annotations/{ann_id}`。

---

## Step 3：创建模型并启动训练

### 用途

用户选择标注数据，创建自定义模型并启动训练任务。

### 请求

```http
POST /models
Content-Type: application/json
```

#### 3.1 分类模型（如建筑提取）

```json
{
  "name": "哈尔滨建筑提取模型",
  "model_type": "classification",
  "task_type": "building_extraction",
  "region_id": "harbin",
  "class_ids": ["cls_abc123"],
  "epochs": 20,
  "description": "基于用户标注训练的建筑提取模型"
}
```

#### 3.2 变化检测模型

```json
{
  "name": "哈尔滨变化检测模型",
  "model_type": "change_detection",
  "task_type": "change_detection",
  "region_id": "harbin",
  "class_ids": ["cls_abc123"],
  "epochs": 20,
  "description": "基于用户标注训练的两期变化检测模型"
}
```

### 响应

```json
{
  "id": "model_xyz789",
  "name": "哈尔滨建筑提取模型",
  "model_type": "classification",
  "task_type": "building_extraction",
  "status": "training",
  "region_id": "harbin",
  "class_ids": ["cls_abc123"],
  "created_at": "2025-06-26T10:10:00",
  "job_id": "job_jkl012"
}
```

### 后端做了什么

1. 读取该用户的所有标注数据。
2. 根据 `model_type` 选择训练引擎：
   - `classification`：用单期 embedding + mask 训练 `LogisticRegression`。
   - `change_detection`：用两期 embedding 差分 + 变化 mask 训练 `LogisticRegression`。
3. 保存模型文件到 `users/{user_id}/models/model_xyz789/model.pkl`。
4. 保存模型元数据到 `users/{user_id}/models/registry.json`。
5. 启动后台训练线程，返回 `job_id`。

### 前端提示

- 创建模型后，立即拿到 `job_id`，进入轮询状态。
- 如果 `class_ids` 为空数组，系统默认使用所有类别。

---

## Step 4：轮询训练进度

### 用途

训练是异步的，前端需要轮询 `/models/jobs/{job_id}` 获取进度。

### 请求

```http
GET /models/jobs/job_jkl012
```

### 响应（训练中）

```json
{
  "job_id": "job_jkl012",
  "status": "running",
  "progress": 0.6,
  "message": "Training classifier...",
  "started_at": "2025-06-26T10:10:01",
  "updated_at": "2025-06-26T10:10:05"
}
```

### 响应（完成）

```json
{
  "job_id": "job_jkl012",
  "status": "completed",
  "progress": 1.0,
  "message": "Training completed",
  "started_at": "2025-06-26T10:10:01",
  "updated_at": "2025-06-26T10:10:10"
}
```

### 响应（失败）

```json
{
  "job_id": "job_jkl012",
  "status": "failed",
  "progress": 0.0,
  "message": "No training samples found for class cls_abc123",
  "started_at": "2025-06-26T10:10:01",
  "updated_at": "2025-06-26T10:10:03"
}
```

### 前端提示

- 建议每隔 2-3 秒轮询一次。
- 状态为 `completed` 后，可以调用 `GET /models/{model_id}` 查看最新模型信息。
- 状态为 `failed` 时，展示 `message` 给用户。

---

## Step 5：单张推理

### 用途

对指定 Patch 运行模型推理，生成结果图。

### 请求

```http
POST /models/model_xyz789/infer
Content-Type: application/json
```

#### 5.1 分类任务

```json
{
  "region_id": "harbin",
  "patch_id": "patch_000001",
  "month": "2025-04"
}
```

#### 5.2 变化检测任务

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
  "status": "success",
  "result_file": "infer_model_xyz789_harbin_patch_000001_2025-04.png",
  "region_id": "harbin",
  "patch_id": "patch_000001",
  "task_type": "building_extraction"
}
```

### 前端提示

- 模型状态必须为 `ready` 才能推理。
- 结果图会自动保存到 `users/{user_id}/results/` 目录。

---

## Step 6：批量推理

### 用途

一次推理多个 Patch，最多支持 100 个。

### 请求

```http
POST /models/model_xyz789/infer_batch
Content-Type: application/json
```

```json
{
  "requests": [
    {"region_id": "harbin", "patch_id": "patch_000000", "month": "2025-04"},
    {"region_id": "harbin", "patch_id": "patch_000001", "month": "2025-04"},
    {"region_id": "harbin", "patch_id": "patch_000002", "month": "2025-04"}
  ]
}
```

### 响应

```json
{
  "status": "success",
  "results": [
    {
      "region_id": "harbin",
      "patch_id": "patch_000000",
      "month": "2025-04",
      "result_file": "infer_model_xyz789_harbin_patch_000000_2025-04.png"
    },
    {
      "region_id": "harbin",
      "patch_id": "patch_000001",
      "month": "2025-04",
      "result_file": "infer_model_xyz789_harbin_patch_000001_2025-04.png"
    },
    {
      "region_id": "harbin",
      "patch_id": "patch_000002",
      "month": "2025-04",
      "result_file": "infer_model_xyz789_harbin_patch_000002_2025-04.png"
    }
  ]
}
```

### 前端提示

- 批量推理适合地图可视化工况：用户框选一片区域，一次性推理所有 Patch。
- 如果某个 Patch 失败，`results` 中对应条目会包含 `error` 字段。

---

## Step 7：展示结果图

### 用途

获取推理生成的 PNG 结果图，用于前端展示。

### 请求

```http
GET /models/results/infer_model_xyz789_harbin_patch_000001_2025-04.png
```

### 响应

直接返回 PNG 图片，Content-Type 为 `image/png`。

### 前端展示方式

```html
<img src="http://60.31.21.42:22065/models/results/infer_model_xyz789_harbin_patch_000001_2025-04.png" alt="推理结果" />
```

或者把图片叠加在地图上作为图层：

```javascript
// 以 Leaflet 为例
L.imageOverlay(
  'http://60.31.21.42:22065/models/results/infer_model_xyz789_harbin_patch_000001_2025-04.png',
  [[45.74, 126.5], [45.76, 126.55]]
).addTo(map);
```

---

## 状态流转说明

### 模型状态

```text
pending ──▶ training ──▶ ready
              │
              ▼
           failed
```

| 状态 | 含义 | 可执行操作 |
|------|------|------------|
| `pending` | 已创建，等待训练 | 可删除 |
| `training` | 训练中 | 可查看进度 |
| `ready` | 训练完成，可使用 | 可推理、可删除 |
| `failed` | 训练失败 | 可查看错误信息、可删除 |

### 训练任务状态

```text
queued ──▶ running ──▶ completed
             │
             ▼
           failed
```

---

## 错误处理

### 常见错误码

| HTTP 状态码 | 场景 | 前端处理 |
|-------------|------|----------|
| 400 | 请求参数错误，如缺少字段、无效 geometry | 展示具体错误信息 |
| 401 | API Key 无效或缺失 | 提示用户登录或检查 API Key |
| 404 | 模型/标注/类别不存在 | 检查 ID 是否正确 |
| 409 | 模型正在训练中，无法重复训练 | 等待当前训练完成 |
| 422 | Pydantic 校验失败 | 根据返回 detail 修正字段 |
| 500 | 服务器内部错误 | 联系后端排查 |

### 典型错误示例

**没有训练样本**

```json
{
  "detail": "No training samples found for class cls_abc123"
}
```

**模型未就绪**

```json
{
  "detail": "Model model_xyz789 is not ready (status: training)"
}
```

**批量推理超限**

```json
{
  "detail": "Batch size exceeds maximum of 100"
}
```

---

## 完整前端代码示例

以下是一个简化的 Vue/React 风格伪代码，展示完整流程。

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

// 1. 创建分类
const cls = await request('/annotations/classes', 'POST', {
  name: '建筑用地',
  color: '#FF0000',
});

// 2. 创建标注（mask 为例）
const maskBase64 = getMaskBase64FromCanvas(); // 前端 Canvas 导出
const ann = await request('/annotations', 'POST', {
  region_id: 'harbin',
  patch_id: 'patch_000000',
  month: '2025-04',
  class_id: cls.id,
  task_type: 'building_extraction',
  geometry: {
    type: 'mask',
    mask_b64: maskBase64,
  },
});

// 3. 创建模型并训练
const model = await request('/models', 'POST', {
  name: '哈尔滨建筑提取模型',
  model_type: 'classification',
  task_type: 'building_extraction',
  region_id: 'harbin',
  class_ids: [cls.id],
  epochs: 20,
});

// 4. 轮询训练进度
const pollTraining = setInterval(async () => {
  const job = await request(`/models/jobs/${model.job_id}`);
  updateProgress(job.progress);
  if (job.status === 'completed') {
    clearInterval(pollTraining);
    runInference(model.id);
  } else if (job.status === 'failed') {
    clearInterval(pollTraining);
    alert('训练失败：' + job.message);
  }
}, 2000);

// 5. 单张推理
async function runInference(modelId) {
  const result = await request(`/models/${modelId}/infer`, 'POST', {
    region_id: 'harbin',
    patch_id: 'patch_000001',
    month: '2025-04',
  });
  // 6. 展示结果图
  const imgUrl = `${BASE_URL}/models/results/${result.result_file}`;
  document.getElementById('result-image').src = imgUrl;
}
```

---

## 附：接口速查表

| 步骤 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 创建分类 | POST | `/annotations/classes` | 创建训练类别 |
| 创建标注 | POST | `/annotations` | 保存标注样本 |
| 创建模型 | POST | `/models` | 创建模型并启动训练 |
| 查看进度 | GET | `/models/jobs/{job_id}` | 轮询训练状态 |
| 查看模型 | GET | `/models/{model_id}` | 获取模型信息 |
| 单张推理 | POST | `/models/{model_id}/infer` | 推理一个 Patch |
| 批量推理 | POST | `/models/{model_id}/infer_batch` | 批量推理 |
| 获取结果图 | GET | `/models/results/{filename}` | 下载 PNG 结果 |

---

## 版本记录

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2025-06-26 | 初稿，覆盖自定义训练完整工作流 |
