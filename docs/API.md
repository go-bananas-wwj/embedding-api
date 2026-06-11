# Embedding API 接口文档 v1.0

> **文档版本**: 1.0  
> **服务版本**: 0.1.0  
> **最后更新**: 2026-06-10  
> **GitHub**: [go-bananas-wwj/embedding-api](https://github.com/go-bananas-wwj/embedding-api)

---

## 📖 阅读指南

本文档面向**前端开发团队**，用通俗语言解释每一个接口的用途、参数和返回值。如果你遇到不理解的术语，先看下面的「核心概念」章节。

---

## 🧠 核心概念

### 什么是 Patch？

Patch（图块）是我们把一整块大区域切分成的小方块。每个 Patch 对应地球上的一块矩形区域，有固定的经纬度范围。

- **哈尔滨新区**: 共 424 个 Patch，覆盖约 127km²
- **海淀区**: 共 320 个 Patch，覆盖约 80km²
- 每个 Patch 都有唯一 ID，如 `patch_000000`

**类比**: Patch 就像地图上的一个个「瓦片格子」，前端地图上的每个小方块可能对应一个 Patch。

### 什么是 Embedding（嵌入）？

Embedding 是深度学习模型把卫星影像「翻译」成的高维向量。可以理解为：模型「看懂」了这张图，并用一串数字记录了它的特征。

- **哈尔滨**: 嵌入以 **PNG 图片**形式存储（64×64 像素，RGB 三通道），便于直接展示
- **海淀**: 嵌入以 **NPY 数组**形式存储（64 通道 × 128×128），是原始数学向量

**类比**: 如果卫星影像是一篇文章，Embedding 就是这篇文章的「摘要」—— 浓缩了关键信息，但不可直接阅读。

### 什么是下游任务？

下游任务是在 Embedding 基础上做的「具体监测」，比如：

| 任务英文名 | 中文名 | 监测内容 |
|-----------|--------|---------|
| `construction` | 建筑工地监测 | 哪里在新建工地 |
| `building_change` | 建筑变化监测 | 哪里新建了楼房 |
| `farmland` | 耕地非农非粮监测 | 耕地是否被非法占用 |
| `land_conversion` | 土地转换监测 | 土地用途是否改变 |
| `change_detection` | 变化检测 | 基于两期 embedding 差分检测变化区域 |

**版本说明**:
- **V1**: 单期监测，只有一个时间点的结果
- **V2**: 变化监测，对比两个时间点的差异（如 `2025-04_vs_2025-06`）

### 什么是瓦片（Tile）？

瓦片是切成小片的 PNG 图片，前端地图库（如 Leaflet、Mapbox）可以直接叠加到地图上显示监测结果。

---

## 🔧 快速开始

### Base URL

**生产环境**:
```
http://60.31.21.42:22065
```

**本地开发**:
```
http://localhost:9061
```

### 在线文档

- **Swagger UI**（交互式调试）: `http://60.31.21.42:22065/docs`
- **ReDoc**（美观文档）: `http://60.31.21.42:22065/redoc`

### 启动服务

```bash
# 方式一：直接运行
cd embedding-api
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 9061

# 方式二：Docker
docker-compose up -d
```

> **环境变量配置**（可选，生产环境建议设置）：
> - `CONFIG_FILE` — 指定配置文件路径，默认 `./config.yaml`
> - `LOG_LEVEL` — 日志级别，默认 `INFO`
> - `TZ` — 时区，默认 `Asia/Shanghai`
>
> **CORS**：服务已配置跨域，前端可直接从浏览器调用。如需限制特定域名，修改 `app/main.py` 中的 `CORSMiddleware` 配置。

---

## 📡 接口总览

| 分类 | 接口 | 说明 |
|------|------|------|
| **基础** | `GET /health` | 服务健康检查 |
| **区域** | `GET /regions` | 列出所有区域 |
| **区域** | `GET /regions/{region_id}` | 区域详情 |
| **Patch** | `GET /regions/{region_id}/patches` | Patch 列表（支持分页+地理过滤） |
| **Patch** | `GET /regions/{region_id}/patches/{patch_id}` | 单个 Patch 详情 |
| **嵌入** | `GET /.../embedding?format=png\|npy\|json` | 查询嵌入数据 |
| **任务** | `GET /regions/{region_id}/tasks` | 列出下游任务 |
| **任务** | `GET /.../tasks/{task_type}/summary` | 任务统计摘要 |
| **任务** | `GET /.../tasks/{task_type}/result` | 单 Patch 结果图 |
| **任务** | `GET /.../tasks/{task_type}/prediction` | 原始预测数据 |
| **任务** | `GET /.../tasks/{task_type}/label` | 标签数据 |
| **瓦片** | `GET /.../tasks/{task_type}/tiles` | 列出可用瓦片 |
| **瓦片** | `GET /.../tiles/{z}/{x}/{y}.png` | 地图瓦片 |

---

## 🔍 详细接口

### 1. 健康检查

检查服务是否正常运行。

```
GET /health
```

**什么时候用**: 页面加载时检查后端可用性，或服务监控心跳。

**请求参数**: 无

**成功响应** (200):
```json
{
  "status": "ok",
  "version": "0.1.0",
  "regions": ["harbin", "haidian"]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 服务状态，"ok" 表示正常 |
| `version` | string | API 版本号 |
| `regions` | string[] | 当前支持的区域列表 |

---

### 2. 列出所有区域

获取系统支持的所有监测区域，以及每个区域的基本信息。

```
GET /regions
```

**什么时候用**: 前端页面初始化时，加载区域选择下拉框。

**请求参数**: 无

**成功响应** (200):
```json
{
  "regions": [
    {
      "id": "harbin",
      "name": "哈尔滨新区",
      "patch_count": 424,
      "tasks": [
        "construction",
        "building_change",
        "farmland",
        "land_conversion",
        "demolition"
      ]
    },
    {
      "id": "haidian",
      "name": "海淀区",
      "patch_count": 320,
      "tasks": []
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 区域标识符，用于后续接口 |
| `name` | string | 区域中文名 |
| `patch_count` | int | 该区域包含的 Patch 总数 |
| `tasks` | string[] | 该区域支持的下游任务列表 |

**注意**: 海淀区 `tasks` 为空，因为下游任务尚未完成。后续添加任务后自动更新。

---

### 3. 获取区域详情

获取某个区域的详细信息，包括所有下游任务的版本信息。

```
GET /regions/{region_id}
```

**什么时候用**: 用户选择某个区域后，展示该区域的任务菜单。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID，如 `harbin` 或 `haidian` |

**成功响应** (200):
```json
{
  "id": "harbin",
  "name": "哈尔滨新区",
  "patch_count": 424,
  "tasks": {
    "construction": {
      "name": "建筑工地监测",
      "description": "监测建筑工地变化情况",
      "versions": ["v1", "v2"]
    },
    "building_change": {
      "name": "建筑变化监测",
      "description": "监测建筑物变化情况",
      "versions": ["v1"]
    }
  },
  "embeddings": ["v2"]
}
```

**错误响应**:
- `404`: 区域不存在

---

### 4. 列出 Patch

分页获取某个区域的所有 Patch，支持**地理范围过滤**。

```
GET /regions/{region_id}/patches?page=1&page_size=20&bbox=minx,miny,maxx,maxy
```

**什么时候用**:
- 前端地图初始化时加载可见区域的 Patch
- 搜索框输入地理范围后筛选 Patch
- 分页展示 Patch 列表

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `page` | int | 否 | 1 | 页码，从 1 开始 |
| `page_size` | int | 否 | 20 | 每页数量，**最大 100**，超过将返回 422 错误 |
| `bbox` | string | 否 | - | 地理范围过滤，格式：`minx,miny,maxx,maxy`（WGS84 坐标系） |

**坐标顺序**：`minx,miny,maxx,maxy` 分别对应 **最小经度、最小纬度、最大经度、最大纬度**。

**bbox 示例**:
```
# 过滤哈尔滨新区中心区域
bbox=126.5,45.74,126.55,45.76
```

**成功响应** (200):
```json
{
  "total": 424,
  "page": 1,
  "page_size": 20,
  "patches": [
    {
      "patch_id": "patch_000000",
      "bounds_wgs84": [126.51631, 45.743707, 126.532242, 45.755574],
      "sources": {
        "dem": 1,
        "dynamic_world": 13,
        "jrc_water": 1,
        "landsat": 38,
        "modis_lst": 13,
        "modis_ndvi": 13,
        "s1": 100,
        "s1_hr": 4,
        "s2": 182,
        "s2_hr": 5,
        "worldcover": 1
      },
      "time_range": ["2023-01", "2025-10"],
      "has_embedding": true,
      "available_tasks": ["construction", "building_change", "farmland"]
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `total` | int | 符合条件的 Patch 总数 |
| `page` | int | 当前页码 |
| `page_size` | int | 每页数量 |
| `has_next` | bool | 是否有下一页（`total > page * page_size` 时为 `true`） |
| `patch_id` | string | Patch 唯一标识 |
| `bounds_wgs84` | float[4] | 经纬度范围 `[min_lng, min_lat, max_lng, max_lat]` |
| `sources` | object | 各数据源包含的影像/样本数量（如 `s2: 182` 表示 Sentinel-2 有 182 个时间切片） |
| `time_range` | string[2] | 数据时间范围 `[开始年月, 结束年月]`，格式为 `YYYY-MM` |
| `has_embedding` | bool | 是否有嵌入数据 |
| `available_tasks` | string[] | 该 Patch 有结果的下游任务（前端可用此字段动态渲染「查看任务」按钮，只显示有数据的任务） |

**前端提示**: `bounds_wgs84` 可以直接用于在地图上绘制矩形框，表示 Patch 的位置。前端可通过 `has_next` 判断是否需要显示「加载更多」按钮。

---

### 5. 获取 Patch 详情

获取单个 Patch 的完整信息。

```
GET /regions/{region_id}/patches/{patch_id}
```

**什么时候用**: 用户点击地图上的某个 Patch 时，展示该 Patch 的详细信息。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `patch_id` | string | 是 | Patch ID，如 `patch_000000` |

**成功响应**: 与「列出 Patch」中的单个 Patch 结构相同。

**错误响应**:
- `404`: Patch 不存在

---

### 6. 获取嵌入数据

获取某个 Patch 的 Embedding（嵌入向量）。支持三种返回格式。

```
GET /regions/{region_id}/patches/{patch_id}/embedding?format=png
```

**什么时候用**:
- `format=png`: 前端展示嵌入的可视化热图
- `format=npy`: 下载原始数据用于进一步分析
- `format=json`: 快速查看嵌入的统计信息

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `patch_id` | string | 是 | Patch ID |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `format` | string | 否 | `png` | 输出格式：`png`、`npy`、`json`、`cache` |
| `version` | string | 否 | - | Embedding 版本：`v1`（V4 模型）/`v2`（V5 模型）；不传时依次尝试可用版本 |
| `month` | string | 否 | - | 时间序列月份，如 `2025-04`（哈尔滨）或 `20251201`（海淀）；不传时自动回退到该 Patch 第一个可用月份 |

**四种格式的区别**:

#### format=png（默认）

返回 **PNG 图片**，Content-Type: `image/png`

- **哈尔滨**: 64×64 像素的 RGB 可视化图
- **海淀**: 如果有可视化图则返回，否则尝试返回多源数据合成图

**前端用法**: 直接放在 `<img>` 标签里展示。

```html
<img src="http://60.31.21.42:22065/regions/harbin/patches/patch_000000/embedding?format=png" />
```

#### format=npy

返回 **NPY 二进制文件**，Content-Type: `application/octet-stream`

- **哈尔滨**: 如果存在 `.npy` 文件则返回
- **海淀**: 返回 64×128×128 的 float32 数组

**前端用法**: 需要在前端用 JavaScript 解析 numpy 格式（可用 `jsnumpy` 库），或下载后交给 Python 后端处理。

```javascript
// 下载并解析 NPY
const response = await fetch('/regions/haidian/patches/patch_000000/embedding?format=npy');
const arrayBuffer = await response.arrayBuffer();
// 使用 jsnumpy 解析
```

#### format=cache

**自动选择可用格式**，优先返回 PNG 图片，没有 PNG 则返回 NPY 二进制。

- **适用场景**：前端不确定某个 Patch 有哪些格式时，让后端自动选择
- **返回 Content-Type**：根据实际返回的数据决定（`image/png` 或 `application/octet-stream`）

```html
<!-- 自动选择最佳格式展示 -->
<img src="/regions/harbin/patches/patch_000000/embedding?format=cache" />
```

#### format=json

返回 **JSON 统计信息**，Content-Type: `application/json`

```json
{
  "patch_id": "patch_000000",
  "shape": [60, 8],
  "dtype": "float32",
  "min": -0.476,
  "max": 0.404,
  "mean": -0.005
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `shape` | int[] | 数组维度。实际值因区域和模型版本而异，如 `[60, 8]` 表示 60 个通道 × 8 个特征 |
| `dtype` | string | 数据类型，`float32` 或 `uint8` |
| `min` | float | 最小值 |
| `max` | float | 最大值 |
| `mean` | float | 平均值 |

**前端提示**: `shape` 的维度含义因模型架构不同而变化。Embedding 不是普通 RGB 图片，每个维度代表模型提取的一个特征分量。如需要具体语义，请联系后端团队获取模型说明。

---

### 7. 列出下游任务

获取某个区域支持的所有下游监测任务。

```
GET /regions/{region_id}/tasks
```

**什么时候用**: 前端侧边栏或下拉菜单展示可选的监测任务类型。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |

**成功响应** (200):
```json
{
  "tasks": [
    {
      "id": "construction",
      "name": "建筑工地监测",
      "description": "监测建筑工地变化情况",
      "versions": ["v1", "v2"]
    },
    {
      "id": "building_change",
      "name": "建筑变化监测",
      "description": "监测建筑物变化情况",
      "versions": ["v1"]
    },
    {
      "id": "farmland",
      "name": "耕地非农非粮监测",
      "description": "监测耕地被非法占用情况",
      "versions": ["v1"]
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 任务标识符，用于后续接口 |
| `name` | string | 任务中文名 |
| `description` | string | 任务描述 |
| `versions` | string[] | 可用版本号列表 |

**版本说明**:
- `v1`: 基于单期数据的监测结果
- `v2`: 基于两期对比的变化检测结果，需要指定 `period` 参数

---

### 8. 任务统计摘要

获取某个下游任务的整体统计信息。

```
GET /regions/{region_id}/tasks/{task_type}/summary?version=v1&period=2025-04_vs_2025-06
```

**什么时候用**: 前端仪表盘展示「该任务共发现多少个图斑、多少个 Patch 有异常」等统计数字。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `task_type` | string | 是 | 任务类型，如 `construction` |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `version` | string | 否 | `v1` | 版本号 |
| `period` | string | 否 | - | 时间周期，部分任务版本需要指定，如 `2025-04_vs_2025-06` |

**成功响应** (200):
```json
{
  "task": "construction",
  "name": "建筑工地监测",
  "version": "v2",
  "period": "2025-04_vs_2025-06",
  "grid_size": 64,
  "total_polygons": 24,
  "total_patches": 424,
  "positive_patches": 31,
  "negative_patches": 393
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `task` | string | 任务英文名 |
| `name` | string | 任务中文名 |
| `version` | string | 版本号 |
| `period` | string | 时间段（V2 特有） |
| `grid_size` | int | 网格尺寸（像素） |
| `total_polygons` | int | 总共发现的变化图斑数量 |
| `total_patches` | int | 该区域 Patch 总数 |
| `positive_patches` | int | 有异常的 Patch 数量 |
| `negative_patches` | int | 无异常的 Patch 数量 |

**业务含义**: `positive_patches` / `total_patches` 就是异常检出率。例如 31/424 ≈ 7.3% 的 Patch 发现了建筑工地。

---

### 9. 单 Patch 任务结果

获取某个 Patch 在特定下游任务上的结果图。

```
GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result?format=png&version=v1&period=...
```

**什么时候用**: 用户点击某个 Patch，查看该 Patch 的监测结果可视化图。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `region_id` | string | 是 | 区域 ID |
| `patch_id` | string | 是 | Patch ID |
| `task_type` | string | 是 | 任务类型 |

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `format` | string | 否 | `png` | `png` 返回图片，`npy` 返回原始数组 |
| `version` | string | 否 | `v1` | 版本号 |
| `period` | string | 否 | - | 时间周期（如 `2025-10`），部分任务版本需要指定 |

**成功响应**:
- `format=png`: 返回 PNG 图片 (`image/png`)
- `format=npy`: 返回 NPY 文件 (`application/octet-stream`)

**前端用法**:
```html
<img src="/regions/harbin/patches/patch_000000/tasks/construction/result?format=png" />
```

**错误响应**:
- `404`: 该 Patch 在该任务下没有结果

---

### 10. 原始预测数据

下载某个 Patch 的原始预测数组（未经阈值化的概率图）。

```
GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/prediction?version=v1&period=...
```

**什么时候用**: 前端需要做自定义可视化（如调整阈值显示不同置信度区域），或下载给分析师进一步处理。

**返回**: NPY 二进制文件 (`application/octet-stream`)

**数据说明**: 预测值范围通常在 0~1 之间，表示模型认为该像素属于目标类别的概率。值越高，置信度越高。

**前端提示**: 可以用 JavaScript 的 `numpy-js` 或 `numjs` 库解析 NPY 文件。

---

### 11. 标签数据

获取某个 Patch 的真实标签（Ground Truth）。标签来源可能是人工标注或自动生成。

```
GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/label?version=v1&period=...
```

**什么时候用**: 对比「模型预测」和「真实标签」，计算准确率或生成混淆矩阵。

**返回**:
- 如果存在 `.npy` 标签文件：返回 NPY 二进制数组
- 如果存在 `meta.json` 元数据文件：返回 JSON 元数据
- 如果两者都不存在：返回 `404`

---

### 12. 列出可用瓦片

获取某个任务的所有瓦片文件列表。

```
GET /regions/{region_id}/tasks/{task_type}/tiles?version=v1&period=...
```

**什么时候用**: 前端地图组件初始化时，预加载瓦片索引。

**成功响应**:
```json
{
  "tiles": [
    {"patch_id": "patch_000000", "period": "2025-10", "filename": "patch_000000_2025-10.png"},
    {"patch_id": "patch_000001", "period": "2025-10", "filename": "patch_000001_2025-10.png"}
  ],
  "total": 2
}
```

---

### 13. 获取地图瓦片

标准的 XYZ 瓦片接口，可被 Leaflet、Mapbox、OpenLayers 等地图库直接使用。

```
GET /regions/{region_id}/tasks/{task_type}/tiles/{z}/{x}/{y}.png?version=v1&period=...
```

**什么时候用**: 在地图上叠加监测结果图层。

**路径参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `z` | int | 是 | 缩放级别 (zoom level) |
| `x` | int | 是 | 瓦片 X 坐标 |
| `y` | int | 是 | 瓦片 Y 坐标 |

**前端用法** (Leaflet 示例):
```javascript
L.tileLayer(
  '/regions/harbin/tasks/construction/tiles/{z}/{x}/{y}.png?version=v1',
  { opacity: 0.7 }
).addTo(map);
```

**注意**: 当前版本瓦片按 Patch 切分，非标准 XYZ 切片。如需标准地图瓦片服务，后续可扩展。

---

## ⚠️ 错误处理

### 统一错误格式

所有错误返回统一的 JSON 结构：

```json
{
  "detail": "错误描述信息"
}
```

### 状态码说明

| 状态码 | 含义 | 场景 |
|--------|------|------|
| `200` | 成功 | 请求正常处理 |
| `400` | 请求格式错误 | Patch ID 格式非法、路径穿越尝试 |
| `404` | 未找到 | 区域/Patch/任务/结果不存在 |
| `413` | 请求体过大 | 文件超过 1GB 限制或图像超过 5000 万像素 |
| `422` | 参数校验失败 | 查询参数格式不正确（如 page=-1、bbox=nan、page_size=999） |
| `500` | 服务器错误 | 内部异常，需联系后端排查 |
| `501` | 未实现 | 瓦片服务等功能尚未实现 |
| `503` | 服务不可用 | 服务正在启动中或重启中，稍后重试 |

> **注意**：请求 `/regions/{region_id}/patches/{patch_id}/embedding?format=png` 时，如果该区域该 Patch 只有 `.npy` 格式，响应头会包含 `X-Available-Format: npy`，提示前端可以请求 `format=npy` 获取原始数据。

### 常见错误场景

**区域不存在**:
```bash
GET /regions/beijing
# → 404 {"detail": "Region 'beijing' not found"}
```

**Patch 不存在**:
```bash
GET /regions/harbin/patches/patch_999999
# → 404 {"detail": "Patch 'patch_999999' not found"}
```

**嵌入不存在**:
```bash
GET /regions/haidian/patches/patch_999999/embedding
# → 404 {"detail": "Embedding not found for patch 'patch_999999'"}
```

---

## 🔧 配置与扩展

### 配置文件位置

```
config.yaml
```

### 热重载机制

服务通过 `watchdog` 监听 `config.yaml` 的文件变化。修改配置后**无需重启服务**，约 1 秒内自动生效。

### 添加新区域

在 `config.yaml` 的 `regions` 下添加：

```yaml
regions:
  shenzhen:
    name: "深圳"
    patches_meta: "/data/shenzhen/patches_meta.json"
    embeddings:
      v1: "/data/shenzhen/embeddings"
    tasks:
      construction:
        name: "建筑工地监测"
        versions:
          v1:
            results: "/data/shenzhen/results/construction"
            predictions: "/data/shenzhen/predictions/construction"
            labels: "/data/shenzhen/labels/construction"
```

保存后，访问 `/regions` 即可看到新区域。

### 添加新任务

在对应区域的 `tasks` 下添加：

```yaml
      green_space:
        name: "绿地变化监测"
        description: "监测城市绿地变化"
        versions:
          v1:
            results: "/data/results/green_space"
            predictions: "/data/predictions/green_space"
            labels: "/data/labels/green_space"
```

---

## 💡 前端集成最佳实践

### 1. 页面加载流程

```
1. GET /health          → 检查服务可用
2. GET /regions         → 加载区域选择器
3. GET /regions/{id}/patches?page=1&bbox={map_view}  → 加载地图可见区域的 Patch
4. 用户点击 Patch → GET /regions/{id}/patches/{pid} → 展示 Patch 详情
```

### 2. 地图叠加流程

```
1. GET /regions/{id}/tasks → 加载任务选择器
2. 用户选择任务 → GET /regions/{id}/tasks/{task}/tiles → 加载瓦片列表
3. 地图移动时 → GET /.../tiles/{z}/{x}/{y}.png → 动态加载瓦片
```

### 3. 前端 Fetch 示例

```typescript
// 1. 配置基地址（开发/生产环境切换）
const API_BASE = process.env.VITE_API_URL || 'http://60.31.21.42:22065';

// 2. 获取带 bbox 过滤的 Patch 列表
async function fetchPatches(regionId: string, bbox: string, page = 1, pageSize = 20) {
  const url = new URL(`${API_BASE}/regions/${regionId}/patches`);
  url.searchParams.set('page', String(page));
  url.searchParams.set('page_size', String(pageSize));
  if (bbox) url.searchParams.set('bbox', bbox);

  const resp = await fetch(url.toString());
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(`API ${resp.status}: ${err.detail}`);
  }
  return resp.json(); // { total, page, page_size, patches, has_next }
}

// 3. 加载任务结果图
function getResultImageUrl(regionId: string, patchId: string, taskType: string) {
  return `${API_BASE}/regions/${regionId}/patches/${patchId}/tasks/${taskType}/result?format=png`;
}
```

### 4. 性能建议

- **分页加载**: Patch 数量多（400+），始终使用分页，通过 `has_next` 判断是否还有更多数据
- **bbox 过滤**: 地图移动时只请求可视区域的 Patch，减少数据传输
- **图片缓存**: PNG 结果图可设置浏览器缓存（内容不变），避免重复请求
- **懒加载**: 嵌入大图和任务结果图按需加载，不要一次性请求所有 Patch
- **错误降级**: 遇到 `404` 时不应中断整个应用，而是显示「暂无数据」提示；遇到 `503` 时提示用户稍后重试

---

## 📊 数据速查表

### 哈尔滨新区 (harbin)

| 属性 | 数值 |
|------|------|
| Patch 总数 | 424 |
| 覆盖范围 | 126.5°E ~ 126.7°E, 45.7°N ~ 45.8°N |
| 时间范围 | 2023-01 ~ 2025-10 |
| 数据源 | Sentinel-2, Sentinel-1, Landsat, DEM, WorldCover 等 |
| 嵌入格式 | PNG 64×64 RGB |
| 下游任务 | 5 个（construction, building_change, farmland, land_conversion, change_detection） |

### 海淀区 (haidian)

| 属性 | 数值 |
|------|------|
| Patch 总数 | 320 |
| 覆盖范围 | 116.2°E ~ 116.3°E, 39.9°N ~ 40.1°N |
| 时间范围 | 2025-02 ~ 2026-04 |
| 数据源 | Sentinel-2, Sentinel-1, Landsat, DEM, WorldCover 等 |
| 嵌入格式 | NPY 64×128×128 float32 |
| 下游任务 | 暂无（预留扩展） |

---

## 📞 技术支持

- **GitHub Issues**: [go-bananas-wwj/embedding-api/issues](https://github.com/go-bananas-wwj/embedding-api/issues)
- **服务监控**: `GET /health`
- **在线调试**: `http://60.31.21.42:22065/docs`
- **本地调试**: `http://localhost:9061/docs`
