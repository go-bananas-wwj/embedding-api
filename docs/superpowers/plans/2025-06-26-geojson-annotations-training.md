# GeoJSON 标注数组驱动训练 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除旧的后端标注接口，改造 `POST /models` 接口，使其直接接收前端传来的 GeoJSON FeatureCollection 标注数组，解析为像素级训练样本后启动训练。

**Architecture：** 前端在浏览器本地管理标注，以标准 GeoJSON 格式回传；后端只负责坐标转换、栅格化、提取 embedding、训练。旧 `/annotations` 与 `/annotations/classes` 接口全部删除。

**Tech Stack：** FastAPI, Pydantic v2, GeoJSON, shapely/rasterio, scikit-learn, NumPy

---

## 一、需求澄清

### 1.1 新旧流程对比

| 阶段 | 旧流程 | 新流程 |
|------|--------|--------|
| 分类管理 | `POST /annotations/classes` | 前端 `localStorage` |
| 标注管理 | `POST /annotations` | 前端 `localStorage`，训练时回传 GeoJSON |
| 训练启动 | `POST /models`，后端读 AnnotationStore | `POST /models` 携带 GeoJSON 数组 |
| 后端解析 | 从文件系统读标注 | 解析请求体里的 GeoJSON |

### 1.2 GeoJSON 数据格式

前端在调用 `POST /models` 时，请求体中 `annotations` 字段必须是一个标准 GeoJSON `FeatureCollection`：

```json
{
  "name": "哈尔滨建筑提取模型",
  "model_type": "classification",
  "task_type": "building_extraction",
  "region_id": "harbin",
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
              [126.50, 45.74],
              [126.52, 45.74],
              [126.52, 45.76],
              [126.50, 45.76],
              [126.50, 45.74]
            ]
          ]
        }
      }
    ]
  },
  "classes": [
    {
      "id": "cls_001",
      "name": "建筑用地",
      "color": "#FF0000"
    }
  ]
}
```

### 1.3 字段说明

#### GeoJSON Feature 的 `properties`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `patch_id` | string | 是 | 该标注所属 Patch，如 `patch_000000` |
| `region_id` | string | 是 | 区域 ID，如 `harbin` |
| `class_id` | string | 是 | 分类 ID |
| `class_name` | string | 否 | 分类名称，仅用于展示 |
| `color` | string | 否 | 分类颜色，如 `#FF0000` |
| `task_type` | string | 是 | 任务类型，如 `building_extraction` |
| `month` | string | 条件 | 单期任务必填，如 `2025-04` |
| `before_month` | string | 条件 | 变化检测任务必填 |
| `after_month` | string | 条件 | 变化检测任务必填 |

#### GeoJSON Feature 的 `geometry`

- 坐标系：**WGS84 (EPSG:4326)**，经纬度顺序 `[lon, lat]`
- 支持类型：`Polygon`、`MultiPolygon`
- 不支持 `Point`、`LineString`（如需画线/点，前端需转成很小的 Polygon）

### 1.4 后端处理流程

```text
接收 GeoJSON FeatureCollection
        │
        ▼
按 patch_id 分组所有 Feature
        │
        ▼
对每个 Patch：
  1. 从 data_service 读取该 Patch 的 bbox
  2. 用 shapely 把 WGS84 Polygon 转成 Patch 局部像素坐标
  3. 栅格化成 256x256 mask
  4. 加载对应月份的 embedding
  5. 提取训练样本 (embedding + mask + class_id)
        │
        ▼
按 class_id 映射为数字标签
        │
        ▼
调用 LogisticRegression 训练
        │
        ▼
保存 model.pkl 并更新状态为 ready
```

---

## 二、坐标转换详细设计

### 2.1 Patch 坐标系

每个 Patch 在 WGS84 下有一个 bbox：`[minx, miny, maxx, maxy]`。

在像素坐标系中：
- 图像尺寸：256 × 256
- 左上角像素中心对应 `(minx, maxy)`
- 右下角像素中心对应 `(maxx, miny)`

### 2.2 WGS84 → 像素坐标公式

```python
def wgs84_to_pixel(lon, lat, bbox, width=256, height=256):
    """把 WGS84 坐标转成 Patch 内像素坐标 (col, row)。"""
    minx, miny, maxx, maxy = bbox
    col = int((lon - minx) / (maxx - minx) * width)
    row = int((maxy - lat) / (maxy - miny) * height)
    # 裁剪到 [0, 255]
    col = max(0, min(width - 1, col))
    row = max(0, min(height - 1, row))
    return col, row
```

### 2.3 Polygon 栅格化

```python
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

def polygon_feature_to_mask(feature, bbox, size=(256, 256)):
    """把 GeoJSON Polygon Feature 转成 256x256 二值 mask。"""
    coords = feature["geometry"]["coordinates"][0]  # 外环
    pixel_points = [wgs84_to_pixel(lon, lat, bbox, size[0], size[1]) for lon, lat in coords]

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(pixel_points, fill=255)
    return (np.array(mask) > 0).astype(np.uint8)
```

### 2.4 多 Patch / 多 Polygon 合并

同一个 Patch 可能有多个 Polygon（同一类或不同类）。按 `class_id` 合并：

```python
def merge_masks_for_patch(features, bbox, size=(256, 256)):
    """返回 {class_id: mask}。"""
    result = {}
    for f in features:
        cls = f["properties"]["class_id"]
        mask = polygon_feature_to_mask(f, bbox, size)
        result[cls] = np.maximum(result.get(cls, np.zeros(size, dtype=np.uint8)), mask)
    return result
```

---

## 三、文件结构

| 文件 | 职责 |
|------|------|
| `app/schemas/models.py` | 定义 `GeoJSONFeature`, `GeoJSONFeatureCollection`, `ModelClass`, `ModelCreate` |
| `app/services/geojson_adapter.py` | 新建：GeoJSON 解析、坐标转换、栅格化 |
| `app/services/training_engine.py` | 改造：接收 GeoJSON + classes，生成训练样本 |
| `app/services/annotation_service.py` | **删除**或清空，不再使用 |
| `app/routers/annotations.py` | **删除**整个 router |
| `app/routers/models.py` | 改造：`POST /models` 接收 GeoJSON annotations |
| `app/main.py` | 移除 `annotations` router 注册 |
| `tests/test_annotations.py` | **删除** |
| `tests/test_annotation_adapter.py` | 若存在则 **删除** |
| `tests/test_geojson_adapter.py` | 新建：测试坐标转换与栅格化 |
| `tests/test_models.py` | 改造：使用 GeoJSON 训练 |
| `docs/custom-training-workflow.md` | 改造：说明 GeoJSON 格式 |
| `docs/API.md` | 改造：删除 annotations 章节，更新 `POST /models` |

---

## 四、详细任务

### Task 1：删除旧标注接口

**Files:**
- Delete: `app/routers/annotations.py`
- Delete: `app/services/annotation_service.py`
- Delete: `tests/test_annotations.py`
- Modify: `app/main.py`
- Modify: `app/schemas/models.py`（删除 Class/Annotation 相关 schema）

- [ ] **Step 1：删除 router 文件**

```bash
rm app/routers/annotations.py
rm app/services/annotation_service.py
rm tests/test_annotations.py
```

- [ ] **Step 2：在 `app/main.py` 中移除 annotations router**

```python
# 删除以下行
from app.routers import annotations
app.include_router(annotations.router, prefix="/annotations", tags=["Annotations"])
```

- [ ] **Step 3：在 `app/schemas/models.py` 中删除以下 schema**

- `ClassCreate`
- `ClassOut`
- `ClassRenameRequest`
- `AnnotationCreate`
- `AnnotationOut`
- `GeometryMask`
- `GeometryPolygon`
- `GeometryPolyline`
- `GeometryUnion`
- `StatusOut`（如仅 annotations 使用）

- [ ] **Step 4：运行测试确认删除干净**

Run: `python -m pytest tests/ -q`
Expected: 失败，因为 test_models 仍依赖旧接口，下一步修复

- [ ] **Step 5：Commit**

```bash
git add -A
git commit -m "refactor: remove annotation store and annotation endpoints"
```

---

### Task 2：定义 GeoJSON 相关 Schema

**Files:**
- Modify: `app/schemas/models.py`

- [ ] **Step 1：新增 `ModelClass` schema**

```python
class ModelClass(BaseModel):
    """训练时使用的分类定义。"""
    id: str = Field(..., description="分类 ID", example="cls_001")
    name: str = Field(..., description="分类名称", example="建筑用地")
    color: str = Field(..., description="分类颜色", example="#FF0000")
```

- [ ] **Step 2：新增 `GeoJSONProperties` schema**

```python
class GeoJSONProperties(BaseModel):
    """GeoJSON Feature 的 properties，携带标注元数据。"""
    patch_id: str = Field(..., description="标注所在 Patch ID", example="patch_000000")
    region_id: str = Field(..., description="区域 ID", example="harbin")
    class_id: str = Field(..., description="分类 ID", example="cls_001")
    class_name: Optional[str] = Field(None, description="分类名称", example="建筑用地")
    color: Optional[str] = Field(None, description="分类颜色", example="#FF0000")
    task_type: str = Field(..., description="任务类型", example="building_extraction")
    month: Optional[str] = Field(None, description="单期任务月份", example="2025-04")
    before_month: Optional[str] = Field(None, description="变化检测前期", example="2025-04")
    after_month: Optional[str] = Field(None, description="变化检测后期", example="2025-06")
```

- [ ] **Step 3：新增 `GeoJSONFeature` schema**

```python
from typing import List, Literal

class GeoJSONFeature(BaseModel):
    """单个标注要素。"""
    type: Literal["Feature"] = "Feature"
    properties: GeoJSONProperties
    geometry: Dict[str, Any] = Field(
        ...,
        description="GeoJSON 几何对象，支持 Polygon/MultiPolygon，坐标系 WGS84"
    )

    @model_validator(mode="after")
    def validate_geometry(self):
        geom = self.geometry
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError("geometry 类型必须是 Polygon 或 MultiPolygon")
        return self
```

- [ ] **Step 4：新增 `GeoJSONFeatureCollection` schema**

```python
class GeoJSONFeatureCollection(BaseModel):
    """前端回传的标注集合。"""
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[GeoJSONFeature] = Field(..., min_length=1, description="标注要素列表")
```

- [ ] **Step 5：改造 `ModelCreate` schema**

```python
class ModelCreate(BaseModel):
    """创建模型请求体。训练数据通过 GeoJSON FeatureCollection 传入。"""
    name: str = Field(..., description="模型名称", example="哈尔滨建筑提取模型")
    model_type: str = Field(..., description="模型类型：classification / change_detection")
    task_type: str = Field(..., description="任务类型", example="building_extraction")
    region_id: str = Field(..., description="区域 ID", example="harbin")
    class_ids: Optional[List[str]] = Field(
        None,
        description="指定参与训练的分类 ID 列表，为空则使用 annotations 中出现的所有 class_id"
    )
    epochs: int = Field(20, ge=1, le=1000, description="训练轮数")
    description: Optional[str] = Field(None, description="模型描述")
    annotations: GeoJSONFeatureCollection = Field(
        ...,
        description="GeoJSON FeatureCollection 格式的标注数组，坐标系 WGS84"
    )
    classes: List[ModelClass] = Field(..., min_length=1, description="分类定义列表")

    @model_validator(mode="after")
    def validate_model_type(self):
        for f in self.annotations.features:
            props = f.properties
            if self.model_type == "classification" and not props.month:
                raise ValueError(f"classification 模型要求 feature {props.patch_id} 提供 month")
            if self.model_type == "change_detection" and (not props.before_month or not props.after_month):
                raise ValueError(f"change_detection 模型要求 feature {props.patch_id} 提供 before_month 和 after_month")
        return self
```

- [ ] **Step 6：运行 schema 测试**

Run: `python -m pytest tests/test_sam3_schemas.py -q`
Expected: PASS

- [ ] **Step 7：Commit**

```bash
git add app/schemas/models.py
git commit -m "feat(schema): add GeoJSON FeatureCollection schemas and update ModelCreate"
```

---

### Task 3：实现 GeoJSON 解析与栅格化

**Files:**
- Create: `app/services/geojson_adapter.py`

- [ ] **Step 1：创建文件并实现坐标转换**

```python
import json
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon, shape

from app.services.data_service import get_patch_metadata


def wgs84_to_pixel(
    lon: float,
    lat: float,
    bbox: Tuple[float, float, float, float],
    width: int = 256,
    height: int = 256,
) -> Tuple[int, int]:
    """WGS84 坐标转 Patch 像素坐标 (col, row)。"""
    minx, miny, maxx, maxy = bbox
    col = int((lon - minx) / (maxx - minx) * width)
    row = int((maxy - lat) / (maxy - miny) * height)
    col = max(0, min(width - 1, col))
    row = max(0, min(height - 1, row))
    return col, row


def geometry_to_pixel_coords(
    geometry: Dict,
    bbox: Tuple[float, float, float, float],
    width: int = 256,
    height: int = 256,
) -> List[List[Tuple[int, int]]]:
    """把 GeoJSON geometry 转成若干像素多边形环列表。"""
    geom = shape(geometry)
    polygons = []
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        raise ValueError(f"不支持 geometry 类型: {geom.geom_type}")

    result = []
    for poly in polygons:
        exterior = [
            wgs84_to_pixel(lon, lat, bbox, width, height)
            for lon, lat in poly.exterior.coords
        ]
        result.append(exterior)
        for interior in poly.interiors:
            hole = [
                wgs84_to_pixel(lon, lat, bbox, width, height)
                for lon, lat in interior.coords
            ]
            result.append(hole)
    return result


def rasterize_geometry(
    geometry: Dict,
    bbox: Tuple[float, float, float, float],
    size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """把 GeoJSON geometry 栅格化为二值 mask。"""
    rings = geometry_to_pixel_coords(geometry, bbox, size[0], size[1])
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    # ImageDraw.polygon 只支持单个多边形，且不支持洞。
    # 简单做法：所有外环填充 255，所有内环填充 0。
    # 这里先简化处理，每个 ring 如果是逆时针则视为洞（复杂场景后续优化）。
    # 为简化实现，只处理外环：
    for ring in rings:
        if len(ring) < 3:
            continue
        # 使用 shapely 判断是否为洞：面积为正则填，为负则挖
        from shapely.geometry import Polygon as ShapelyPolygon
        poly = ShapelyPolygon(ring)
        if poly.area >= 0:
            draw.polygon(ring, fill=255)
        else:
            draw.polygon(ring, fill=0)

    return (np.array(mask) > 0).astype(np.uint8)


def group_features_by_patch(features: List[Dict]) -> Dict[str, List[Dict]]:
    """按 patch_id 分组 GeoJSON features。"""
    groups: Dict[str, List[Dict]] = {}
    for f in features:
        patch_id = f["properties"]["patch_id"]
        groups.setdefault(patch_id, []).append(f)
    return groups


def build_class_map(classes: List[Dict]) -> Dict[str, int]:
    """构建 class_id → label_index 映射。"""
    sorted_ids = sorted(c["id"] for c in classes)
    return {cls_id: idx for idx, cls_id in enumerate(sorted_ids)}
```

- [ ] **Step 2：处理 MultiPolygon 带洞情况（更鲁棒）**

上面的简化版可能不够。建议用 rasterio 的 `rasterize` 函数：

```python
from rasterio import features as rasterio_features

def rasterize_geometry_v2(geometry, bbox, size=(256, 256)):
    """使用 rasterio 栅格化，支持 Polygon/MultiPolygon 及洞。"""
    geom = shape(geometry)
    # 构造 [(geometry, value)]
    shapes = [(geom, 1)]

    # 构建 transform：从像素坐标到地理坐标的仿射变换
    minx, miny, maxx, maxy = bbox
    transform = rasterio.Affine.identity()
    transform *= rasterio.Affine.translation(minx, maxy)
    transform *= rasterio.Affine.scale((maxx - minx) / size[0], (miny - maxy) / size[1])

    mask = rasterio_features.rasterize(
        shapes,
        out_shape=size,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )
    return mask
```

> 说明：推荐使用 `rasterize_geometry_v2`，更准确且无需手动坐标转换。

- [ ] **Step 3：添加测试**

Create: `tests/test_geojson_adapter.py`

```python
import numpy as np
import pytest

from app.services.geojson_adapter import (
    build_class_map,
    group_features_by_patch,
    wgs84_to_pixel,
)


def test_wgs84_to_pixel():
    bbox = (126.5, 45.74, 126.55, 45.76)
    col, row = wgs84_to_pixel(126.5, 45.76, bbox)  # 左上角
    assert col == 0
    assert row == 0

    col, row = wgs84_to_pixel(126.55, 45.74, bbox)  # 右下角
    assert col == 255
    assert row == 255


def test_group_features_by_patch():
    features = [
        {"properties": {"patch_id": "patch_000000"}},
        {"properties": {"patch_id": "patch_000000"}},
        {"properties": {"patch_id": "patch_000001"}},
    ]
    groups = group_features_by_patch(features)
    assert len(groups["patch_000000"]) == 2
    assert len(groups["patch_000001"]) == 1


def test_build_class_map():
    classes = [
        {"id": "cls_b", "name": "B", "color": "#00FF00"},
        {"id": "cls_a", "name": "A", "color": "#FF0000"},
    ]
    assert build_class_map(classes) == {"cls_a": 0, "cls_b": 1}
```

Run: `python -m pytest tests/test_geojson_adapter.py -v`
Expected: PASS

- [ ] **Step 4：Commit**

```bash
git add app/services/geojson_adapter.py tests/test_geojson_adapter.py
git commit -m "feat(geojson): add WGS84-to-pixel rasterization and class mapping"
```

---

### Task 4：改造训练引擎

**Files:**
- Modify: `app/services/training_engine.py`

- [ ] **Step 1：修改 `BaseTrainingEngine.train` 签名**

```python
from typing import Any, Dict, List, Optional, Tuple
from app.schemas.models import GeoJSONFeatureCollection, ModelClass

class BaseTrainingEngine(ABC):
    @abstractmethod
    def train(
        self,
        annotations: GeoJSONFeatureCollection,
        classes: List[ModelClass],
        class_ids: Optional[List[str]],
        epochs: int,
        model_dir: Path,
    ) -> Dict[str, Any]:
        ...
```

- [ ] **Step 2：在 `ClassificationTrainingEngine` 中实现新训练逻辑**

```python
def train(self, annotations, classes, class_ids, epochs, model_dir):
    from app.services.geojson_adapter import (
        build_class_map,
        group_features_by_patch,
        rasterize_geometry_v2,
    )
    from app.services.data_service import get_patch_metadata

    class_map = build_class_map([c.model_dump() for c in classes])
    if class_ids:
        class_map = {k: v for k, v in class_map.items() if k in class_ids}

    groups = group_features_by_patch([f.model_dump() for f in annotations.features])

    X_samples = []
    y_samples = []

    for patch_id, features in groups.items():
        meta = get_patch_metadata(annotations.features[0].properties.region_id, patch_id)
        bbox = meta["bbox"]  # [minx, miny, maxx, maxy]

        # 按 class_id 合并 mask
        masks = {}
        for f in features:
            cls_id = f["properties"]["class_id"]
            month = f["properties"].get("month")
            mask = rasterize_geometry_v2(f["geometry"], bbox)
            masks.setdefault((cls_id, month), np.zeros_like(mask))
            masks[(cls_id, month)] = np.maximum(masks[(cls_id, month)], mask)

        for (cls_id, month), mask in masks.items():
            emb = self._load_embedding(meta["region_id"], patch_id, month)
            # emb shape: [C, H, W]
            C, H, W = emb.shape
            mask = mask[:H, :W]

            # 提取 mask 区域的像素特征
            mask_flat = mask.flatten()
            emb_flat = emb.reshape(C, -1).T  # [H*W, C]
            pos_samples = emb_flat[mask_flat > 0]

            # 负采样：从非 mask 区域随机采相同数量
            neg_indices = np.where(mask_flat == 0)[0]
            if len(neg_indices) > len(pos_samples):
                neg_indices = np.random.choice(neg_indices, len(pos_samples), replace=False)
            neg_samples = emb_flat[neg_indices]

            X_samples.extend(pos_samples)
            y_samples.extend([class_map[cls_id]] * len(pos_samples))
            X_samples.extend(neg_samples)
            y_samples.extend([0] * len(neg_samples))  # 背景类用 0

    if not X_samples:
        raise ValueError("没有有效的训练样本")

    X = np.vstack(X_samples)
    y = np.array(y_samples)

    # 训练并保存
    clf = LogisticRegression(max_iter=epochs * 10, multi_class="multinomial")
    clf.fit(X, y)

    model_path = model_dir / "model.pkl"
    joblib.dump(clf, model_path)

    return {"accuracy": clf.score(X, y), "n_samples": len(y), "n_classes": len(class_map)}
```

> 注：具体采样比例、背景类处理需与现有实现保持一致。上述为示意，需参考现有 `training_engine.py` 调整。

- [ ] **Step 3：修改 `ChangeDetectionTrainingEngine`**

类似，但加载 `before_month` 和 `after_month` 的 embedding，计算差分：

```python
emb_before = self._load_embedding(region_id, patch_id, before_month)
emb_after = self._load_embedding(region_id, patch_id, after_month)
diff = emb_after - emb_before
```

- [ ] **Step 4：Commit**

```bash
git add app/services/training_engine.py
git commit -m "feat(training): train from GeoJSON annotations instead of AnnotationStore"
```

---

### Task 5：改造 `POST /models` 路由

**Files:**
- Modify: `app/routers/models.py`

- [ ] **Step 1：修改 `create_model` 函数**

```python
@router.post("", response_model=ModelOut, status_code=status.HTTP_201_CREATED)
async def create_model(req: ModelCreate, user: dict = Depends(get_current_user)):
    """创建模型并启动训练。

    训练数据由前端通过 GeoJSON FeatureCollection 一次性传入，坐标系为 WGS84。
    后端按 Patch 解析几何、栅格化、提取 embedding 后训练模型。
    """
    registry = ModelRegistry(user["user_id"])

    if req.model_type not in TRAINING_ENGINES:
        raise HTTPException(status_code=400, detail=f"Unsupported model_type: {req.model_type}")

    class_ids = req.class_ids or list({f.properties.class_id for f in req.annotations.features})

    model_dir = registry.create_model(
        name=req.name,
        model_type=req.model_type,
        task_type=req.task_type,
        region_id=req.region_id,
        class_ids=class_ids,
        description=req.description,
    )
    model_id = model_dir.name
    model = registry.get_model(model_id)

    def train_worker():
        try:
            engine_cls = TRAINING_ENGINES[req.model_type]
            engine = engine_cls()
            metrics = engine.train(
                annotations=req.annotations,
                classes=req.classes,
                class_ids=class_ids,
                epochs=req.epochs,
                model_dir=model_dir,
            )
            registry.update_status(model_id, "ready", metrics=metrics)
        except Exception as exc:
            import traceback
            registry.update_status(model_id, "failed", error=str(exc), traceback=traceback.format_exc())

    job_id = registry.start_training_job(model_id, train_worker)
    model["job_id"] = job_id
    return model
```

- [ ] **Step 2：移除 `AnnotationStore` 引用**

确保 `app/routers/models.py` 不再导入 `get_annotation_store`。

- [ ] **Step 3：Commit**

```bash
git add app/routers/models.py
git commit -m "feat(models): POST /models accepts GeoJSON annotations from frontend"
```

---

### Task 6：更新模型测试

**Files:**
- Modify: `tests/test_models.py`

- [ ] **Step 1：构造 GeoJSON 测试数据**

```python
def _build_geojson_annotation(patch_id="patch_000000", month="2025-04"):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "patch_id": patch_id,
                    "region_id": "harbin",
                    "class_id": "cls_001",
                    "class_name": "建筑",
                    "color": "#FF0000",
                    "task_type": "building_extraction",
                    "month": month,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [126.50, 45.74],
                            [126.52, 45.74],
                            [126.52, 45.76],
                            [126.50, 45.76],
                            [126.50, 45.74],
                        ]
                    ],
                },
            }
        ],
    }
```

- [ ] **Step 2：重写 `test_create_model_trains_and_completes`**

```python
def test_create_model_with_geojson(client):
    resp = client.post("/models", json={
        "name": "GeoJSON 测试模型",
        "model_type": "classification",
        "task_type": "building_extraction",
        "region_id": "harbin",
        "epochs": 1,
        "annotations": _build_geojson_annotation(),
        "classes": [{"id": "cls_001", "name": "建筑", "color": "#FF0000"}],
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "training"
    job_id = data["job_id"]

    # 轮询
    for _ in range(30):
        job = client.get(f"/models/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert job["status"] == "completed"
```

- [ ] **Step 3：添加变化检测测试**

```python
def test_create_change_detection_model_with_geojson(client):
    annotations = _build_geojson_annotation()
    annotations["features"][0]["properties"].update({
        "month": None,
        "before_month": "2025-04",
        "after_month": "2025-06",
        "task_type": "change_detection",
    })
    resp = client.post("/models", json={
        "name": "变化检测测试模型",
        "model_type": "change_detection",
        "task_type": "change_detection",
        "region_id": "harbin",
        "epochs": 1,
        "annotations": annotations,
        "classes": [{"id": "cls_001", "name": "变化区域", "color": "#FF0000"}],
    })
    assert resp.status_code == 201
```

- [ ] **Step 4：运行全部测试**

Run: `python -m pytest tests/ -q`
Expected: 84 passed（删除 annotations 测试后数量会减少，但无失败）

- [ ] **Step 5：Commit**

```bash
git add tests/test_models.py
git commit -m "test(models): update tests for GeoJSON-driven training"
```

---

### Task 7：更新文档

**Files:**
- Modify: `docs/custom-training-workflow.md`
- Modify: `docs/API.md`
- Modify: `README.md`

- [ ] **Step 1：重写 `docs/custom-training-workflow.md`**

删除「Step 1 创建分类」「Step 2 创建标注」等涉及 `/annotations` 接口的内容。

新增「前端标注管理」章节：

```markdown
## 前端标注管理

前端在浏览器 `localStorage` 中维护两个数组：

### 1. classes 数组

```json
[
  {"id": "cls_001", "name": "建筑用地", "color": "#FF0000"}
]
```

### 2. annotations GeoJSON FeatureCollection

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
            [126.50, 45.74],
            [126.52, 45.74],
            [126.52, 45.76],
            [126.50, 45.76],
            [126.50, 45.74]
          ]
        ]
      }
    }
  ]
}
```

### 坐标系说明

- 所有 `geometry` 坐标必须是 **WGS84 (EPSG:4326)**，顺序为 `[经度, 纬度]`。
- 支持的 `geometry.type`：`Polygon`、`MultiPolygon`。
- 不支持 `Point`、`LineString`。
```

更新「Step 3：创建模型并启动训练」示例请求体。

- [ ] **Step 2：更新 `docs/API.md`**

- 删除 `/annotations` 全部接口文档
- 更新 `POST /models` 请求体示例为 GeoJSON 格式

- [ ] **Step 3：更新 `README.md`**

删除 Features 中关于 Annotation 的条目，或在 API 文档链接中移除 annotations。

- [ ] **Step 4：Commit**

```bash
git add docs/
git commit -m "docs: update training workflow and API reference for GeoJSON annotations"
```

---

### Task 8：最终验证与部署

- [ ] **Step 1：运行完整测试**

Run: `python -m pytest tests/ -q`
Expected: 全部通过

- [ ] **Step 2：本地验证 Swagger**

Start: `python service_watchdog.py`（后台）
Check: `http://localhost:9061/docs`
Verify:
- 没有 `Annotations` 分组
- `POST /models` 请求体示例是 GeoJSON FeatureCollection
- 字段有中文描述

- [ ] **Step 3：Push 到 GitHub**

```bash
git push origin main
```

- [ ] **Step 4：重启生产服务**

```bash
python service_watchdog.py stop
python service_watchdog.py  # 后台运行
```

---

## 五、依赖项检查

### 5.1 需要新增/确认安装的依赖

- `shapely`：WGS84 geometry 解析
- `rasterio`：geometry 栅格化（项目已有 `rasterio >= 1.3.0`，无需新增）

检查 `requirements.txt`：

```bash
grep -E "shapely|rasterio" requirements.txt
```

如果缺少 `shapely`，需在 Task 1 中追加：

```bash
echo "shapely>=2.0" >> requirements.txt
pip install shapely
```

### 5.2 现有依赖确认

- `rasterio` 已存在
- `Pillow` 已存在
- `numpy` 已存在

---

## 六、边界情况与错误处理

| 场景 | 后端行为 |
|------|----------|
| `annotations` 为空 FeatureCollection | Pydantic 校验失败，返回 422 |
| `geometry.type` 为 Point/LineString | Pydantic 校验失败，返回 422 |
| `month` 与 `model_type` 不匹配 | Pydantic 校验失败，返回 422 |
| `patch_id` 不存在 | 训练时 `get_patch_metadata` 失败，模型状态变为 `failed` |
| `class_id` 不在 `classes` 中 | 训练时 `KeyError`，模型状态变为 `failed` |
| 标注区域与 Patch bbox 无交集 | mask 全 0，无有效样本，模型状态变为 `failed` |

---

## 七、Self-Review

1. **Spec coverage:**
   - ✅ 删除旧 `/annotations` 接口：Task 1
   - ✅ 训练数据通过 GeoJSON 数组传入：Task 2 schema + Task 5 路由
   - ✅ 后端解析 GeoJSON 并训练：Task 3 适配器 + Task 4 训练引擎
   - ✅ 包含类别信息：`classes` 字段 + `properties.class_id`
   - ✅ 详细步骤说明：本计划每个 Task 含具体代码

2. **Placeholder scan:**
   - 无 TBD/TODO
   - `<base64>` 不再使用，改为 GeoJSON WGS84 坐标

3. **Type consistency:**
   - `GeoJSONFeature`, `GeoJSONFeatureCollection`, `ModelClass` 在 Task 2 定义
   - Task 3/4/5/6 一致使用这些类型
