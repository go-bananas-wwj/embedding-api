# 前端本地标注数组驱动训练 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 改造 `/models` 训练接口，使其不再依赖后端 `annotations` 存储，而是直接接收前端从浏览器 localStorage 中回传的标注数组，并据此完成模型训练。

**Architecture:** 前端负责标注的增删改查与持久化，后端只负责在创建模型时一次性接收完整标注数组、校验格式、提取训练样本并启动训练。保持现有训练引擎不变，仅在其前增加一个「数组 → 训练样本」适配层。

**Tech Stack:** FastAPI, Pydantic v2, scikit-learn, NumPy, Pillow

---

## 背景与需求

当前流程：
1. 前端调用 `POST /annotations/classes` 创建分类
2. 前端调用 `POST /annotations` 保存标注
3. 前端调用 `POST /models` 创建模型，后端去 `AnnotationStore` 读取标注

新流程：
1. 前端在浏览器 localStorage 中自行管理分类和标注（增删改查）
2. 前端调用 `POST /models` 时，把完整的标注数组一并提交
3. 后端解析该数组，提取训练样本并启动训练

因此需要：
- 定义前端回传标注数组的精确格式
- 修改 `POST /models` 请求体，增加 `annotations` 字段
- 修改训练引擎，支持从数组解析样本
- 更新相关文档与测试

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `app/schemas/models.py` | 新增 `FrontendAnnotation`, `FrontendClass`, `ModelCreate` 请求体改造 |
| `app/services/annotation_service.py` | 新增从数组生成训练样本的辅助方法（或新建 `annotation_adapter.py`） |
| `app/services/training_engine.py` | 修改训练入口，接收标注数组而非 `AnnotationStore` |
| `app/services/inference_engine.py` | 无需改动 |
| `app/routers/models.py` | 修改 `POST /models`，从请求体读取 `annotations` 并传给训练引擎 |
| `tests/test_models.py` | 更新训练测试，使用数组形式构造训练数据 |
| `docs/custom-training-workflow.md` | 更新工作流文档，说明前端 localStorage + 数组回传模式 |
| `docs/API.md` | 更新 `POST /models` 示例请求体 |

---

## 关键设计：前端标注数组格式

前端在调用 `POST /models` 时，需要提交如下结构的数组：

```json
{
  "name": "哈尔滨建筑提取模型",
  "model_type": "classification",
  "task_type": "building_extraction",
  "region_id": "harbin",
  "epochs": 20,
  "annotations": [
    {
      "id": "ann_001",
      "region_id": "harbin",
      "patch_id": "patch_000000",
      "month": "2025-04",
      "class_id": "cls_001",
      "task_type": "building_extraction",
      "geometry": {
        "type": "mask",
        "mask_b64": "iVBORw0KGgoAAAANSUhEUgAA..."
      },
      "before_month": null,
      "after_month": null
    }
  ],
  "classes": [
    {
      "id": "cls_001",
      "name": "建筑用地",
      "color": "#FF0000"
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `annotations` | array | 是 | 标注对象数组 |
| `annotations[].id` | string | 否 | 前端生成的唯一标识，仅用于前端管理 |
| `annotations[].region_id` | string | 是 | 区域 ID，如 `harbin` |
| `annotations[].patch_id` | string | 是 | Patch ID，如 `patch_000000` |
| `annotations[].month` | string | 条件 | 单期任务必填，如 `2025-04` |
| `annotations[].before_month` | string | 条件 | 变化检测任务必填 |
| `annotations[].after_month` | string | 条件 | 变化检测任务必填 |
| `annotations[].class_id` | string | 是 | 所属分类 ID |
| `annotations[].task_type` | string | 是 | 任务类型 |
| `annotations[].geometry` | object | 是 | 几何图形，支持 `mask` / `polygon` / `polyline` |
| `annotations[].geometry.type` | string | 是 | `mask`、`polygon`、`polyline` |
| `annotations[].geometry.mask_b64` | string | 条件 | `type=mask` 时必填，base64 PNG |
| `annotations[].geometry.points` | array | 条件 | `type=polygon` 或 `polyline` 时必填 |
| `classes` | array | 是 | 分类数组，用于训练时映射类别 |
| `classes[].id` | string | 是 | 分类 ID |
| `classes[].name` | string | 是 | 分类名称 |
| `classes[].color` | string | 是 | 颜色，如 `#FF0000` |

### 校验规则

- `model_type=classification` 时：每个标注必须包含 `month`
- `model_type=change_detection` 时：每个标注必须包含 `before_month` 和 `after_month`
- `annotations` 数组不能为空
- `class_ids` 若为空，默认使用 `annotations` 中出现的所有 `class_id`

---

## Task 1: 定义前端标注相关 Pydantic Schema

**Files:**
- Modify: `app/schemas/models.py`

- [ ] **Step 1: 新增 `GeometryType` 枚举**

```python
from enum import Enum

class GeometryType(str, Enum):
    MASK = "mask"
    POLYGON = "polygon"
    POLYLINE = "polyline"
```

- [ ] **Step 2: 新增 `FrontendGeometry` schema**

```python
class FrontendGeometry(BaseModel):
    type: GeometryType = Field(..., description="几何类型：mask/polygon/polyline")
    mask_b64: Optional[str] = Field(None, description="type=mask 时的 base64 PNG 掩膜")
    points: Optional[List[List[int]]] = Field(None, description="type=polygon/polyline 时的像素坐标点")

    @model_validator(mode="after")
    def validate_geometry(self):
        if self.type == GeometryType.MASK and not self.mask_b64:
            raise ValueError("mask 类型必须提供 mask_b64")
        if self.type in (GeometryType.POLYGON, GeometryType.POLYLINE) and not self.points:
            raise ValueError(f"{self.type} 类型必须提供 points")
        return self
```

- [ ] **Step 3: 新增 `FrontendClass` schema**

```python
class FrontendClass(BaseModel):
    id: str = Field(..., description="分类 ID", example="cls_001")
    name: str = Field(..., description="分类名称", example="建筑用地")
    color: str = Field(..., description="分类颜色", example="#FF0000")
```

- [ ] **Step 4: 新增 `FrontendAnnotation` schema**

```python
class FrontendAnnotation(BaseModel):
    id: Optional[str] = Field(None, description="前端标注唯一 ID")
    region_id: str = Field(..., description="区域 ID", example="harbin")
    patch_id: str = Field(..., description="Patch ID", example="patch_000000")
    month: Optional[str] = Field(None, description="单期任务月份", example="2025-04")
    before_month: Optional[str] = Field(None, description="变化检测前期月份", example="2025-04")
    after_month: Optional[str] = Field(None, description="变化检测后期月份", example="2025-06")
    class_id: str = Field(..., description="所属分类 ID", example="cls_001")
    task_type: str = Field(..., description="任务类型", example="building_extraction")
    geometry: FrontendGeometry = Field(..., description="几何图形")
```

- [ ] **Step 5: 修改 `ModelCreate` schema**

在 `ModelCreate` 中新增两个可选字段：

```python
class ModelCreate(BaseModel):
    name: str = Field(..., description="模型名称")
    model_type: str = Field(..., description="模型类型：classification 或 change_detection")
    task_type: str = Field(..., description="任务类型")
    region_id: str = Field(..., description="训练区域 ID")
    class_ids: Optional[List[str]] = Field(None, description="指定参与训练的分类 ID 列表，为空则使用全部")
    epochs: int = Field(20, ge=1, le=1000, description="训练轮数")
    description: Optional[str] = Field(None, description="模型描述")
    annotations: List[FrontendAnnotation] = Field(default_factory=list, description="前端回传的标注数组")
    classes: List[FrontendClass] = Field(default_factory=list, description="前端回传的分类数组")

    @model_validator(mode="after")
    def validate_training_data(self):
        if not self.annotations:
            raise ValueError("annotations 数组不能为空")
        if self.model_type == "classification":
            for ann in self.annotations:
                if not ann.month:
                    raise ValueError("classification 模型要求每个标注提供 month")
        elif self.model_type == "change_detection":
            for ann in self.annotations:
                if not ann.before_month or not ann.after_month:
                    raise ValueError("change_detection 模型要求每个标注提供 before_month 和 after_month")
        return self
```

- [ ] **Step 6: 运行 schema 相关测试**

Run: `python -m pytest tests/test_sam3_schemas.py tests/test_models.py -q`
Expected: 可能有失败，因为 `ModelCreate` 新增必填字段影响了旧测试，后续 Task 修复

- [ ] **Step 7: Commit**

```bash
git add app/schemas/models.py
git commit -m "feat(schema): add FrontendAnnotation/FrontendClass and update ModelCreate"
```

---

## Task 2: 新增标注数组 → 训练样本适配器

**Files:**
- Create: `app/services/annotation_adapter.py`
- Modify: `app/services/annotation_service.py`（可选，复用 base64/polygon 解析逻辑）

- [ ] **Step 1: 创建 `annotation_adapter.py`**

```python
import base64
import binascii
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from app.schemas.models import FrontendAnnotation, FrontendClass, FrontendGeometry


def geometry_to_mask(geometry: FrontendGeometry, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """把前端 geometry 转成 256x256 二值 mask。"""
    if geometry.type == "mask":
        return _base64_to_mask(geometry.mask_b64, size)
    elif geometry.type == "polygon":
        return _rasterize_polygon(geometry.points, size)
    elif geometry.type == "polyline":
        return _rasterize_polyline(geometry.points, size)
    raise ValueError(f"未知 geometry 类型: {geometry.type}")


def _base64_to_mask(data: str, size: Tuple[int, int]) -> np.ndarray:
    try:
        img = Image.open(BytesIO(base64.b64decode(data))).convert("L")
    except (binascii.Error, Exception) as exc:
        raise ValueError(f"无效 mask base64: {exc}") from exc
    img = img.resize(size, Image.NEAREST)
    return (np.array(img) > 0).astype(np.uint8)


def _rasterize_polygon(points: List[List[int]], size: Tuple[int, int]) -> np.ndarray:
    from PIL import ImageDraw
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([(p[0], p[1]) for p in points], fill=255)
    return (np.array(mask) > 0).astype(np.uint8)


def _rasterize_polyline(points: List[List[int]], size: Tuple[int, int]) -> np.ndarray:
    from PIL import ImageDraw
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.line([(p[0], p[1]) for p in points], fill=255, width=2)
    return (np.array(mask) > 0).astype(np.uint8)


def build_class_map(classes: List[FrontendClass]) -> Dict[str, int]:
    """把分类列表映射成 {class_id: label_index}。"""
    return {cls.id: idx for idx, cls in enumerate(sorted(classes, key=lambda c: c.id))}
```

- [ ] **Step 2: 添加测试**

Create: `tests/test_annotation_adapter.py`

```python
import numpy as np
import pytest

from app.schemas.models import FrontendClass, FrontendGeometry
from app.services.annotation_adapter import build_class_map, geometry_to_mask


def test_build_class_map():
    classes = [
        FrontendClass(id="cls_b", name="B", color="#00FF00"),
        FrontendClass(id="cls_a", name="A", color="#FF0000"),
    ]
    assert build_class_map(classes) == {"cls_a": 0, "cls_b": 1}


def test_polygon_to_mask():
    geom = FrontendGeometry(
        type="polygon",
        points=[[0, 0], [255, 0], [255, 255], [0, 255]],
    )
    mask = geometry_to_mask(geom)
    assert mask.shape == (256, 256)
    assert mask.sum() > 0
```

Run: `python -m pytest tests/test_annotation_adapter.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add app/services/annotation_adapter.py tests/test_annotation_adapter.py
git commit -m "feat(adapter): add geometry-to-mask and class mapping utilities"
```

---

## Task 3: 改造训练引擎接收标注数组

**Files:**
- Modify: `app/services/training_engine.py`
- Modify: `app/services/model_registry.py`（训练入口参数）

- [ ] **Step 1: 修改 `BaseTrainingEngine.train` 签名**

```python
from typing import List
from app.schemas.models import FrontendAnnotation, FrontendClass

class BaseTrainingEngine(ABC):
    @abstractmethod
    def train(
        self,
        annotations: List[FrontendAnnotation],
        classes: List[FrontendClass],
        class_ids: Optional[List[str]],
        epochs: int,
        model_dir: Path,
    ) -> Dict[str, Any]:
        ...
```

- [ ] **Step 2: 修改 `ClassificationTrainingEngine.train`**

伪代码核心逻辑：

```python
def train(self, annotations, classes, class_ids, epochs, model_dir):
    class_map = build_class_map(classes)
    if class_ids:
        class_map = {k: v for k, v in class_map.items() if k in class_ids}

    X, y = [], []
    for ann in annotations:
        emb = load_embedding(ann.region_id, ann.patch_id, ann.month)  # [C, H, W]
        mask = geometry_to_mask(ann.geometry)  # [H, W]
        for c in range(emb.shape[0]):
            X.append(emb[c][mask > 0])
        y.append(class_map[ann.class_id])

    # ... 训练并保存 model.pkl
```

> 注：具体实现需参考现有 `training_engine.py` 中的 embedding 加载、采样、模型保存逻辑，保持行为一致。

- [ ] **Step 3: 修改 `ChangeDetectionTrainingEngine.train`**

类似地，使用 `before_month` 和 `after_month` 计算差分 embedding：

```python
emb_before = load_embedding(ann.region_id, ann.patch_id, ann.before_month)
emb_after = load_embedding(ann.region_id, ann.patch_id, ann.after_month)
diff = emb_after - emb_before
mask = geometry_to_mask(ann.geometry)
```

- [ ] **Step 4: 运行训练引擎测试**

Run: `python -m pytest tests/test_models.py -v`
Expected: 需要同步更新测试数据，见 Task 4

- [ ] **Step 5: Commit**

```bash
git add app/services/training_engine.py
git commit -m "refactor(training): accept frontend annotation array instead of AnnotationStore"
```

---

## Task 4: 修改 `POST /models` 路由

**Files:**
- Modify: `app/routers/models.py`

- [ ] **Step 1: 修改 `create_model` 函数**

```python
@router.post("", response_model=ModelOut, status_code=status.HTTP_201_CREATED)
async def create_model(req: ModelCreate, user: dict = Depends(get_current_user)):
    """创建模型并启动训练。训练数据来自前端回传的 annotations/classes 数组。"""
    registry = ModelRegistry(user["user_id"])

    if req.model_type not in TRAINING_ENGINES:
        raise HTTPException(status_code=400, detail=f"Unsupported model_type: {req.model_type}")

    model_dir = registry.create_model(
        name=req.name,
        model_type=req.model_type,
        task_type=req.task_type,
        region_id=req.region_id,
        class_ids=req.class_ids or list({ann.class_id for ann in req.annotations}),
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
                class_ids=req.class_ids,
                epochs=req.epochs,
                model_dir=model_dir,
            )
            registry.update_status(model_id, "ready", metrics=metrics)
        except Exception as exc:
            registry.update_status(model_id, "failed", error=str(exc))

    job_id = registry.start_training_job(model_id, train_worker)
    model["job_id"] = job_id
    return model
```

- [ ] **Step 2: 移除对 `AnnotationStore` 的依赖**

确保 `models.py` 不再 `from app.services.annotation_service import get_annotation_store`。

- [ ] **Step 3: 运行模型路由测试**

Run: `python -m pytest tests/test_models.py -v`
Expected: 当前测试失败，因为测试仍使用旧的 `AnnotationStore` 方式

- [ ] **Step 4: Commit**

```bash
git add app/routers/models.py
git commit -m "feat(models): accept frontend annotations array in POST /models"
```

---

## Task 5: 更新测试

**Files:**
- Modify: `tests/test_models.py`

- [ ] **Step 1: 重构训练测试用例**

旧测试可能先创建标注再创建模型。新测试应直接构造 `annotations` 数组：

```python
def test_create_model_with_frontend_annotations(client):
    annotations = [
        {
            "id": "ann_001",
            "region_id": "harbin",
            "patch_id": "patch_000000",
            "month": "2025-04",
            "class_id": "cls_001",
            "task_type": "building_extraction",
            "geometry": {
                "type": "mask",
                "mask_b64": "<base64 of a small 256x256 mask>",
            },
        }
    ]
    classes = [
        {"id": "cls_001", "name": "建筑", "color": "#FF0000"},
    ]

    resp = client.post("/models", json={
        "name": "测试模型",
        "model_type": "classification",
        "task_type": "building_extraction",
        "region_id": "harbin",
        "annotations": annotations,
        "classes": classes,
        "epochs": 1,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "training"
    assert "job_id" in data
```

- [ ] **Step 2: 添加缺失字段校验测试**

```python
def test_create_model_without_annotations_returns_422(client):
    resp = client.post("/models", json={
        "name": "测试模型",
        "model_type": "classification",
        "task_type": "building_extraction",
        "region_id": "harbin",
        "annotations": [],
        "classes": [],
    })
    assert resp.status_code == 422
```

- [ ] **Step 3: 添加 change_detection 测试**

```python
def test_create_change_detection_model_with_annotations(client):
    annotations = [
        {
            "region_id": "harbin",
            "patch_id": "patch_000000",
            "before_month": "2025-04",
            "after_month": "2025-06",
            "class_id": "cls_001",
            "task_type": "change_detection",
            "geometry": {"type": "mask", "mask_b64": "<base64>"},
        }
    ]
    classes = [{"id": "cls_001", "name": "变化区域", "color": "#FF0000"}]

    resp = client.post("/models", json={
        "name": "变化检测模型",
        "model_type": "change_detection",
        "task_type": "change_detection",
        "region_id": "harbin",
        "annotations": annotations,
        "classes": classes,
        "epochs": 1,
    })
    assert resp.status_code == 201
```

- [ ] **Step 4: 运行全部测试**

Run: `python -m pytest tests/ -q`
Expected: 84 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_models.py
git commit -m "test(models): update tests for frontend annotation array workflow"
```

---

## Task 6: 更新自定义训练工作流文档

**Files:**
- Modify: `docs/custom-training-workflow.md`

- [ ] **Step 1: 修改流程说明**

把「Step 1 创建分类」和「Step 2 创建标注」改为「前端本地管理分类和标注」。

新增说明：

```markdown
### 前端本地存储建议

前端可以使用 `localStorage` 或 `IndexedDB` 存储标注数据，结构如下：

```json
{
  "classes": [
    {"id": "cls_001", "name": "建筑用地", "color": "#FF0000"}
  ],
  "annotations": [
    {
      "id": "ann_001",
      "region_id": "harbin",
      "patch_id": "patch_000000",
      "month": "2025-04",
      "class_id": "cls_001",
      "task_type": "building_extraction",
      "geometry": {"type": "mask", "mask_b64": "..."}
    }
  ]
}
```

在调用 `POST /models` 时，把整个对象中的 `classes` 和 `annotations` 一起传给后端。
```

- [ ] **Step 2: 修改「Step 3：创建模型并启动训练」示例请求体**

把示例改为包含 `annotations` 和 `classes` 字段。

- [ ] **Step 3: 删除或标注旧的后端标注接口**

说明 `POST /annotations`、`POST /annotations/classes` 等接口现在不是必须调用的，仅作为可选保留。

- [ ] **Step 4: Commit**

```bash
git add docs/custom-training-workflow.md
git commit -m "docs: update custom training workflow for frontend-local annotations"
```

---

## Task 7: 更新 API 参考文档与 Swagger 描述

**Files:**
- Modify: `docs/API.md`
- Modify: `app/routers/models.py`（`POST /models` 的 docstring/description）

- [ ] **Step 1: 更新 `docs/API.md` 中 `POST /models` 的示例请求体**

包含 `annotations` 和 `classes` 数组。

- [ ] **Step 2: 更新 Swagger 中 `POST /models` 的中文描述**

```python
@router.post("", response_model=ModelOut, status_code=status.HTTP_201_CREATED)
async def create_model(req: ModelCreate, user: dict = Depends(get_current_user)):
    """创建自定义模型并启动异步训练。

    训练所需标注数据由前端通过 `annotations` 和 `classes` 字段一次性传入，
    后端不再依赖 `/annotations` 接口存储的标注。前端可先在浏览器 localStorage
    中管理标注，训练时回传完整数组。
    """
```

- [ ] **Step 3: Commit**

```bash
git add docs/API.md app/routers/models.py
git commit -m "docs: update POST /models examples and swagger description"
```

---

## Task 8: 最终验证与推送

- [ ] **Step 1: 运行完整测试**

Run: `python -m pytest tests/ -q`
Expected: 84+ passed

- [ ] **Step 2: 本地验证 Swagger**

Start: `uvicorn app.main:app --host 0.0.0.0 --port 9061 --env-file .env`
Check: `http://localhost:9061/docs`
Verify: `POST /models` 请求体示例包含 `annotations` 和 `classes` 字段，字段有中文描述

- [ ] **Step 3: 重启生产服务**

```bash
python service_watchdog.py stop
python service_watchdog.py  # 在后台运行
```

- [ ] **Step 4: Push 到 GitHub**

```bash
git push origin main
```

---

## 兼容性与边界

### 向后兼容

- 如果前端继续调用 `POST /annotations` 和 `POST /annotations/classes`，这些接口仍然可用。
- 但 `POST /models` 不再主动读取 `AnnotationStore`；训练数据必须通过 `annotations` 字段传入。

### 边界情况

- `annotations` 为空数组：返回 422
- `classes` 为空数组：返回 422
- `model_type=classification` 但标注缺少 `month`：返回 422
- `model_type=change_detection` 但标注缺少 `before_month`/`after_month`：返回 422
- 某个 `class_id` 在 `classes` 中找不到：训练时抛异常，模型状态变为 `failed`

---

## Self-Review

1. **Spec coverage:**
   - ✅ 前端 localStorage 管理标注：Task 6 文档说明
   - ✅ 训练时回传数组：Task 4 路由改造
   - ✅ 后端解析数组：Task 2 适配器 + Task 3 训练引擎
   - ✅ 定义数组格式：计划开头「前端标注数组格式」+ Task 1 schema
   - ✅ 包括类别信息：`classes` 字段

2. **Placeholder scan:**
   - 无 TBD/TODO
   - `<base64>` 在示例中作为占位符，但这是前端需要生成的动态数据，已说明来源

3. **Type consistency:**
   - `FrontendAnnotation`, `FrontendClass`, `FrontendGeometry` 在 Task 1 定义，后续 Task 2/3/4/5 一致使用
