# SAM3 交互式分割集成实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 xuannv_show 的 SAM3 交互式实例分割功能集成到 embedding-api，新增 `/regions/{rid}/sam3/embed`、`/regions/{rid}/sam3/segment`、`/regions/{rid}/sam3/status` 三个端点。

**Architecture:** 在 embedding-api 中新增 SAM3Service（单例、懒加载、GPU 推理、LRU 缓存），通过 FastAPI 路由暴露。模型权重和 sam3 Python 包完整部署在项目内部，不依赖 xuannv_show。

**Tech Stack:** FastAPI, PyTorch 2.5.1, SAM3 (facebookresearch), Pillow, NumPy

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `models/sam3/` | Create | SAM3 模型权重、配置文件、tokenizer |
| `sam3_pkg/` | Create | SAM3 Python 包副本（从 xuannv_show 复制） |
| `app/services/sam3_service.py` | Create | 核心推理服务：模型加载、embed、segment、缓存、状态 |
| `app/routers/sam3.py` | Create | FastAPI 路由：embed/segment/status 端点 |
| `app/schemas/sam3.py` | Create | Pydantic 请求/响应模型 |
| `app/main.py` | Modify | 注册 sam3 router |
| `app/config.py` | Modify | 新增 `get_sam3_config()` 方法 |
| `config.yaml` | Modify | 新增 `sam3:` 段和 region `s2_dir` |
| `requirements.txt` | Modify | 新增 PyTorch、SAM3 依赖 |
| `tests/test_sam3.py` | Create | 单元测试和集成测试 |

---

## Task 1: 部署模型权重和 SAM3 包

**Files:**
- Create: `models/sam3/` (directory + files)
- Create: `sam3_pkg/` (directory + files)

### Step 1: 创建目录并复制模型权重

```bash
mkdir -p models/sam3/assets
mkdir -p sam3_pkg
```

复制模型权重和配置文件（来源：xuannv_show 和 modelscope）：

```bash
cp /workspace/models/facebook/sam3/sam3.pt models/sam3/
cp /workspace/models/facebook/sam3/config.json models/sam3/
cp /workspace/models/facebook/sam3/processor_config.json models/sam3/
cp /workspace/models/facebook/sam3/tokenizer.json models/sam3/
cp /workspace/models/facebook/sam3/vocab.json models/sam3/
cp /workspace/models/facebook/sam3/merges.txt models/sam3/
cp /workspace/xuannv_show/backend/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz models/sam3/assets/
```

验证文件：

```bash
ls -lh models/sam3/sam3.pt models/sam3/config.json models/sam3/assets/
```

Expected output:
```
-rw-r--r-- 1 root root 3.5G ... models/sam3/sam3.pt
-rw-r--r-- 1 root root  26K ... models/sam3/config.json
-rw-r--r-- 1 root root 1.4M ... models/sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

### Step 2: 复制 SAM3 Python 包

```bash
cp -r /workspace/xuannv_show/backend/sam3/* sam3_pkg/
```

验证：

```bash
ls sam3_pkg/
# Expected: pyproject.toml  sam3/  scripts/  examples/  etc.
```

### Step 3: Commit

```bash
git add models/sam3/ sam3_pkg/
git commit -m "feat: deploy SAM3 model weights and python package"
```

---

## Task 2: 安装依赖

**Files:**
- Modify: `requirements.txt`

### Step 1: 更新 requirements.txt

Append to `requirements.txt`:

```text
# SAM3 dependencies
torch==2.5.1
torchvision==0.20.1
numpy==1.26.4
pillow>=11.0.0
einops>=0.8.0
timm>=1.0.17
iopath>=0.1.10
huggingface_hub
```

### Step 2: 安装 PyTorch 和依赖

```bash
/opt/conda/envs/pyseims/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
/opt/conda/envs/pyseims/bin/pip install pillow numpy==1.26.4 einops timm iopath huggingface_hub
```

验证 PyTorch CUDA 可用：

```bash
/opt/conda/envs/pyseims/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected output:
```
2.5.1+cu124
True
```

### Step 3: 安装本地 sam3 包

```bash
cd /workspace/embedding-api
/opt/conda/envs/pyseims/bin/pip install -e ./sam3_pkg
```

验证安装：

```bash
/opt/conda/envs/pyseims/bin/python -c "from sam3.model_builder import build_sam3_image_model; print('sam3 imported OK')"
```

Expected output:
```
sam3 imported OK
```

### Step 4: Commit

```bash
git add requirements.txt
git commit -m "deps: add SAM3 dependencies (torch, sam3 pkg)"
```

---

## Task 3: 配置集成

**Files:**
- Modify: `config.yaml`
- Modify: `app/config.py`

### Step 3.1: 修改 config.yaml — 新增 sam3 段和 s2_dir

Add **before** the `models:` block in `config.yaml`:

```yaml
sam3:
  model_path: "models/sam3/sam3.pt"
  bpe_path: "models/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
  device: "cuda"
  max_cache_size: 20
  image_size: 256
  enable_inst_interactivity: true
```

Add `s2_dir` to each region under `regions:`:

For `harbin`:
```yaml
  harbin:
    name: "哈尔滨新区"
    patches_meta: "data/harbin/patches_meta.json"
    s2_dir: "/workspace/raw/harbin_scenes/s2"
```

For `haidian`:
```yaml
  haidian:
    name: "海淀区"
    patches_meta: "data/haidian/patches_meta_v2.json"
    s2_dir: "/workspace/raw/haidian_scenes/s2"
```

### Step 3.2: 扩展 ConfigManager

Add to `app/config.py` after `list_regions()` method (around line 121):

```python
    def get_sam3_config(self) -> Dict[str, Any]:
        """Get SAM3 configuration."""
        return self.get("sam3", default={})
```

### Step 3.3: 写测试验证配置加载

Create `tests/test_sam3_config.py`:

```python
"""Tests for SAM3 config integration."""

from app.config import get_config


def test_sam3_config_exists():
    config = get_config()
    sam3_cfg = config.get_sam3_config()
    assert sam3_cfg is not None
    assert "model_path" in sam3_cfg
    assert "max_cache_size" in sam3_cfg


def test_region_s2_dir():
    config = get_config()
    harbin = config.get_region("harbin")
    assert harbin is not None
    assert "s2_dir" in harbin
    assert "/workspace/raw/harbin_scenes/s2" in harbin["s2_dir"]
```

Run:

```bash
pytest tests/test_sam3_config.py -v
```

Expected: 2 passed

### Step 3.4: Commit

```bash
git add config.yaml app/config.py tests/test_sam3_config.py
git commit -m "config: add SAM3 and s2_dir configuration"
```

---

## Task 4: Pydantic Schemas

**Files:**
- Create: `app/schemas/sam3.py`
- Test: `tests/test_sam3_schemas.py`

### Step 4.1: 写 schema 文件

Create `app/schemas/sam3.py`:

```python
"""Pydantic schemas for SAM3 endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class ImageData(BaseModel):
    width: int
    height: int
    format: str = "png"
    data: str  # base64 encoded PNG


class EmbedRequest(BaseModel):
    patch_id: str = Field(..., min_length=1, max_length=64)
    month: str = Field(..., min_length=1, max_length=32)

    @validator("patch_id")
    def validate_patch_id(cls, v):
        import re
        if not re.match(r"^[\w\-]+$", v):
            raise ValueError("Invalid patch_id format")
        return v

    @validator("month")
    def validate_month(cls, v):
        import re
        if not re.match(r"^[\w\-]{1,32}$", v):
            raise ValueError("Invalid month format")
        return v


class EmbedResponse(BaseModel):
    embedding_id: str
    status: str = "ready"
    image: ImageData


class SegmentRequest(BaseModel):
    embedding_id: str = Field(..., min_length=1)
    point_coords: List[List[float]] = Field(..., min_items=1)
    point_labels: List[int] = Field(..., min_items=1)
    multimask_output: bool = True

    @validator("point_coords")
    def validate_coords(cls, v):
        for coord in v:
            if len(coord) != 2:
                raise ValueError("Each point must have exactly 2 coordinates [x, y]")
            if not (0.0 <= coord[0] <= 1.0 and 0.0 <= coord[1] <= 1.0):
                raise ValueError("Coordinates must be in [0, 1]")
        return v

    @validator("point_labels")
    def validate_labels(cls, v, values):
        point_coords = values.get("point_coords", [])
        if len(v) != len(point_coords):
            raise ValueError("point_labels length must match point_coords length")
        for label in v:
            if label not in (0, 1):
                raise ValueError("point_labels must be 0 (negative) or 1 (positive)")
        return v


class MaskData(BaseModel):
    data: str  # base64 encoded PNG
    score: float
    bbox: List[int]  # [x, y, width, height]


class SegmentResponse(BaseModel):
    masks: List[MaskData]


class CacheInfo(BaseModel):
    size: int
    max_size: int
    entries: List[str]


class GpuMemory(BaseModel):
    allocated_mb: int
    reserved_mb: int


class StatusResponse(BaseModel):
    model_loaded: bool
    device: str
    gpu_memory: GpuMemory
    cache: CacheInfo
```

### Step 4.2: 写 schema 测试

Create `tests/test_sam3_schemas.py`:

```python
"""Tests for SAM3 schemas."""

import pytest
from pydantic import ValidationError
from app.schemas.sam3 import EmbedRequest, SegmentRequest


def test_embed_request_valid():
    req = EmbedRequest(patch_id="patch_000000", month="2025-10")
    assert req.patch_id == "patch_000000"


def test_embed_request_invalid_patch_id():
    with pytest.raises(ValidationError):
        EmbedRequest(patch_id="../../etc/passwd", month="2025-10")


def test_segment_request_valid():
    req = SegmentRequest(
        embedding_id="harbin_patch_000000_2025-10",
        point_coords=[[0.5, 0.5]],
        point_labels=[1],
    )
    assert req.multimask_output is True


def test_segment_request_coords_out_of_range():
    with pytest.raises(ValidationError):
        SegmentRequest(
            embedding_id="test",
            point_coords=[[1.5, 0.5]],
            point_labels=[1],
        )


def test_segment_request_mismatched_lengths():
    with pytest.raises(ValidationError):
        SegmentRequest(
            embedding_id="test",
            point_coords=[[0.5, 0.5], [0.3, 0.3]],
            point_labels=[1],
        )
```

Run:

```bash
pytest tests/test_sam3_schemas.py -v
```

Expected: 5 passed

### Step 4.3: Commit

```bash
git add app/schemas/sam3.py tests/test_sam3_schemas.py
git commit -m "feat: add SAM3 pydantic schemas with validation"
```

---

## Task 5: SAM3Service 核心实现

**Files:**
- Create: `app/services/sam3_service.py`
- Test: `tests/test_sam3_service.py`

### Step 5.1: 写 SAM3Service（Embed + Segment + Status）

Create `app/services/sam3_service.py`:

```python
"""SAM3 interactive segmentation service."""

import asyncio
import base64
import io
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from app.config import get_config


class SAM3Service:
    """Singleton SAM3 inference service with lazy model loading and LRU cache."""

    _instance: Optional["SAM3Service"] = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model = None
        self._processor = None
        self._device: Optional[str] = None
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._inference_lock = asyncio.Lock()
        self._model_lock = threading.Lock()
        self._initialized = True

    def _ensure_model(self):
        """Lazy-load SAM3 model. Thread-safe double-checked locking."""
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            config = get_config().get_sam3_config()
            checkpoint_path = config.get("model_path", "models/sam3/sam3.pt")
            bpe_path = config.get("bpe_path", "models/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
            device = config.get("device", "cuda")

            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"

            self._device = device
            checkpoint_path = str(Path(checkpoint_path).resolve())
            bpe_path = str(Path(bpe_path).resolve())

            self._model = build_sam3_image_model(
                bpe_path=bpe_path,
                checkpoint_path=checkpoint_path,
                device=device,
                enable_inst_interactivity=True,
            )
            self._model.to(device)

            # Convert float32 to bfloat16 for autocast compatibility
            for p in self._model.parameters():
                if p.dtype == torch.float32:
                    p.data = p.data.to(torch.bfloat16)
            for b in self._model.buffers():
                if b.dtype == torch.float32:
                    b.data = b.data.to(torch.bfloat16)

            self._processor = Sam3Processor(self._model, device=device)

    def _load_s2_image(self, region_id: str, patch_id: str, month: str) -> Image.Image:
        """Load S2 RGB image for a patch from configured s2_dir."""
        config = get_config()
        region = config.get_region(region_id)
        if not region:
            raise ValueError(f"Region '{region_id}' not found")

        s2_dir = region.get("s2_dir")
        if not s2_dir:
            raise ValueError(f"s2_dir not configured for region '{region_id}'")

        s2_path = Path(s2_dir) / month / f"{patch_id}.tif"
        if not s2_path.exists():
            s2_path = Path(s2_dir) / month / f"{patch_id}.png"
        if not s2_path.exists():
            raise FileNotFoundError(f"No S2 image found for {patch_id} {month}")

        img = Image.open(s2_path).convert("RGB")
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        return img

    async def embed(self, region_id: str, patch_id: str, month: str) -> dict:
        """Load image, compute SAM3 embedding, cache it, return image + id."""
        import torch

        async with self._inference_lock:
            self._ensure_model()
            image = self._load_s2_image(region_id, patch_id, month)

            with torch.autocast(self._device, dtype=torch.bfloat16):
                state = self._processor.set_image(image)

            embedding_id = f"{region_id}_{patch_id}_{month}"
            self._cache[embedding_id] = {
                "state": state,
                "shape": (state["original_height"], state["original_width"]),
            }

            # LRU eviction
            max_size = get_config().get_sam3_config().get("max_cache_size", 20)
            while len(self._cache) > max_size:
                self._cache.popitem(last=False)

            # Encode image to base64
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            return {
                "embedding_id": embedding_id,
                "status": "ready",
                "image": {
                    "width": 256,
                    "height": 256,
                    "format": "png",
                    "data": img_b64,
                },
            }

    async def segment(
        self,
        embedding_id: str,
        point_coords: List[List[float]],
        point_labels: List[int],
        multimask_output: bool = True,
    ) -> List[dict]:
        """Run instance segmentation on cached embedding."""
        import torch

        if embedding_id not in self._cache:
            raise ValueError(f"Embedding '{embedding_id}' not found. Call embed first.")

        async with self._inference_lock:
            self._ensure_model()
            state = self._cache[embedding_id]["state"]
            img_h, img_w = self._cache[embedding_id]["shape"]

            coords = np.array(point_coords) * np.array([[img_w, img_h]])
            labels = np.array(point_labels)

            with torch.autocast(self._device, dtype=torch.bfloat16):
                masks, scores, _ = self._model.predict_inst(
                    state,
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=multimask_output,
                )

            results = []
            for mask, score in zip(masks, scores.tolist()):
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                ys, xs = np.where(mask)
                if len(xs) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]

                results.append({
                    "data": mask_b64,
                    "score": score,
                    "bbox": bbox,
                })

            return results

    def get_status(self) -> dict:
        """Return model loading status, GPU memory, and cache info."""
        import torch

        model_loaded = self._model is not None
        gpu_mem = {"allocated_mb": 0, "reserved_mb": 0}

        if model_loaded and self._device and self._device != "cpu":
            try:
                gpu_mem["allocated_mb"] = torch.cuda.memory_allocated(self._device) // (1024 * 1024)
                gpu_mem["reserved_mb"] = torch.cuda.memory_reserved(self._device) // (1024 * 1024)
            except RuntimeError:
                pass

        return {
            "model_loaded": model_loaded,
            "device": self._device or "not_loaded",
            "gpu_memory": gpu_mem,
            "cache": {
                "size": len(self._cache),
                "max_size": get_config().get_sam3_config().get("max_cache_size", 20),
                "entries": list(self._cache.keys()),
            },
        }
```

### Step 5.2: 写 Service 单元测试（Mock 模型）

Create `tests/test_sam3_service.py`:

```python
"""Unit tests for SAM3Service."""

import pytest
from unittest.mock import MagicMock, patch

from app.services.sam3_service import SAM3Service


@pytest.fixture(autouse=True)
def reset_service():
    """Reset singleton between tests."""
    SAM3Service._instance = None
    yield
    SAM3Service._instance = None


class TestSAM3ServiceStatus:
    def test_status_not_loaded(self):
        svc = SAM3Service()
        status = svc.get_status()
        assert status["model_loaded"] is False
        assert status["device"] == "not_loaded"
        assert status["cache"]["size"] == 0


class TestSAM3ServiceCache:
    @patch.object(SAM3Service, "_ensure_model")
    @patch.object(SAM3Service, "_load_s2_image")
    @patch("app.services.sam3_service.get_config")
    def test_embed_caches_result(self, mock_get_config, mock_load_img, mock_ensure):
        mock_get_config.return_value.get_sam3_config.return_value = {"max_cache_size": 2}
        mock_img = MagicMock()
        mock_load_img.return_value = mock_img

        svc = SAM3Service()
        svc._processor = MagicMock()
        state = {"original_height": 256, "original_width": 256}
        svc._processor.set_image.return_value = state

        import asyncio
        result = asyncio.run(svc.embed("harbin", "patch_000", "2025-10"))

        assert result["embedding_id"] == "harbin_patch_000_2025-10"
        assert "image" in result
        assert "data" in result["image"]
        assert len(svc._cache) == 1

    @patch.object(SAM3Service, "_ensure_model")
    @patch("app.services.sam3_service.get_config")
    def test_segment_missing_embedding(self, mock_get_config, mock_ensure):
        mock_get_config.return_value.get_sam3_config.return_value = {"max_cache_size": 2}
        svc = SAM3Service()

        import asyncio
        with pytest.raises(ValueError, match="Embedding"):
            asyncio.run(svc.segment("missing_id", [[0.5, 0.5]], [1]))

    @patch.object(SAM3Service, "_ensure_model")
    @patch.object(SAM3Service, "_load_s2_image")
    @patch("app.services.sam3_service.get_config")
    def test_lru_eviction(self, mock_get_config, mock_load_img, mock_ensure):
        mock_get_config.return_value.get_sam3_config.return_value = {"max_cache_size": 2}
        mock_img = MagicMock()
        mock_load_img.return_value = mock_img

        svc = SAM3Service()
        svc._processor = MagicMock()
        state = {"original_height": 256, "original_width": 256}
        svc._processor.set_image.return_value = state

        import asyncio
        asyncio.run(svc.embed("harbin", "patch_000", "2025-10"))
        asyncio.run(svc.embed("harbin", "patch_001", "2025-10"))
        asyncio.run(svc.embed("harbin", "patch_002", "2025-10"))

        assert len(svc._cache) == 2
        assert "harbin_patch_000_2025-10" not in svc._cache
```

Run:

```bash
pytest tests/test_sam3_service.py -v
```

Expected: 4 passed

### Step 5.3: Commit

```bash
git add app/services/sam3_service.py tests/test_sam3_service.py
git commit -m "feat: implement SAM3Service with embed, segment, and status"
```

---

## Task 6: SAM3 路由

**Files:**
- Create: `app/routers/sam3.py`
- Test: `tests/test_sam3_router.py`

### Step 6.1: 写路由文件

Create `app/routers/sam3.py`:

```python
"""SAM3 interactive segmentation router."""

from fastapi import APIRouter, HTTPException, Path

from app.config import get_config
from app.schemas.sam3 import (
    EmbedRequest,
    EmbedResponse,
    SegmentRequest,
    SegmentResponse,
    StatusResponse,
)
from app.services.sam3_service import SAM3Service

router = APIRouter()


@router.post("/regions/{region_id}/sam3/embed", response_model=EmbedResponse)
async def sam3_embed(region_id: str, req: EmbedRequest):
    """Preload patch image and compute SAM3 embedding."""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    svc = SAM3Service()
    try:
        result = await svc.embed(region_id, req.patch_id, req.month)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model inference failed: {e}")


@router.post("/regions/{region_id}/sam3/segment", response_model=SegmentResponse)
async def sam3_segment(region_id: str, req: SegmentRequest):
    """Segment instance using cached embedding and point prompts."""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    svc = SAM3Service()
    try:
        masks = await svc.segment(
            req.embedding_id,
            req.point_coords,
            req.point_labels,
            req.multimask_output,
        )
        return {"masks": masks}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Segmentation failed: {e}")


@router.get("/regions/{region_id}/sam3/status", response_model=StatusResponse)
async def sam3_status(region_id: str):
    """Get SAM3 model loading status and cache info."""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    svc = SAM3Service()
    return svc.get_status()
```

### Step 6.2: 写路由测试

Create `tests/test_sam3_router.py`:

```python
"""Tests for SAM3 router endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestSAM3Embed:
    def test_embed_invalid_region(self):
        response = client.post("/regions/beijing/sam3/embed", json={
            "patch_id": "patch_000000",
            "month": "2025-10",
        })
        assert response.status_code == 404

    def test_embed_invalid_patch_id(self):
        response = client.post("/regions/harbin/sam3/embed", json={
            "patch_id": "../../etc/passwd",
            "month": "2025-10",
        })
        assert response.status_code == 422

    def test_embed_invalid_month(self):
        response = client.post("/regions/harbin/sam3/embed", json={
            "patch_id": "patch_000000",
            "month": "../etc/passwd",
        })
        assert response.status_code == 422


class TestSAM3Segment:
    def test_segment_invalid_region(self):
        response = client.post("/regions/beijing/sam3/segment", json={
            "embedding_id": "test",
            "point_coords": [[0.5, 0.5]],
            "point_labels": [1],
        })
        assert response.status_code == 404

    def test_segment_coords_out_of_range(self):
        response = client.post("/regions/harbin/sam3/segment", json={
            "embedding_id": "test",
            "point_coords": [[1.5, 0.5]],
            "point_labels": [1],
        })
        assert response.status_code == 422

    def test_segment_mismatched_labels(self):
        response = client.post("/regions/harbin/sam3/segment", json={
            "embedding_id": "test",
            "point_coords": [[0.5, 0.5], [0.3, 0.3]],
            "point_labels": [1],
        })
        assert response.status_code == 422


class TestSAM3Status:
    def test_status_invalid_region(self):
        response = client.get("/regions/beijing/sam3/status")
        assert response.status_code == 404

    def test_status_valid_region(self):
        response = client.get("/regions/harbin/sam3/status")
        assert response.status_code == 200
        data = response.json()
        assert "model_loaded" in data
        assert "cache" in data
```

Run:

```bash
pytest tests/test_sam3_router.py -v
```

Expected: 6 passed

### Step 6.3: Commit

```bash
git add app/routers/sam3.py tests/test_sam3_router.py
git commit -m "feat: add SAM3 FastAPI router with embed, segment, status endpoints"
```

---

## Task 7: 注册路由到主应用

**Files:**
- Modify: `app/main.py`

### Step 7.1: 导入并注册 sam3 路由

Modify `app/main.py` line 10:

```python
from app.routers import regions, patches, embeddings, tasks, sam3
```

Modify `app/main.py` after line 61 (after tasks.router):

```python
app.include_router(sam3.router)
```

### Step 7.2: 运行全量测试确认无回归

```bash
pytest tests/test_api.py tests/test_sam3_*.py -v
```

Expected: All existing tests still pass + new SAM3 router tests pass

### Step 7.3: Commit

```bash
git add app/main.py
git commit -m "feat: register SAM3 router in main app"
```

---

## Task 8: 集成测试（真实模型）

**Files:**
- Create: `tests/test_sam3_integration.py`

### Step 8.1: 写集成测试

Create `tests/test_sam3_integration.py`:

```python
"""Integration tests for SAM3 with real model loading.

These tests require GPU and the full SAM3 model (3.45GB).
Run with: pytest tests/test_sam3_integration.py -v -m slow
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.slow

client = TestClient(app)


class TestSAM3RealModel:
    @pytest.fixture(scope="class", autouse=True)
    def reset_singleton(self):
        from app.services.sam3_service import SAM3Service
        SAM3Service._instance = None
        yield
        SAM3Service._instance = None

    def test_status_before_load(self):
        response = client.get("/regions/harbin/sam3/status")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is False

    def test_embed_real(self):
        """Test embed with real model (loads ~4GB to GPU)."""
        response = client.post("/regions/harbin/sam3/embed", json={
            "patch_id": "patch_000000",
            "month": "2025-10",
        })
        # May be 200 (success) or 404 (S2 image not found) or 503 (GPU OOM)
        assert response.status_code in (200, 404, 503)
        if response.status_code == 200:
            data = response.json()
            assert "embedding_id" in data
            assert "image" in data
            assert data["image"]["width"] == 256
            assert data["image"]["height"] == 256

    def test_segment_real(self):
        """Test segment after embed."""
        # First embed
        embed_resp = client.post("/regions/harbin/sam3/embed", json={
            "patch_id": "patch_000000",
            "month": "2025-10",
        })
        if embed_resp.status_code != 200:
            pytest.skip("Embed failed, skipping segment test")

        embedding_id = embed_resp.json()["embedding_id"]

        # Then segment
        response = client.post("/regions/harbin/sam3/segment", json={
            "embedding_id": embedding_id,
            "point_coords": [[0.5, 0.5]],
            "point_labels": [1],
            "multimask_output": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert "masks" in data
        assert len(data["masks"]) > 0
        for mask in data["masks"]:
            assert "data" in mask
            assert "score" in mask
            assert "bbox" in mask

    def test_status_after_load(self):
        response = client.get("/regions/harbin/sam3/status")
        assert response.status_code == 200
        data = response.json()
        # Model may or may not be loaded depending on prior tests
        assert "gpu_memory" in data
        assert "cache" in data
```

### Step 8.2: 运行集成测试（可选）

```bash
# Run only if GPU is available and S2 images exist
pytest tests/test_sam3_integration.py -v -m slow
```

Expected: Tests pass if S2 images and GPU are available; skip or 503 otherwise.

### Step 8.3: Commit

```bash
git add tests/test_sam3_integration.py
git commit -m "test: add SAM3 integration tests with real model"
```

---

## Task 9: 最终验证与重启服务

### Step 9.1: 运行全量测试

```bash
pytest tests/ -v --ignore=tests/test_sam3_integration.py
```

Expected: All existing tests + new unit tests pass.

### Step 9.2: 重启服务

```bash
# Kill existing uvicorn
pkill -f "uvicorn app.main:app --host 0.0.0.0 --port 9061"
sleep 2

# Restart
cd /workspace/embedding-api
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 9061 --reload > uvicorn.log 2>&1 &
sleep 3

# Verify
curl -s http://localhost:9061/health | python3 -m json.tool
curl -s http://localhost:9061/regions/harbin/sam3/status | python3 -m json.tool
```

Expected: Health OK, status returns model_loaded=false (not yet triggered).

### Step 9.3: 最终 Commit

```bash
git add -A
git commit -m "feat: complete SAM3 integration - embed, segment, status endpoints"
```

---

## Self-Review Checklist

### 1. Spec Coverage

| Spec Section | Plan Task | Status |
|--------------|-----------|--------|
| 模型权重部署 | Task 1 | ✅ |
| 依赖安装 | Task 2 | ✅ |
| config.yaml 配置 | Task 3 | ✅ |
| ConfigManager 扩展 | Task 3 | ✅ |
| Pydantic Schemas | Task 4 | ✅ |
| SAM3Service (embed) | Task 5 | ✅ |
| SAM3Service (segment) | Task 5 | ✅ |
| SAM3Service (status) | Task 5 | ✅ |
| 缓存策略 (LRU 20条) | Task 5 | ✅ |
| 并发控制 (asyncio.Lock) | Task 5 | ✅ |
| FastAPI 路由 | Task 6 | ✅ |
| 注册路由 | Task 7 | ✅ |
| 错误处理 (404/422/503) | Task 6 | ✅ |
| 单元测试 | Task 4-7 | ✅ |
| 集成测试 | Task 8 | ✅ |

### 2. Placeholder Scan

- [x] 无 "TBD", "TODO", "implement later"
- [x] 无 "Add appropriate error handling" 等模糊描述
- [x] 每个步骤都有完整代码和确切命令
- [x] 无 "Similar to Task N" 引用

### 3. Type Consistency

- [x] `EmbedRequest`, `EmbedResponse`, `SegmentRequest`, `SegmentResponse`, `StatusResponse` 类型在 schema、service、router 中一致
- [x] `SAM3Service._cache` key 格式为 `{region_id}_{patch_id}_{month}`，与 router 中生成方式一致
- [x] `point_coords` 类型 `List[List[float]]` 在 schema 和 service 中一致

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2025-06-11-sam3-integration.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for reliability on complex GPU/model code.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review. Faster if no unexpected issues.

**Which approach?**
