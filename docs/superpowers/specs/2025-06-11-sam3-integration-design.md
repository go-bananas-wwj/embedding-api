# SAM3 交互式分割集成设计文档

**日期**: 2025-06-11  
**项目**: embedding-api  
**范围**: 将 xuannv_show 的 SAM3 交互式实例分割功能集成到 embedding-api 后台服务

---

## 1. 背景与目标

### 1.1 现状

- **embedding-api** 目前是一个纯数据服务（只读），通过 FastAPI 提供预计算的 embedding、任务预测和标签查询。
- **xuannv_show** 已具备完整的 SAM3 交互式分割能力：用户选择 patch → 预加载影像 → 点击点坐标 → 获取分割掩码。

### 1.2 目标

在 **embedding-api** 中新增 SAM3 交互式分割端点，使其具备运行时推理能力，同时：
- 不修改 xuannv_show 项目的任何文件
- 将 SAM3 模型权重和依赖完整部署在 embedding-api 项目内部
- 保持与现有 API 架构风格一致

---

## 2. 方案选择

选择 **方案 A：在 pyseims 环境中完整部署**。

理由：
- 单进程服务，延迟最低，适合交互式场景
- 直接复用 xuannv_show 经过验证的 SAM3Client 逻辑（复制到本项目，不修改源文件）
- 将 xuannv_show 的 sam3 Python 包完整复制到本项目的 `sam3_pkg/`，通过 `pip install -e ./sam3_pkg` 安装（不依赖外部路径）
- 代码量和架构复杂度最低

---

## 3. 模型权重与文件部署

### 3.1 目录结构

```
embedding-api/
├── models/
│   └── sam3/
│       ├── sam3.pt                          ← 主模型权重 (3.45GB)
│       ├── config.json                      ← 模型配置
│       ├── processor_config.json            ← Processor 配置
│       ├── tokenizer.json                   ← Tokenizer
│       ├── vocab.json                       ← Vocab
│       ├── merges.txt                       ← BPE merges
│       └── assets/
│           └── bpe_simple_vocab_16e6.txt.gz ← BPE tokenizer 字典
└── sam3_pkg/                                ← SAM3 Python 包副本
    ├── pyproject.toml
    └── sam3/
        ├── __init__.py
        ├── model_builder.py
        ├── model/
        │   ├── sam3_image.py
        │   ├── sam3_image_processor.py
        │   └── ...
        └── assets/
```

### 3.2 文件来源

| 文件 | 来源路径 | 大小 |
|------|----------|------|
| `sam3.pt` | `/workspace/data/models/facebook/sam3/sam3.pt` | ~3.45GB |
| `config.json` | `/workspace/data/models/facebook/sam3/config.json` | ~26KB |
| `processor_config.json` | `/workspace/data/models/facebook/sam3/processor_config.json` | ~1.7KB |
| `tokenizer.json` | `/workspace/data/models/facebook/sam3/tokenizer.json` | ~3.6MB |
| `vocab.json` | `/workspace/data/models/facebook/sam3/vocab.json` | ~862KB |
| `merges.txt` | `/workspace/data/models/facebook/sam3/merges.txt` | ~525KB |
| `bpe_simple_vocab_16e6.txt.gz` | `/workspace/projects/xuannv-show/backend/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz` | ~1.36MB |
| `sam3_pkg/` | `/workspace/projects/xuannv-show/backend/sam3/` 完整复制 | ~100MB |

> **注意**: `sam3_pkg/` 是源码副本，后续若 xuannv_show 的 sam3 有更新，可手动同步。

---

## 4. 依赖安装

在 `pyseims` 环境（Python 3.9）中执行：

```bash
# 1. PyTorch + CUDA
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 2. SAM3 包依赖
pip install pillow numpy==1.26.4 einops timm iopath huggingface_hub

# 3. 本地安装 sam3 包（不修改 xuannv_show 源文件）
pip install -e ./sam3_pkg
```

更新 `requirements.txt`：

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

---

## 5. API 端点设计

### 5.1 端点清单

| 方法 | 路径 | 功能 | 认证 |
|------|------|------|------|
| `POST` | `/regions/{region_id}/sam3/embed` | 预加载 patch 影像，计算 embedding，返回影像 + embedding_id | 无需 |
| `POST` | `/regions/{region_id}/sam3/segment` | 基于 embedding 和点提示生成分割掩码 | 无需 |
| `GET` | `/regions/{region_id}/sam3/status` | 查询 SAM3 模型加载状态、GPU 内存、缓存信息 | 无需 |

### 5.2 Embed 端点

**请求**:
```json
POST /regions/harbin/sam3/embed
{
  "patch_id": "patch_000000",
  "month": "2025-10"
}
```

**响应**:
```json
{
  "embedding_id": "harbin_patch_000000_2025-10",
  "status": "ready",
  "image": {
    "width": 256,
    "height": 256,
    "format": "png",
    "data": "iVBORw0KGgoAAAANSUhEUg..."
  }
}
```

**说明**:
- `image.data` 为 S2 RGB 自然色影像的 base64 PNG（256×256）
- 前端解码后可立即显示影像，供用户点击
- `embedding_id` 格式: `{region_id}_{patch_id}_{month}`

### 5.3 Segment 端点

**请求**:
```json
POST /regions/harbin/sam3/segment
{
  "embedding_id": "harbin_patch_000000_2025-10",
  "point_coords": [[0.52, 0.48], [0.35, 0.62]],
  "point_labels": [1, 0],
  "multimask_output": true
}
```

**响应**:
```json
{
  "masks": [
    {
      "data": "iVBORw0KGgoAAAANSUhEUg...",
      "score": 0.95,
      "bbox": [120, 110, 35, 42]
    },
    {
      "data": "iVBORw0KGgoAAAANSUhEUg...",
      "score": 0.87,
      "bbox": [118, 108, 40, 45]
    }
  ]
}
```

**说明**:
- `point_coords`: 归一化坐标 `[[x, y], ...]`，范围 `[0, 1]`
- `point_labels`: `1` = 正样本（前景），`0` = 负样本（背景）
- `multimask_output`: `true` 返回 3 个候选掩码，`false` 返回 1 个
- `bbox`: `[x, y, width, height]` 像素坐标

### 5.4 Status 端点

**响应**:
```json
{
  "model_loaded": true,
  "device": "cuda:6",
  "gpu_memory": {
    "allocated_mb": 3842,
    "reserved_mb": 4096
  },
  "cache": {
    "size": 3,
    "max_size": 20,
    "entries": [
      "harbin_patch_000000_2025-10",
      "harbin_patch_000001_2025-10"
    ]
  }
}
```

---

## 6. 服务层架构

### 6.1 新增文件

```
app/
├── routers/
│   └── sam3.py              # FastAPI 路由定义
├── services/
│   └── sam3_service.py      # SAM3Service 核心推理逻辑
├── schemas/
│   └── sam3.py              # Pydantic 请求/响应模型
└── config.py                # 新增 sam3 配置段（已有文件修改）
```

### 6.2 S2 RGB 影像加载

SAM3 需要 256×256 的 RGB 自然色影像作为输入。在 embedding-api 中，S2 影像从 `config.yaml` 配置的 `s2_dir` 路径加载：

```yaml
regions:
  harbin:
    s2_dir: "/workspace/data/raw/harbin_scenes/s2"
```

`SAM3Service` 内部调用 `load_s2_rgb_natural(patch_id, month, out_size=256)`：
- 根据 `region_id` 找到对应的 `s2_dir`
- 定位 `{s2_dir}/{month}/{patch_id}.tif` 或 `.png`
- 读取 RGB 波段 → 归一化 → resize 到 256×256

> 该函数逻辑从 xuannv_show 的 `app/services/patch_image_loader.py` 复制/改编到 `app/services/sam3_service.py` 内部，不引入外部依赖。

### 6.3 SAM3Service (`app/services/sam3_service.py`)

核心类，负责模型加载、embedding 计算、分割推理和缓存管理。

```python
class SAM3Service:
    """Singleton SAM3 inference service."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._lock = asyncio.Lock()
        self._model_lock = threading.Lock()

    def load_model(self) -> None:
        """Lazy-load SAM3 model. Thread-safe double-checked locking."""
        ...

    async def embed(self, region_id: str, patch_id: str, month: str) -> dict:
        """Load S2 RGB, compute embedding, cache it, return image + embedding_id."""
        ...

    async def segment(
        self,
        embedding_id: str,
        point_coords: list[list[float]],
        point_labels: list[int],
        multimask_output: bool = True,
    ) -> list[dict]:
        """Run instance segmentation on cached embedding."""
        ...

    def get_status(self) -> dict:
        """Return model/GPU/cache status."""
        ...
```

### 6.3 关键行为

- **懒加载**: 首次调用 `embed()` 时加载模型，避免服务启动时长时间阻塞
- **双检锁**: 多并发请求同时到达时，仅加载一次模型（`threading.Lock`）
- **异步锁**: `embed()` 和 `segment()` 使用 `asyncio.Lock` 串行化，防止 GPU 内存冲突
- **LRU 缓存**: `OrderedDict` 实现，最大 20 条，超限时移除最旧的 embedding

---

## 7. 数据流

### 7.1 Embed 流程

```
Client
  │ POST /regions/{rid}/sam3/embed {patch_id, month}
  ▼
Router (app/routers/sam3.py)
  ├── Validate patch_id (regex)
  ├── Validate month (regex)
  └── Call SAM3Service.embed()
        │
        ▼
        Load S2 RGB image
          ├── Resolve embedding path from config.yaml
          └── Load .npy → extract RGB bands → normalize → resize 256×256
        │
        ▼
        SAM3Service.load_model() [lazy, once]
          ├── build_sam3_image_model(checkpoint, device)
          ├── Convert float32 → bfloat16
          └── Sam3Processor(model, device)
        │
        ▼
        processor.set_image(image) → state dict
        │
        ▼
        Cache state in OrderedDict [key=embedding_id]
        │
        ▼
        Encode image to base64 PNG
        │
        ▼
        Return {embedding_id, status, image}
```

### 7.2 Segment 流程

```
Client
  │ POST /regions/{rid}/sam3/segment {embedding_id, point_coords, point_labels}
  ▼
Router
  ├── Validate embedding_id format
  ├── Validate point_coords (0~1 range)
  └── Call SAM3Service.segment()
        │
        ▼
        Lookup cache by embedding_id
          ├── Miss → raise 404
          └── Hit → get state dict
        │
        ▼
        Normalize coords → pixel coords (× image shape)
        │
        ▼
        model.predict_inst(state, point_coords, point_labels, multimask_output)
        │
        ▼
        Encode each mask → base64 PNG
        │
        ▼
        Return {masks: [{data, score, bbox}, ...]}
```

---

## 8. 缓存与并发策略

### 8.1 Embedding 缓存

- **数据结构**: `OrderedDict[str, dict]`
- **Key**: `{region_id}_{patch_id}_{month}`
- **Value**: `{"state": <sam3_state>, "shape": (H, W)}`
- **容量上限**: 20 个 embedding（每个约 200-500MB GPU 内存）
- **淘汰策略**: LRU，超出时 `popitem(last=False)`

### 8.2 并发控制

- **模型加载锁**: `threading.Lock()`，确保只有一个线程执行模型初始化
- **推理锁**: `asyncio.Lock()`，串行化所有 `embed()` 和 `segment()` 调用
  - 原因: SAM3 推理占用大量 GPU 显存，并发推理极易 OOM
  - 影响: 单请求延迟 <1s，对交互式场景可接受

### 8.3 GPU 设备选择

复用 xuannv_show 的策略：
1. 优先 `cuda:6`（业务约束）
2. Fallback 到显存最空闲的 CUDA 设备
3. 无 GPU 时回退到 `cpu`

---

## 9. 配置集成

### 9.1 config.yaml 新增段

```yaml
sam3:
  model_path: "models/sam3/sam3.pt"
  bpe_path: "models/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
  device: "cuda"           # "cuda", "cpu", or "cuda:6"
  max_cache_size: 20
  image_size: 256          # S2 RGB 输入尺寸
  enable_inst_interactivity: true
```

### 9.2 ConfigManager 扩展

`app/config.py` 的 `ConfigManager` 新增：
- `get_sam3_config()` → 返回 SAM3 配置字典
- 热重载兼容（watchdog 监控 config.yaml）

---

## 10. 错误处理

| 场景 | HTTP 状态码 | 错误详情 | 处理逻辑 |
|------|-------------|----------|----------|
| 模型文件不存在 | `503` | `SAM3 model checkpoint not found at ...` | 启动时检查，缺失则标记服务不可用 |
| GPU OOM | `503` | `GPU out of memory. Cache cleared, please retry.` | 清空缓存重试一次，仍失败则报错 |
| 无效 `patch_id` | `400` / `404` | `Invalid patch_id format` | 复用现有 `_validate_patch_id` |
| 无效 `month` | `422` | `Invalid month format` | 复用现有 month 正则校验 |
| `embedding_id` 不存在 | `404` | `Embedding not found. Call embed first.` | 缓存未命中 |
| `point_coords` 越界 | `422` | `point_coords must be in [0, 1]` | Pydantic validator |
| 区域不存在 | `404` | `Region not found` | 复用现有 `region_exists()` |
| 影像加载失败 | `404` | `No S2 image found for patch_id month` | 文件系统缺失 |

---

## 11. 测试策略

### 11.1 单元测试 (`tests/test_sam3.py`)

- **路由校验测试**: 无效 patch_id、越界 point_coords、非法 month → 422
- **缓存测试**: embed → cache hit → segment → embed 新 patch → LRU 淘汰
- **状态端点测试**: 模型加载前后状态变化
- **Mock 测试**: 使用 mock SAM3 模型，验证数据流正确性

### 11.2 集成测试

- **真实推理测试**（标记 `@pytest.mark.slow`）:
  - 加载真实 SAM3 模型
  - 对 `patch_000000` 执行完整 embed + segment
  - 验证返回的 mask 尺寸和 score 范围

### 11.3 并发测试

- 同时发送 5 个 embed 请求，验证：
  - 模型只加载一次
  - 无 GPU OOM
  - 请求按序完成

---

## 12. 实施文件清单

### 12.1 新增文件

| 文件 | 说明 |
|------|------|
| `models/sam3/` | 模型权重和配置文件目录 |
| `sam3_pkg/` | SAM3 Python 包副本 |
| `app/routers/sam3.py` | SAM3 API 路由 |
| `app/services/sam3_service.py` | SAM3 推理服务 |
| `app/schemas/sam3.py` | Pydantic 模型 |
| `tests/test_sam3.py` | 单元测试 |

### 12.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| `app/main.py` | 注册 `sam3` router |
| `app/config.py` | 新增 `get_sam3_config()` |
| `config.yaml` | 新增 `sam3:` 配置段 |
| `requirements.txt` | 新增 PyTorch、SAM3 依赖 |

---

## 13. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| pyseims 环境安装 PyTorch 失败 | 服务无法推理 | 先验证 `torch.cuda.is_available()`；失败则优雅降级为 CPU 推理 |
| GPU 内存不足 | OOM，服务崩溃 | LRU 缓存限制 20 条；OOM 时自动清空缓存重试；提供 `/status` 供监控 |
| 模型加载时间过长 | 首次请求超时 | 懒加载 + 服务启动后自动预热（warmup） |
| sam3_pkg 与 xuannv_show 版本不同步 | 行为差异 | 文档化同步流程；定期 diff 检查 |
| 并发请求导致 OOM | 服务不可用 | `asyncio.Lock` 串行化推理；客户端队列处理 |

---

## 14. 验收标准

- [ ] `POST /regions/harbin/sam3/embed` 返回 `embedding_id` + base64 影像
- [ ] `POST /regions/harbin/sam3/segment` 接收点坐标返回分割掩码
- [ ] `GET /regions/harbin/sam3/status` 返回模型/GPU/缓存状态
- [ ] 无效 patch_id / month 返回 422
- [ ] embedding_id 不存在返回 404
- [ ] 缓存 LRU 淘汰正常工作
- [ ] 模型只加载一次
- [ ] 单元测试全部通过
- [ ] 集成测试（真实模型）通过
