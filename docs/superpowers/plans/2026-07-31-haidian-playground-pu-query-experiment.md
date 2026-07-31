# 海淀学校田径操场 PU + Query 实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用可信 OSM 学校田径操场 Polygon 复现 PU + Query 杂散误检，解释操场与误检区域的得分差异，并比较严格阈值和双阈值连通约束。

**Architecture:** 新增一个 OSM 数据模块，负责查询、缓存、筛选和将 WGS84 操场几何栅格化到海淀 128×128 Patch；新增一个独立实验脚本，直接复用生产 `train_pu_query` 与 `score_pu_query`，按支持/校准/测试 Patch 隔离运行三种方法。实验只输出 `Tmp` 画廊、JSON 指标和可复现清单，不修改线上 API。

**Tech Stack:** Python 3.9、NumPy、Shapely、PyProj、Rasterio、SciPy、Pillow、Matplotlib、生产 `app.services.pu_query`

## Global Constraints

- 只纳入 `leisure=track|pitch` 且 `sport=athletics` 的典型学校田径操场。
- 排除普通篮球场、网球场、单独足球场、体育馆和不可靠线要素。
- 使用海淀 2026 年 4 月 P10C 64 维 embedding 与同期高分辨率光学影像。
- 支持、校准、测试 Patch 不重叠；测试标签不得参与阈值选择。
- OSM 是不完整参考真值，未标注区域不得自动作为严格负样本解释。
- 本阶段只做离线实验，不修改线上 API 行为。

---

### Task 1: OSM 操场查询、筛选与 Patch 栅格化

**Files:**
- Create: `scripts/playground_osm.py`
- Create: `tests/test_playground_osm.py`
- Generate: `data/haidian/labels/osm_playgrounds/osm_raw.json`
- Generate: `data/haidian/labels/osm_playgrounds/playgrounds.geojson`
- Generate: `data/haidian/labels/osm_playgrounds/manifest.json`

**Interfaces:**
- Consumes: `data/haidian/patches_meta_v2.json`
- Produces: `fetch_overpass(bounds: tuple[float, float, float, float], cache_path: Path) -> dict`
- Produces: `extract_playgrounds(payload: dict) -> list[PlaygroundFeature]`
- Produces: `rasterize_feature(feature: PlaygroundFeature, patch: dict, shape: tuple[int, int] = (128, 128)) -> np.ndarray`
- Produces: `build_dataset(output_root: Path) -> dict`

- [ ] **Step 1: 写 OSM 标签过滤与栅格化失败测试**

```python
def test_extract_playgrounds_keeps_only_athletics_areas():
    payload = {"elements": [
        {"type": "way", "id": 1, "tags": {"leisure": "track", "sport": "athletics"},
         "geometry": [{"lon": 116.30, "lat": 39.95}, {"lon": 116.31, "lat": 39.95},
                      {"lon": 116.31, "lat": 39.96}, {"lon": 116.30, "lat": 39.95}]},
        {"type": "way", "id": 2, "tags": {"leisure": "pitch", "sport": "basketball"},
         "geometry": [{"lon": 116.30, "lat": 39.95}] * 4},
    ]}
    assert [item.osm_id for item in extract_playgrounds(payload)] == [1]


def test_rasterized_playground_intersects_only_matching_patch():
    feature = PlaygroundFeature(
        osm_id=1,
        name="测试田径场",
        geometry=Polygon([(116.30, 39.95), (116.31, 39.95),
                          (116.31, 39.96), (116.30, 39.96)]),
        tags={"leisure": "track", "sport": "athletics"},
    )
    mask = rasterize_feature(feature, PATCH_FIXTURE)
    assert mask.dtype == np.bool_
    assert 0 < mask.sum() < mask.size
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `/opt/conda/envs/pyseims/bin/python -m pytest tests/test_playground_osm.py -q`

Expected: FAIL with `ModuleNotFoundError: scripts.playground_osm`

- [ ] **Step 3: 实现 Overpass 查询和严格标签过滤**

```python
OVERPASS_QUERY = """
[out:json][timeout:120];
(
  way["leisure"="track"]["sport"="athletics"]({south},{west},{north},{east});
  way["leisure"="pitch"]["sport"="athletics"]({south},{west},{north},{east});
);
out tags geom;
"""

@dataclass(frozen=True)
class PlaygroundFeature:
    osm_id: int
    name: str
    geometry: Polygon
    tags: dict[str, str]


def extract_playgrounds(payload: dict) -> list[PlaygroundFeature]:
    result = []
    for element in payload.get("elements", []):
        tags = element.get("tags", {})
        if tags.get("sport") != "athletics":
            continue
        if tags.get("leisure") not in {"track", "pitch"}:
            continue
        coordinates = [(point["lon"], point["lat"]) for point in element.get("geometry", [])]
        if len(coordinates) < 4:
            continue
        polygon = Polygon(coordinates).buffer(0)
        if polygon.is_empty or not polygon.is_valid:
            continue
        result.append(PlaygroundFeature(
            osm_id=int(element["id"]),
            name=tags.get("name:zh") or tags.get("name") or f"OSM way {element['id']}",
            geometry=polygon,
            tags=tags,
        ))
    return result
```

- [ ] **Step 4: 实现 WGS84 几何到 Patch 像素掩膜的转换**

使用 `pyproj.Transformer.from_crs("EPSG:4326", patch["crs"], always_xy=True)`，
`rasterio.transform.from_bounds(*patch["bounds"], width=128, height=128)` 和
`rasterio.features.rasterize` 生成布尔掩膜。只保留至少覆盖 4 个 embedding 像素
且与 Patch 相交的样本。

- [ ] **Step 5: 下载并缓存 OSM 原始响应，生成人工可审查清单**

Run:

```bash
/opt/conda/envs/pyseims/bin/python scripts/playground_osm.py \
  --output data/haidian/labels/osm_playgrounds
```

Expected:

- `osm_raw.json` 保存原始 Overpass 响应
- `playgrounds.geojson` 保存筛选后的 Polygon 与 OSM ID
- `manifest.json` 记录元素 ID、名称、标签、命中 Patch 和像素数
- 至少 3 个可信 Polygon；不足时脚本以非零状态退出并打印覆盖限制

- [ ] **Step 6: 运行 OSM 模块测试**

Run: `/opt/conda/envs/pyseims/bin/python -m pytest tests/test_playground_osm.py -q`

Expected: PASS

- [ ] **Step 7: 提交 OSM 数据模块与可复现清单**

```bash
git add scripts/playground_osm.py tests/test_playground_osm.py \
  data/haidian/labels/osm_playgrounds/playgrounds.geojson \
  data/haidian/labels/osm_playgrounds/manifest.json
git commit -m "feat: build Haidian OSM athletics playground labels"
```

---

### Task 2: 阈值与连通区域后处理

**Files:**
- Create: `scripts/playground_pu_postprocess.py`
- Create: `tests/test_playground_pu_postprocess.py`

**Interfaces:**
- Consumes: production PU + Query continuous score arrays
- Produces: `strict_threshold(scores: list[np.ndarray], labels: list[np.ndarray]) -> float`
- Produces: `hysteresis_prediction(score: np.ndarray, high: float, low: float, min_pixels: int) -> np.ndarray`
- Produces: `binary_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]`
- Produces: `component_statistics(prediction: np.ndarray, score: np.ndarray) -> list[dict]`

- [ ] **Step 1: 写双阈值连通约束失败测试**

```python
def test_hysteresis_keeps_low_score_pixels_only_when_connected_to_seed():
    score = np.zeros((12, 12), dtype=np.float32)
    score[2:5, 2:5] = 0.62
    score[3, 3] = 0.91
    score[8:10, 8:10] = 0.70
    result = hysteresis_prediction(score, high=0.85, low=0.60, min_pixels=4)
    assert result[2:5, 2:5].all()
    assert not result[8:10, 8:10].any()


def test_hysteresis_removes_tiny_seeded_component():
    score = np.zeros((8, 8), dtype=np.float32)
    score[1, 1] = 0.95
    result = hysteresis_prediction(score, high=0.90, low=0.60, min_pixels=4)
    assert not result.any()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `/opt/conda/envs/pyseims/bin/python -m pytest tests/test_playground_pu_postprocess.py -q`

Expected: FAIL with missing module/function

- [ ] **Step 3: 实现严格阈值校准**

只使用校准 Patch，在生产阈值到得分 `99.9%` 分位数范围内扫描 180 个阈值。
主排序指标为 F1；F1 相同时依次选择 Precision 更高、预测面积更小的阈值。

- [ ] **Step 4: 实现双阈值连通区域约束**

```python
def hysteresis_prediction(score, high, low, min_pixels):
    candidates = score >= low
    seeds = score >= high
    components, count = ndimage.label(candidates, structure=np.ones((3, 3)))
    result = np.zeros_like(candidates)
    for component_id in range(1, count + 1):
        component = components == component_id
        if component.sum() >= min_pixels and np.any(seeds[component]):
            result[component] = True
    return result
```

- [ ] **Step 5: 实现指标和连通区域诊断**

返回 `precision`、`recall`、`f1`、`iou`、`positive_ratio`、
`component_count`；每个连通区域额外记录面积、平均分、最大分和边界框。

- [ ] **Step 6: 运行后处理测试**

Run: `/opt/conda/envs/pyseims/bin/python -m pytest tests/test_playground_pu_postprocess.py -q`

Expected: PASS

- [ ] **Step 7: 提交后处理模块**

```bash
git add scripts/playground_pu_postprocess.py tests/test_playground_pu_postprocess.py
git commit -m "feat: add guarded PU query playground postprocessing"
```

---

### Task 3: 运行隔离实验并诊断误检分数

**Files:**
- Create: `scripts/experiment_playground_pu_query.py`
- Create: `tests/test_playground_pu_query_experiment.py`
- Generate: `Tmp/playground_pu_query_20260731/experiment_manifest.json`
- Generate: `Tmp/playground_pu_query_20260731/metrics.json`
- Generate: `Tmp/playground_pu_query_20260731/score_groups.json`

**Interfaces:**
- Consumes: `build_dataset` manifest、`train_pu_query`、`score_pu_query`
- Consumes: `strict_threshold`、`hysteresis_prediction`、`binary_metrics`
- Produces: `split_patch_groups(items: list[dict], seed: int = 42) -> dict[str, list[str]]`
- Produces: `run_experiment(output: Path) -> dict`

- [ ] **Step 1: 写数据隔离失败测试**

```python
def test_support_calibration_and_test_patches_are_disjoint():
    split = split_patch_groups(PATCH_ITEMS, seed=42)
    support = set(split["support"])
    calibration = set(split["calibration"])
    test = set(split["test"])
    assert support.isdisjoint(calibration)
    assert support.isdisjoint(test)
    assert calibration.isdisjoint(test)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `/opt/conda/envs/pyseims/bin/python -m pytest tests/test_playground_pu_query_experiment.py -q`

Expected: FAIL with missing module/function

- [ ] **Step 3: 实现固定种子的支持/校准/测试划分**

优先按 OSM 元素分组，避免同一操场跨 Patch 时泄漏。支持集使用 1–2 个操场
Polygon，校准集至少 1 个独立操场，测试集包含所有剩余操场 Patch 和至少 8 个
固定随机背景 Patch。

- [ ] **Step 4: 使用生产 PU + Query 训练并保存 checkpoint 摘要**

```python
model = train_pu_query([
    (item["support_key"], np.load(item["embedding"]), np.load(item["mask"]))
    for item in support_items
])
```

保存自动阈值、`training_f05`、前景/背景中心范数、支持 Polygon 数量和支持 Patch。
不复制算法实现，保证实验与线上逻辑一致。

- [ ] **Step 5: 在校准集选择严格阈值与双阈值参数**

- `baseline_threshold = model["threshold"]`
- `strict = strict_threshold(calibration_scores, calibration_labels)`
- `high = strict`
- `low` 在 `[baseline_threshold, strict]` 中扫描，按校准 F1 选择
- `min_pixels` 在 `{4, 8, 16, 24}` 中按校准 F1 选择

参数选定后冻结，再运行测试 Patch。

- [ ] **Step 6: 统计真实操场与高分误检的差异**

对 OSM 操场内部、边界 3 像素环、基线误检连通区域、随机未标注背景分别保存：
样本数、均值、标准差、P05、P25、中位数、P75、P95、最大值。额外统计
`foreground_similarity`、`background_similarity` 和最终 margin，记录每个
误检连通区域在光学影像中的位置供人工复核。

- [ ] **Step 7: 运行完整实验**

Run:

```bash
/opt/conda/envs/pyseims/bin/python scripts/experiment_playground_pu_query.py \
  --month 202604 \
  --labels data/haidian/labels/osm_playgrounds/manifest.json \
  --output Tmp/playground_pu_query_20260731
```

Expected: 输出三种方法的逐 Patch 和汇总指标、分数分组统计及完整实验清单。

- [ ] **Step 8: 运行实验模块测试**

Run:

```bash
/opt/conda/envs/pyseims/bin/python -m pytest \
  tests/test_playground_osm.py \
  tests/test_playground_pu_postprocess.py \
  tests/test_playground_pu_query_experiment.py \
  tests/test_pu_query.py -q
```

Expected: PASS

- [ ] **Step 9: 提交实验脚本**

```bash
git add scripts/experiment_playground_pu_query.py \
  tests/test_playground_pu_query_experiment.py
git commit -m "test: diagnose playground PU query false positives"
```

---

### Task 4: 生成可放大画廊并验证结论

**Files:**
- Create: `scripts/build_playground_pu_query_gallery.py`
- Generate: `Tmp/playground_pu_query_20260731/index.html`
- Generate: `Tmp/playground_pu_query_20260731/assets/*.png`
- Generate: `Tmp/playground_pu_query_20260731/report.md`

**Interfaces:**
- Consumes: Task 3 的 `experiment_manifest.json`、`metrics.json`、`score_groups.json`
- Produces: 可放大 HTML 画廊与人类可读结论

- [ ] **Step 1: 生成每个案例的八列对照图**

按设计顺序绘制光学影像、OSM 标签、模型可见标签、连续得分热图、基线、
严格阈值、双阈值结果和最终叠加。训练、校准、测试身份必须显示在标题中。

- [ ] **Step 2: 绘制得分分布**

输出操场内部、边界、误检区域和随机背景的直方图与箱线图；所有图共享同一
分数坐标范围，并用竖线标出基线阈值、严格阈值和双阈值上下界。

- [ ] **Step 3: 生成人类可读诊断报告**

报告必须回答：

- 当前 PU + Query 是否存在操场杂散误检
- 操场和误检像素的 margin 分布是否可由单一阈值分离
- Query adaptation 对预测面积的影响
- 严格阈值造成的 Recall 损失
- 双阈值连通约束减少了多少误检连通区域
- OSM 标签不完整造成的指标偏差

- [ ] **Step 4: 启动画廊服务并做资源检查**

Run:

```bash
/opt/conda/envs/pyseims/bin/python -m http.server 61846 --bind 0.0.0.0
curl -I http://127.0.0.1:61846/Tmp/playground_pu_query_20260731/
```

Expected: HTTP 200，所有图片资源 HTTP 200。

- [ ] **Step 5: 最终验证**

Run:

```bash
/opt/conda/envs/pyseims/bin/python -m pytest \
  tests/test_playground_osm.py \
  tests/test_playground_pu_postprocess.py \
  tests/test_playground_pu_query_experiment.py \
  tests/test_pu_query.py -q
git diff --check
```

Expected: 所有测试通过且无空白错误。

- [ ] **Step 6: 提交画廊生成器与报告**

```bash
git add scripts/build_playground_pu_query_gallery.py \
  Tmp/playground_pu_query_20260731/report.md
git commit -m "docs: visualize playground PU query diagnosis"
```

