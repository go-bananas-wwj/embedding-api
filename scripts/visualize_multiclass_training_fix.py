"""Build a local visual review for traditional and DINOv3 multi-class training."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform

from app.routers.models import _train_class_heads
from app.schemas.models import GeoJSONFeature, GeoJSONFeatureCollection, ModelClass
from app.services.inference_engine import InferenceEngine
from app.services.model_registry import get_model_registry
from app.services.s2_ml import resolve_s2_path


ROOT = Path(__file__).resolve().parents[1]
MONTH = "202604"
TRAIN_PATCH = "patch_000018"
TARGET_PATCHES = ["patch_000018", "patch_000019", "patch_000020", "patch_000021"]
CLASSES = [
    ModelClass(id="building", name="建筑物", color="#DA9A20"),
    ModelClass(id="road", name="道路", color="#20BEDA"),
]


def _load_label(task: str, patch_id: str) -> np.ndarray:
    return np.load(ROOT / f"data/haidian/tasks/{task}/v1/labels/{patch_id}.npy") > 0


def _largest_polygons(mask: np.ndarray, source_path: Path, limit: int = 2):
    with rasterio.open(source_path) as src:
        transform = src.transform
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
    candidates = []
    for geometry, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        if not value:
            continue
        native = shape(geometry)
        candidates.append((native.area, native))
    candidates.sort(reverse=True, key=lambda item: item[0])
    return [
        mapping(shapely_transform(transformer.transform, geometry))
        for _, geometry in candidates[:limit]
    ]


def _annotations() -> GeoJSONFeatureCollection:
    source = resolve_s2_path("haidian", TRAIN_PATCH, MONTH)
    specs = [
        ("building_extraction", CLASSES[0]),
        ("road_extraction", CLASSES[1]),
    ]
    features = []
    for task, class_def in specs:
        label = _load_label(task, TRAIN_PATCH)
        for geometry in _largest_polygons(label, source):
            features.append(
                GeoJSONFeature.model_validate(
                    {
                        "type": "Feature",
                        "properties": {
                            "patch_id": TRAIN_PATCH,
                            "region_id": "haidian",
                            "class_id": class_def.id,
                            "class_name": class_def.name,
                            "color": class_def.color,
                            "month": MONTH,
                        },
                        "geometry": geometry,
                    }
                )
            )
    return GeoJSONFeatureCollection(type="FeatureCollection", features=features)


def _rgb(patch_id: str) -> np.ndarray:
    path = resolve_s2_path("haidian", patch_id, MONTH)
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        names = {str(name).upper(): i for i, name in enumerate(src.descriptions) if name}
        rgb = np.stack([data[names["B04"]], data[names["B03"]], data[names["B02"]]], axis=-1)
    valid = np.isfinite(rgb) & (rgb > 0)
    out = np.zeros_like(rgb)
    for channel in range(3):
        values = rgb[..., channel][valid[..., channel]]
        if values.size:
            low, high = np.percentile(values, [2, 98])
            out[..., channel] = np.clip((rgb[..., channel] - low) / max(high - low, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def _annotation_preview(rgb: np.ndarray) -> np.ndarray:
    result = rgb.copy()
    building = _load_label("building_extraction", TRAIN_PATCH)
    road = _load_label("road_extraction", TRAIN_PATCH)
    selected = np.zeros_like(building)
    source = resolve_s2_path("haidian", TRAIN_PATCH, MONTH)
    for task in ("building_extraction", "road_extraction"):
        mask = _load_label(task, TRAIN_PATCH)
        with rasterio.open(source) as src:
            for geometry in _largest_polygons(mask, source):
                from app.services.geojson_adapter import rasterize_geometry_to_grid

                selected |= rasterize_geometry_to_grid(
                    geometry,
                    crs=src.crs,
                    transform=src.transform,
                    height=src.height,
                    width=src.width,
                ).astype(bool)
    color = np.zeros_like(result)
    color[building & selected] = [218, 154, 32]
    color[road & selected] = [32, 190, 218]
    result[selected] = (0.35 * result[selected] + 0.65 * color[selected]).astype(np.uint8)
    return result


def _train(method: str, annotations: GeoJSONFeatureCollection) -> tuple[str, dict]:
    user_id = f"local_multiclass_review_{method}"
    registry = get_model_registry(user_id)
    model_id = registry.create_model(
        name=f"海淀双类别-{method}",
        model_type="single_time_detection",
        classes=[item.model_dump() for item in CLASSES],
        task_type="custom_detection",
        region_id="haidian",
        requested_training_method=method,
        feature_source="sentinel2_l2a" if method == "traditional_ml" else method,
    )
    result = _train_class_heads(
        model_id=model_id,
        user_id=user_id,
        region_id="haidian",
        task_type="custom_detection",
        model_type="single_time_detection",
        embedding_version="not_applicable",
        epochs=15,
        annotations=annotations,
        classes=CLASSES,
        class_ids=[item.id for item in CLASSES],
        training_method=method,
    )
    registry.update_model(model_id, status="completed")
    checkpoint = joblib.load(result["model_path"])
    return model_id, {"user_id": user_id, "result": result, "checkpoint": checkpoint}


def _save_overlay(rgb: np.ndarray, prediction: np.ndarray, path: Path) -> None:
    alpha = np.any(prediction != 0, axis=-1)
    overlay = rgb.copy()
    overlay[alpha] = (0.45 * rgb[alpha] + 0.55 * prediction[alpha]).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / f"Tmp/traditional_dinov3_multiclass_{timestamp}"
    output.mkdir(parents=True)
    annotations = _annotations()
    (output / "request_annotations.geojson").write_text(
        json.dumps(annotations.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trained = {
        method: _train(method, annotations)
        for method in ("traditional_ml", "dinov3_sat493m")
    }

    rows = []
    for patch_id in TARGET_PATCHES:
        rgb = _rgb(patch_id)
        rgb_name = f"{patch_id}_optical.png"
        Image.fromarray(rgb).save(output / rgb_name)
        cells = [f'<figure><img src="{rgb_name}"><figcaption>Sentinel-2 光学影像</figcaption></figure>']
        if patch_id == TRAIN_PATCH:
            annotation_name = f"{patch_id}_annotations.png"
            Image.fromarray(_annotation_preview(rgb)).save(output / annotation_name)
            cells.append(
                f'<figure><img src="{annotation_name}"><figcaption>模型实际看到的少量标注</figcaption></figure>'
            )
        else:
            cells.append(
                '<figure class="empty"><div>未提供标注</div><figcaption>跨 Patch 泛化测试</figcaption></figure>'
            )
        for method, label in (
            ("traditional_ml", "传统方法结果"),
            ("dinov3_sat493m", "DINOv3 结果"),
        ):
            model_id, info = trained[method]
            result_path = InferenceEngine(info["user_id"]).infer(
                model_id=model_id,
                region_id="haidian",
                patch_id=patch_id,
                month=MONTH,
            )
            prediction = np.array(Image.open(result_path).convert("RGB"))
            name = f"{patch_id}_{method}.png"
            _save_overlay(rgb, prediction, output / name)
            cells.append(f'<figure><img src="{name}"><figcaption>{label}</figcaption></figure>')
        rows.append(f'<section><h2>{patch_id}</h2><div class="grid">{"".join(cells)}</div></section>')

    head_rows = []
    report = {}
    for method, (model_id, info) in trained.items():
        heads = info["checkpoint"]["class_heads"]
        report[method] = {
            "model_id": model_id,
            "class_heads": heads,
            "model_path": info["result"]["model_path"],
        }
        head_rows.append(
            f"<tr><td>{html.escape(method)}</td><td>{model_id}</td>"
            f"<td>{len(heads)}</td><td>{'、'.join(item['class_name'] for item in heads)}</td></tr>"
        )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>传统方法与 DINOv3 多类别训练检查</title>
<style>
body{{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;margin:0;background:#f4f6f8;color:#17212b}}
main{{max-width:1440px;margin:auto;padding:24px}}h1{{font-size:28px;margin:0 0 8px}}p{{color:#53606d}}
.legend span{{display:inline-flex;align-items:center;margin-right:20px}}.dot{{width:14px;height:14px;margin-right:6px}}
table{{border-collapse:collapse;width:100%;background:white;margin:18px 0}}th,td{{padding:10px;border-bottom:1px solid #dce2e8;text-align:left}}
section{{background:white;margin:14px 0;padding:14px;border-left:4px solid #3578b8}}h2{{font-size:18px;margin:0 0 10px}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}figure{{margin:0}}img,.empty div{{width:100%;aspect-ratio:1;object-fit:cover;background:#e8edf1}}
figcaption{{font-weight:600;text-align:center;padding:7px}}.empty div{{display:grid;place-items:center;color:#7b8792}}
@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<h1>自定义训练多类别修复：本地效果检查</h1>
<p>训练 Patch：{TRAIN_PATCH}，月份：2026 年 4 月。每种方法都在同一个模型中训练“建筑物”和“道路”两个独立分类头。</p>
<div class="legend"><span><i class="dot" style="background:#DA9A20"></i>建筑物</span><span><i class="dot" style="background:#20BEDA"></i>道路</span></div>
<table><thead><tr><th>训练方式</th><th>本地模型 ID</th><th>分类头数量</th><th>包含类别</th></tr></thead><tbody>{''.join(head_rows)}</tbody></table>
{''.join(rows)}
</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
