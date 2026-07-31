"""Tests for the focused optical-texture playground gallery."""

import json
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts.build_playground_xuannv_gallery import build_gallery


PATCH_IDS = [
    "patch_000059",
    "patch_000060",
    "patch_000064",
    "patch_000076",
    "patch_000232",
    "patch_000249",
    "patch_000154",
    "patch_000139",
]


def _write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_optical(path: Path, offset: int):
    values = np.arange(3 * 24 * 24, dtype=np.uint16).reshape(3, 24, 24)
    values = values + offset
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=24,
        height=24,
        count=3,
        dtype=values.dtype,
        transform=from_origin(0, 24, 1, 1),
    ) as dataset:
        dataset.write(values)


def _synthetic_input(tmp_path: Path) -> Path:
    input_dir = tmp_path / "Tmp/playground_texture_20260731"
    arrays_dir = input_dir / "arrays"
    optical_dir = tmp_path / "optical"
    arrays_dir.mkdir(parents=True)
    optical_dir.mkdir()
    artifacts = {}
    per_patch = {}
    roles = (
        ["training"] * 3
        + ["independent_osm"]
        + ["global_high_false_positive"] * 3
        + ["spatial_high_score"]
    )
    for index, (patch_id, role) in enumerate(zip(PATCH_IDS, roles)):
        score = np.linspace(-0.6, 1.3, 64, dtype=np.float32).reshape(8, 8)
        boundary = np.zeros((8, 8), dtype=np.float32)
        boundary[:, 4] = 1.0
        reference = np.zeros((8, 8), dtype=np.uint8)
        if role in {"training", "independent_osm"}:
            reference[2:6, 2:6] = 1
        masks = {
            "baseline": score >= 0.25,
            "strict": score >= 1.0,
            "guarded": score >= 1.15,
            "area_guard": np.logical_and(score >= 0.55, score < 1.2),
            "texture_boundary_area_guard": np.logical_and(
                np.logical_and(score >= 0.55, score < 1.2),
                boundary < 0.5,
            ),
        }
        arrays = {}
        values = {
            "score_query": score,
            "texture_boundary": boundary,
            "reference": reference,
            **{name: mask.astype(np.uint8) for name, mask in masks.items()},
        }
        for name, value in values.items():
            relative = Path("arrays") / f"{patch_id}_{name}.npy"
            np.save(input_dir / relative, value, allow_pickle=False)
            arrays[name] = relative.as_posix()
        optical_path = optical_dir / f"{patch_id}.tif"
        _write_optical(optical_path, index * 10)
        artifacts[patch_id] = {
            "role": role,
            "arrays": arrays,
            "optical_path": str(optical_path.relative_to(tmp_path)),
        }
        per_patch[patch_id] = {
            "role": role,
            "variants": {
                name: {
                    "positive_ratio": float(mask.mean()),
                    "positive_pixels": int(mask.sum()),
                    "component_count": 1,
                }
                for name, mask in masks.items()
            },
        }
    selection = {
        "training": PATCH_IDS[:3],
        "independent_osm": [PATCH_IDS[3]],
        "global_high_false_positive": PATCH_IDS[4:7],
        "spatial_high_score": [PATCH_IDS[7]],
        "selection_evidence": [
            {
                "patch_id": patch_id,
                "center_wgs84": [116.0 + index, 40.0 - index],
                "baseline_positive_ratio": per_patch[patch_id]["variants"][
                    "baseline"
                ]["positive_ratio"],
                "reason": f"选择原因 {index}",
            }
            for index, patch_id in enumerate(PATCH_IDS)
        ],
    }
    manifest = {
        "experiment": {
            "month": "202604",
            "scope": "selected_typical_patches_only",
            "selected_patch_count": 8,
        },
        "model": {"id": "model_756ed870", "name": "playground_xuannv"},
        "reference_policy": {
            "limitation": "参考标签不完整，未标注区不是可靠负样本。",
            "texture_prior": "只使用光学纹理，没有人工类别先验。",
        },
        "selection": selection,
        "artifacts": {"arrays": artifacts},
    }
    base_metric = {
        "precision": 0.32,
        "recall": 0.53,
        "f1": 0.40,
        "iou": 0.25,
        "positive_ratio": 0.02,
        "component_count": 2,
    }
    metrics = {
        "selection": selection,
        "per_patch": per_patch,
        "reference_relative_metrics": {
            "training_polygons": {
                name: dict(base_metric)
                for name in (
                    "baseline",
                    "guarded",
                    "area_guard",
                    "texture_boundary_area_guard",
                )
            },
            "independent_osm_polygon": {
                name: dict(base_metric)
                for name in (
                    "baseline",
                    "guarded",
                    "area_guard",
                    "texture_boundary_area_guard",
                )
            },
        },
        "texture_assessment": {
            "materially_improved_over_area_guard": False,
            "improved_over_legacy_guarded": True,
            "independent_osm_f1_delta": 0.0016,
            "independent_osm_recall_delta": 0.0,
            "high_false_mean_area_ratio_delta": 0.0,
            "training_recall_delta": -0.0426,
            "verdict": "纹理边界未证明有实质增益",
            "interpretation": "主要改善来自 Patch 内相对种子阈值。",
        },
    }
    _write_json(input_dir / "experiment_manifest.json", manifest)
    _write_json(input_dir / "metrics.json", metrics)
    return input_dir


def test_gallery_builds_only_focused_patches_with_nine_views(tmp_path):
    input_dir = _synthetic_input(tmp_path)

    output = build_gallery(input_dir, repo_root=tmp_path)

    html = output.read_text(encoding="utf-8")
    assert "操场纹理边界实验" in html
    assert "只展示 8 个典型 Patch" in html
    assert "纹理边界未证明有实质增益" in html
    assert "相对种子 + 面积筛选" in html
    assert "纹理边界 + 面积筛选" in html
    assert "现有全局面积保护" in html
    assert "参考标签不完整" in html
    assert "没有人工类别先验" in html
    assert "点击放大" in html

    for patch_id in PATCH_IDS:
        row = re.search(
            rf'<section class="patch-section" data-patch-id="{patch_id}">(.*?)</section>',
            html,
            flags=re.DOTALL,
        )
        assert row, patch_id
        assert row.group(1).count("<figure") == 9

    assert html.count('class="patch-section"') == 8


def test_gallery_warns_about_suspected_training_mislabels(tmp_path):
    input_dir = _synthetic_input(tmp_path)

    output = build_gallery(input_dir, repo_root=tmp_path)

    html = output.read_text(encoding="utf-8")
    assert "P1 数据质量警告：原训练标注疑似错标，当前模型不可上线" in html
    assert "纹理与面积实验仅用于评估错误标注模型的风险抑制" in html
    assert "原训练标注（疑似错标）" in html

    expected_reviews = {
        "patch_000059": "标注主要落在院落和建筑区域，疑似不是操场",
        "patch_000060": "标注位于建筑旁的小块区域，疑似不是操场",
        "patch_000064": "标注落在左侧建成区；真实蓝绿操场位于影像下方",
    }
    for patch_id, review in expected_reviews.items():
        row = re.search(
            rf'<section class="patch-section" data-patch-id="{patch_id}">(.*?)</section>',
            html,
            flags=re.DOTALL,
        )
        assert row, patch_id
        assert "原训练标注（疑似错标）" in row.group(1)
        assert review in row.group(1)


def test_gallery_resources_and_metrics_are_consistent(tmp_path):
    input_dir = _synthetic_input(tmp_path)

    output = build_gallery(input_dir, repo_root=tmp_path)

    html = output.read_text(encoding="utf-8")
    image_sources = re.findall(r'<img[^>]+src="([^"]+)"', html)
    assert image_sources
    assert all(not source.startswith(("/", "file:")) for source in image_sources)
    assert all((input_dir / source).is_file() for source in image_sources)
    assert "+0.0016" in html
    assert "选择原因 0" in html
    assert (input_dir / "assets/selected-patch-area-comparison.png").is_file()
    assert (input_dir / "assets/reference-metric-comparison.png").is_file()
    assert 'id="image-dialog"' in html
    assert "showModal()" in html
