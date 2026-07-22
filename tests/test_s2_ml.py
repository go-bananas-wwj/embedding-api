from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.services import s2_ml


def _write_s2(path: Path, value: float, descriptions=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=6,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 1, 0.1, 0.1),
    ) as dst:
        dst.write(np.full((6, 8, 8), value, dtype=np.float32))
        if descriptions:
            dst.descriptions = descriptions


def test_resolve_s2_path_uses_latest_scene_in_month(tmp_path, monkeypatch):
    root = tmp_path / "s2"
    older = root / "patch_000001" / "20260302.tif"
    latest = root / "patch_000001" / "20260327.tif"
    _write_s2(older, 0.1)
    _write_s2(latest, 0.2)

    class Config:
        def get_region(self, region_id):
            return {"s2_dir": str(root)}

    monkeypatch.setattr(s2_ml, "get_config", lambda: Config())
    assert s2_ml.resolve_s2_path("haidian", "patch_000001", "2026-03") == latest


def test_load_s2_features_builds_bands_and_indices(tmp_path, monkeypatch):
    scene = tmp_path / "s2" / "patch_000001" / "20260327.tif"
    _write_s2(scene, 0.2, s2_ml.CANONICAL_BANDS)

    class Config:
        def get_region(self, region_id):
            return {"s2_dir": str(tmp_path / "s2")}

    monkeypatch.setattr(s2_ml, "get_config", lambda: Config())
    features, valid, path = s2_ml.load_s2_features(
        "haidian", "patch_000001", "2026-03"
    )
    assert features.shape == (10, 8, 8)
    assert valid.all()
    assert path == str(scene)


def test_random_forest_trains_with_positive_and_unlabeled_pixels():
    feature = np.zeros((10, 16, 16), dtype=np.float32)
    feature[:, :8] = 1.0
    positive = np.zeros((16, 16), dtype=bool)
    positive[:8] = True
    valid = np.ones((16, 16), dtype=bool)
    model, threshold, metrics = s2_ml.train_random_forest(
        [(feature, positive, valid)]
    )
    assert model.n_features_in_ == 10
    assert 0.2 <= threshold <= 0.8
    assert metrics["training_f1"] > 0.9
