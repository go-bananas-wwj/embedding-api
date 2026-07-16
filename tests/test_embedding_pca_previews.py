import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "regenerate_embedding_pca_previews.py"
SPEC = importlib.util.spec_from_file_location("regenerate_embedding_pca_previews", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_shared_pca_maps_same_embedding_value_to_same_color(tmp_path):
    month = tmp_path / "202601"
    month.mkdir()
    base = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    first = month / "patch_000000.npy"
    second = month / "patch_000001.npy"
    np.save(first, base)
    np.save(second, base)

    pca, low, high = MODULE.fit_shared_transform([first, second], 128, 42)
    first_rgb = np.asarray(MODULE.render(first, pca, low, high))
    second_rgb = np.asarray(MODULE.render(second, pca, low, high))

    np.testing.assert_array_equal(first_rgb, second_rgb)


def test_feather_month_tiles_reduces_boundary_jump(tmp_path):
    left = np.zeros((8, 8, 3), dtype=np.uint8)
    right = np.full((8, 8, 3), 255, dtype=np.uint8)
    Image.fromarray(left).save(tmp_path / "patch_000000.png")
    Image.fromarray(right).save(tmp_path / "patch_000001.png")
    patches = [
        {"patch_id": "patch_000000", "bounds": [0, 0, 10, 10]},
        {"patch_id": "patch_000001", "bounds": [10, 0, 20, 10]},
    ]

    count = MODULE.feather_month_tiles(tmp_path, patches, 2)
    left_after = np.asarray(Image.open(tmp_path / "patch_000000.png"))
    right_after = np.asarray(Image.open(tmp_path / "patch_000001.png"))

    assert count == 1
    assert np.abs(left_after[:, -1].astype(int) - right_after[:, 0].astype(int)).max() < 255
