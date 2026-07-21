#!/usr/bin/env python3
"""预计算 OlmoEarth-Large 在海淀区数据上的 per-patch tokens 与 project_aggregated.

输入: /workspace/projects/olmo/data/haidian/olmoearth/  (已对齐 OlmoEarth band sets)
输出: /workspace/projects/olmo/embeddings_cache/olmoearth/{patch_id}/
      - tokens_and_masks/{modality}/tokens.pt
      - tokens_and_masks/{modality}/mask.pt
      - project_aggregated/all.pt
      - timestamps_ms.json
      - planet_img.pt

运行:
    conda activate olmoearth
    python scripts/preprocess/extract_olmoearth_embeddings.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling as RioResampling
from rasterio.errors import RasterioIOError
from tqdm import tqdm

sys.path.insert(0, "/workspace/projects/olmo/olmoearth_pretrain")
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/workspace/projects/olmo/data/haidian/olmoearth")
PLANET_ROOT = Path("/workspace/projects/olmo/data/haidian/planetscene")
OUT_ROOT = Path("/workspace/projects/olmo/embeddings_cache/olmoearth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 8

# 模态名 -> (MaskedOlmoEarthSample 字段名, mask 字段名, 总通道数, band set 数, 是否多时相)
MODALITY_CONFIG = {
    "s2": {
        "field": "sentinel2_l2a",
        "mask_field": "sentinel2_l2a_mask",
        "channels": 12,
        "n_band_sets": 3,
        "is_temporal": True,
    },
    "s1": {
        "field": "sentinel1",
        "mask_field": "sentinel1_mask",
        "channels": 2,
        "n_band_sets": 1,
        "is_temporal": True,
    },
    "landsat": {
        "field": "landsat",
        "mask_field": "landsat_mask",
        "channels": 11,
        "n_band_sets": 2,
        "is_temporal": True,
    },
    "dem": {
        "field": "srtm",
        "mask_field": "srtm_mask",
        "channels": 1,
        "n_band_sets": 1,
        "is_temporal": False,
    },
    "worldcover": {
        "field": "worldcover",
        "mask_field": "worldcover_mask",
        "channels": 1,
        "n_band_sets": 1,
        "is_temporal": False,
    },
}


def label_to_timestamp_ms(label: str) -> int:
    """YYYYMMDD -> ms."""
    label = str(label).strip().replace(".TIF", "").replace(".TIFF", "").replace(".tif", "").replace(".tiff", "")
    if len(label) == 8 and label.isdigit():
        dt = datetime(int(label[:4]), int(label[4:6]), int(label[6:8]))
        return int(dt.timestamp() * 1000)
    raise ValueError(label)


def is_in_training_window(stem: str) -> bool:
    """训练窗口 2025-12-01 ~ 2026-05-31."""
    stem = str(stem).strip().replace(".TIF", "").replace(".TIFF", "").replace(".tif", "").replace(".tiff", "")
    if len(stem) != 8 or not stem.isdigit():
        return False
    dt = datetime(int(stem[:4]), int(stem[4:6]), int(stem[6:8]))
    return datetime(2025, 12, 1) <= dt <= datetime(2026, 5, 31, 23, 59, 59)


def ms_to_olmo_timestamp(ts_ms: int) -> tuple[int, int, int]:
    dt = datetime.fromtimestamp(ts_ms / 1000.0)
    return (dt.day, dt.month - 1, dt.year)


def read_modality_tif(path: Path, expected_channels: int) -> np.ndarray | None:
    """读取模态 TIFF 并 resize/pad 到 128×128，返回 (C, 128, 128) float32."""
    try:
        with rasterio.open(path) as src:
            if src.width == 128 and src.height == 128:
                data = src.read().astype(np.float32)
            else:
                data = src.read(out_shape=(src.count, 128, 128), resampling=RioResampling.bilinear).astype(np.float32)
    except (RasterioIOError, OSError):
        return None

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if data.shape[0] < expected_channels:
        pad = np.zeros((expected_channels - data.shape[0], 128, 128), dtype=np.float32)
        data = np.concatenate([data, pad], axis=0)
    elif data.shape[0] > expected_channels:
        data = data[:expected_channels]

    return data


def load_modality_for_patch(patch_dir: Path, modality: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """加载某个 patch 的某个模态所有训练窗口内时间步.

    Returns:
        data:  (T, C, 128, 128) float32
        mask:  (T,) int64  (全 ONLINE_ENCODER)
        ts_ms: list[int]
    """
    cfg = MODALITY_CONFIG[modality]
    mod_dir = patch_dir / modality
    if not mod_dir.exists():
        return np.zeros((0, cfg["channels"], 128, 128), dtype=np.float32), np.array([], dtype=np.int64), []

    tif_paths = sorted(p for p in mod_dir.glob("*.tif*") if is_in_training_window(p.stem))
    if not tif_paths:
        return np.zeros((0, cfg["channels"], 128, 128), dtype=np.float32), np.array([], dtype=np.int64), []

    data_list, ts_list = [], []
    for p in tif_paths:
        arr = read_modality_tif(p, cfg["channels"])
        if arr is None:
            continue
        data_list.append(arr)
        ts_list.append(label_to_timestamp_ms(p.stem))

    if not data_list:
        return np.zeros((0, cfg["channels"], 128, 128), dtype=np.float32), np.array([], dtype=np.int64), []

    data = np.stack(data_list, axis=0)  # (T, C, H, W)
    mask = np.full((data.shape[0],), MaskValue.ONLINE_ENCODER.value, dtype=np.int64)
    return data, mask, ts_list


def align_modality_to_union(
    data: np.ndarray,
    mask: np.ndarray,
    ts_ms: list[int],
    all_ts_ms: list[int],
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """将某个模态的数据和 mask 对齐到 all_ts_ms 时间轴.

    Args:
        data: (T, C, H, W)
        mask: (T,)
        ts_ms: 该模态实际有的时间戳
        all_ts_ms: 统一时间轴
        cfg: MODALITY_CONFIG[mod]

    Returns:
        data_arr: (1, H, W, T_union, C)
        mask_arr: (1, H, W, T_union, Band_Sets)
    """
    BS = cfg["n_band_sets"]
    T_union = len(all_ts_ms)
    C, H, W = cfg["channels"], 128, 128

    if data.shape[0] == 0:
        # 缺失模态：全 MISSING
        data_arr = np.zeros((1, H, W, T_union, C), dtype=np.float32)
        mask_arr = np.full((1, H, W, T_union, BS), MaskValue.MISSING.value, dtype=np.int64)
        return data_arr, mask_arr

    # 建立时间戳到索引的映射
    ts_to_idx = {t: i for i, t in enumerate(ts_ms)}

    aligned_data = np.zeros((T_union, C, H, W), dtype=np.float32)
    aligned_mask = np.full((T_union,), MaskValue.MISSING.value, dtype=np.int64)

    for out_t, t in enumerate(all_ts_ms):
        if t in ts_to_idx:
            src_t = ts_to_idx[t]
            aligned_data[out_t] = data[src_t]
            aligned_mask[out_t] = mask[src_t]

    # data: (T_union, C, H, W) -> (1, H, W, T_union, C)
    data_arr = np.transpose(aligned_data, (2, 3, 0, 1))[np.newaxis, ...]
    # mask: (T_union,) -> (1, H, W, T_union, BS)
    mask_arr = np.broadcast_to(
        aligned_mask.reshape(1, 1, 1, T_union, 1),
        (1, H, W, T_union, BS),
    ).copy()
    return data_arr, mask_arr


def build_sample_for_patch(patch_dir: Path) -> tuple[MaskedOlmoEarthSample, list[int], np.ndarray | None]:
    """为单个 patch 构造 MaskedOlmoEarthSample.

    OlmoEarth 要求所有模态共享同一 T，因此先把各模态对齐到统一时间轴，
    缺失时间步填 0 并标 MISSING。

    返回:
        sample: MaskedOlmoEarthSample
        all_ts_ms: 所有模态时间戳并集排序
        planet_img: (4, 128, 128) 或 None
    """
    # 先加载所有模态，收集时间戳并集
    loaded = {}
    all_ts_set = set()
    for mod, cfg in MODALITY_CONFIG.items():
        data, mask, ts_ms = load_modality_for_patch(patch_dir, mod)
        loaded[mod] = (data, mask, ts_ms)
        all_ts_set.update(ts_ms)

    all_ts_ms = sorted(all_ts_set)
    if not all_ts_ms:
        all_ts_ms = [0]
    # OlmoEarth-Large 预训练时 max_sequence_length=12，pos_embed 长度固定为 12
    if len(all_ts_ms) > 12:
        all_ts_ms = all_ts_ms[-12:]
    T_union = len(all_ts_ms)

    sample_kwargs = {}
    for mod, cfg in MODALITY_CONFIG.items():
        data, mask, ts_ms = loaded[mod]
        data_arr, mask_arr = align_modality_to_union(data, mask, ts_ms, all_ts_ms, cfg)
        sample_kwargs[cfg["field"]] = data_arr
        sample_kwargs[cfg["mask_field"]] = mask_arr

    timestamps = np.array([[ms_to_olmo_timestamp(t) for t in all_ts_ms]], dtype=np.int64)
    sample_kwargs["timestamps"] = timestamps

    sample = MaskedOlmoEarthSample(**sample_kwargs)

    # 读取 PlanetScene（训练窗口内第一个可用文件）
    planet_img = None
    planet_dir = PLANET_ROOT / patch_dir.name
    if planet_dir.exists():
        for p in sorted(planet_dir.glob("*.tif*")):
            if not is_in_training_window(p.stem):
                continue
            try:
                with rasterio.open(p) as src:
                    arr = src.read(out_shape=(4, 128, 128), resampling=RioResampling.bilinear).astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0)
                planet_img = arr
                break
            except (RasterioIOError, OSError):
                continue

    return sample, all_ts_ms, planet_img


def extract_patch(model, patch_dir: Path, out_dir: Path):
    """提取单个 patch 的 OlmoEarth 嵌入并保存."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir = out_dir / "tokens_and_masks"
    proj_dir = out_dir / "project_aggregated"
    tokens_dir.mkdir(exist_ok=True)
    proj_dir.mkdir(exist_ok=True)

    sample, all_ts_ms, planet_img = build_sample_for_patch(patch_dir)

    (out_dir / "timestamps_ms.json").write_text(json.dumps(all_ts_ms))

    if planet_img is not None:
        torch.save(torch.from_numpy(planet_img), out_dir / "planet_img.pt")

    # 手动将所有 ndarray / tensor 移到目标设备（to_device 对 numpy timestamps 处理有 bug）
    sample_dict = sample.as_dict(include_nones=False)
    sample_dict = {
        k: (torch.from_numpy(v).to(DEVICE) if isinstance(v, np.ndarray) else v.to(DEVICE))
        for k, v in sample_dict.items()
    }
    sample = MaskedOlmoEarthSample(**sample_dict)

    model.eval()
    with torch.no_grad():
        # LatentMIM 外层 forward 不接受 fast_pass；直接调用 encoder
        output = model.encoder(sample, patch_size=PATCH_SIZE, fast_pass=False)

    tokens_and_masks = output["tokens_and_masks"]
    project_aggregated = output["project_aggregated"]

    torch.save(project_aggregated.cpu(), proj_dir / "all.pt")

    for mod, cfg in MODALITY_CONFIG.items():
        field = cfg["field"]
        mod_tokens_dir = tokens_dir / field
        mod_tokens_dir.mkdir(exist_ok=True)
        tokens = getattr(tokens_and_masks, field)
        masks = getattr(tokens_and_masks, field + "_mask")
        if tokens is not None:
            torch.save(tokens.cpu(), mod_tokens_dir / "tokens.pt")
        if masks is not None:
            torch.save(masks.cpu(), mod_tokens_dir / "mask.pt")


def main():
    print(f"Device: {DEVICE}")
    print("Loading OlmoEarth-Large...")
    model = load_model_from_id(ModelID.OLMOEARTH_V1_LARGE)
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded.")

    patch_dirs = sorted(p for p in DATA_ROOT.glob("patch_*") if p.is_dir())
    print(f"Found {len(patch_dirs)} patches.")

    for patch_dir in tqdm(patch_dirs, desc="Extracting"):
        out_dir = OUT_ROOT / patch_dir.name
        if (out_dir / "project_aggregated" / "all.pt").exists():
            continue
        try:
            extract_patch(model, patch_dir, out_dir)
        except Exception as e:
            print(f"Error processing {patch_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    main()
