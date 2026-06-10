#!/usr/bin/env python3
"""320 Patch 批量嵌入生产脚本.

基于已有 OlmoEarth 预计算 token 缓存直接切片，运行 Student 模型推理。
修复了所有关键 bug（不复跑 OlmoEarth、用缓存 PlanetScene、正确构造输入等）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, "/workspace/olmo")
sys.path.insert(0, "/workspace/olmo/olmoearth_pretrain")

from src.data.haidian_dataset import distill_collate_fn
from src.models.distill_decoder import OlmoEarthToAEFDistillationDecoder

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

EMB_CACHE_ROOT = Path("/workspace/olmo/embeddings_cache/olmoearth")
CHECKPOINT_PATH = Path("/workspace/olmo/checkpoints/checkpoint_best.pt")
CFG_PATH = Path("/workspace/olmo/configs/distill_large_spatial.yaml")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_student_model(checkpoint_path: str, cfg_path: str):
    """加载 Student 模型，严格匹配训练时的配置."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = OlmoEarthToAEFDistillationDecoder(**cfg["model"]).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def generate_target_dates(start_str: str, end_str: str, interval_days: int) -> list[str]:
    """生成目标日期列表 (YYYYMMDD)."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=interval_days)
    return dates


def find_nearest_timestamp_idx(target_ms: int, ts_ms_list: list[int]) -> int:
    """找到最接近目标时间戳的索引."""
    return min(range(len(ts_ms_list)), key=lambda i: abs(ts_ms_list[i] - target_ms))


def target_date_to_ms(date_str: str) -> int:
    """YYYYMMDD -> ms."""
    dt = datetime.strptime(date_str, "%Y%m%d")
    return int(dt.timestamp() * 1000)


def load_patch_cache(patch_id: str):
    """加载 patch 的预计算 token 缓存.

    Returns:
        tokens_dict: dict[str, Tensor]  各模态全部时间点的 tokens
        masks_dict:  dict[str, Tensor]  各模态全部时间点的 masks
        planet_img:  Tensor (4, 128, 128)
        ts_ms_list:  list[int]  timestamps_ms.json 内容
    """
    cache_dir = EMB_CACHE_ROOT / patch_id

    tokens_dict = {}
    masks_dict = {}
    for mod_key, mod_name in [
        ("sentinel2_l2a", "sentinel2_l2a"),
        ("sentinel1", "sentinel1"),
        ("landsat", "landsat"),
        ("srtm", "srtm"),
        ("worldcover", "worldcover"),
    ]:
        tokens_path = cache_dir / "tokens_and_masks" / mod_name / "tokens.pt"
        masks_path = cache_dir / "tokens_and_masks" / mod_name / "mask.pt"
        tokens = torch.load(tokens_path, map_location="cpu", weights_only=False)
        masks = torch.load(masks_path, map_location="cpu", weights_only=False)
        tokens_dict[mod_key] = tokens
        masks_dict[mod_key + "_mask"] = masks

    planet_img = torch.load(cache_dir / "planet_img.pt", map_location="cpu", weights_only=False)
    project_agg = torch.load(cache_dir / "project_aggregated" / "all.pt", map_location="cpu", weights_only=False)[0]  # (1024,)

    with open(cache_dir / "timestamps_ms.json") as f:
        ts_ms_list = json.load(f)

    return tokens_dict, masks_dict, planet_img, project_agg, ts_ms_list


def build_sample_for_timestep(
    tokens_dict: dict,
    masks_dict: dict,
    planet_img: torch.Tensor,
    project_agg: torch.Tensor,
    ts_ms_list: list[int],
    t_idx: int,
    patch_id: str,
) -> dict:
    """构造单个时间点的 sample，与 HaidianDistillationDataset.__getitem__ 输出一致.

    Returns:
        sample: dict 可直接喂给 distill_collate_fn
    """
    # 1. Slice tokens & masks at t_idx (保留 T=1 维度，与训练代码一致)
    # tokens shape: (B, H_p, W_p, T, BS, C) -> slice [0, :, :, t_idx:t_idx+1, :, :] -> (H_p, W_p, 1, BS, C)
    # masks shape: (B, H_p, W_p, T, BS)    -> slice [0, :, :, t_idx:t_idx+1, :]    -> (H_p, W_p, 1, BS)
    olmo_tokens_dict = {}
    olmo_masks_dict = {}
    for mod_key in ["sentinel2_l2a", "sentinel1", "landsat", "srtm", "worldcover"]:
        tokens = tokens_dict[mod_key][0, :, :, t_idx:t_idx + 1, :, :]  # (H_p, W_p, 1, BS, C)
        masks = masks_dict[mod_key + "_mask"][0, :, :, t_idx:t_idx + 1, :]  # (H_p, W_p, 1, BS)
        olmo_tokens_dict[mod_key] = tokens
        olmo_masks_dict[mod_key + "_mask"] = masks

    # 2. PlanetScene: 直接使用缓存的 planet_img
    # 训练时 planet_valid 是 True（因为 planet_img 存在）
    planet_valid = True

    # 3. AEF target（如果有的话，推理时不需要但 collate_fn 可能需要）
    # 实际上 distill_collate_fn 不强制要求 aef_emb，但 evaluate_spatial.py 中用了
    # 我们这里构造一个 dummy aef_emb 来兼容 collate_fn
    aef_emb = torch.zeros(64, 128, 128)

    # 4. 构造 sample dict (字段名必须与 distill_collate_fn 一致)
    sample = {
        "olmo_tokens_dict": olmo_tokens_dict,
        "olmo_masks_dict": olmo_masks_dict,
        "project_agg": project_agg,
        "planet_img": planet_img,
        "planet_valid": planet_valid,
        "aef_emb": aef_emb,
        "jrc_valid_mask": torch.ones(128, 128, dtype=torch.bool),  # dummy
        "patch_id": patch_id,
        "ts_ms": [ts_ms_list[t_idx]],
    }

    return sample


def _move_to_device(obj, device: torch.device):
    """递归移动 tensor 到指定设备."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    return obj


@torch.no_grad()
def infer_single(model, sample, device):
    """运行 Student 模型推理单个 sample."""
    batch = distill_collate_fn([sample])

    # 递归移动到 GPU（处理 nested dict）
    batch = _move_to_device(batch, device)

    model_out = model(
        batch["olmo_tokens_dict"],
        batch["olmo_masks_dict"],
        planet_img=batch["planet_img"],
        planet_valid=batch["planet_valid"],
    )

    emb_map = model_out[0]  # (1, 64, 128, 128)
    return emb_map[0].cpu().float().numpy()  # (64, 128, 128)


def produce_embeddings(
    patch_ids: list[str],
    target_dates: list[str],
    output_dir: Path,
    model,
    device,
):
    """主生产函数."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 预计算所有目标日期的 ms
    target_ms_list = [target_date_to_ms(d) for d in target_dates]

    total = len(patch_ids) * len(target_dates)
    pbar = tqdm(total=total, desc="Producing embeddings")

    for patch_id in patch_ids:
        patch_out_dir = output_dir / patch_id
        patch_out_dir.mkdir(exist_ok=True)

        # 加载该 patch 的缓存（只加载一次）
        try:
            tokens_dict, masks_dict, planet_img, project_agg, ts_ms_list = load_patch_cache(patch_id)
        except Exception as e:
            print(f"[SKIP] {patch_id}: failed to load cache: {e}")
            pbar.update(len(target_dates))
            continue

        # 转换 planet_img 为 tensor 并移到目标设备
        planet_img = planet_img.to(device)

        for target_date, target_ms in zip(target_dates, target_ms_list):
            out_file = patch_out_dir / f"{patch_id}_{target_date}.npz"
            if out_file.exists():
                pbar.update(1)
                continue

            # 找最近的时间戳索引
            t_idx = find_nearest_timestamp_idx(target_ms, ts_ms_list)
            nearest_ms = ts_ms_list[t_idx]
            gap_days = abs(nearest_ms - target_ms) // (1000 * 86400)

            # 构造 sample
            sample = build_sample_for_timestep(
                tokens_dict, masks_dict, planet_img, project_agg, ts_ms_list, t_idx, patch_id
            )

            # 推理
            try:
                embedding = infer_single(model, sample, device)
            except Exception as e:
                print(f"[SKIP] {patch_id} {target_date}: inference failed: {e}")
                pbar.update(1)
                continue

            # 保存
            np.savez(
                out_file,
                embedding=embedding.astype(np.float32),
                target_date=target_date,
                target_timestamp_ms=target_ms,
                source_timestamp_ms=nearest_ms,
                source_t_idx=t_idx,
                gap_days=int(gap_days),
                patch_id=patch_id,
            )
            pbar.update(1)

    pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--config", default=str(CFG_PATH))
    parser.add_argument("--output_dir", default="/workspace/olmo/outputs/batch_production")
    parser.add_argument("--start_date", default="2025-12-01")
    parser.add_argument("--end_date", default="2026-05-01")
    parser.add_argument("--interval_days", type=int, default=11)
    parser.add_argument("--patches", nargs="+", default=None,
                        help="指定 patch 列表，默认全部 320 个")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print("Loading Student model...")
    model, cfg = load_student_model(args.checkpoint, args.config)
    print("Model loaded.")

    target_dates = generate_target_dates(
        args.start_date, args.end_date, args.interval_days
    )
    print(f"Target dates ({len(target_dates)}): {target_dates}")

    if args.patches:
        patch_ids = sorted(args.patches)
    else:
        patch_ids = sorted([p.name for p in EMB_CACHE_ROOT.glob("patch_*") if p.is_dir()])
    print(f"Patches: {len(patch_ids)}")

    output_dir = Path(args.output_dir)

    t0 = time.time()
    produce_embeddings(patch_ids, target_dates, output_dir, model, DEVICE)
    elapsed = time.time() - t0
    print(f"Done. Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # 打印产出统计
    total_expected = len(patch_ids) * len(target_dates)
    total_actual = sum(1 for _ in output_dir.rglob("*.npz"))
    print(f"Expected: {total_expected}, Actual: {total_actual}")


if __name__ == "__main__":
    main()
