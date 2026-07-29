"""Compare two embedding-distance change maps for Haidian.

This is an offline visualization experiment. It does not modify API routes,
configuration, model weights, or generated task assets.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_ROOT = ROOT / "data/haidian/embeddings/v1"
OPTICAL_ROOT = (
    ROOT / "data/haidian/archive/processed_training_data/extracted/patches/s2"
)
BEFORE_MONTH = "202512"
AFTER_MONTH = "202604"
EPSILON = 1e-8
CMAP = matplotlib.colormaps["coolwarm"]


def smooth_embedding(
    embedding: np.ndarray,
    method: str,
    size: int = 3,
    sigma: float = 1.0,
) -> np.ndarray:
    """Smooth each embedding channel spatially without mixing channels."""
    values = np.asarray(embedding, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("Embedding must have C,H,W shape")
    if method == "mean":
        return ndimage.uniform_filter(
            values,
            size=(1, size, size),
            mode="nearest",
        ).astype(np.float32)
    if method == "gaussian":
        return ndimage.gaussian_filter(
            values,
            sigma=(0.0, sigma, sigma),
            mode="nearest",
        ).astype(np.float32)
    raise ValueError("method must be 'mean' or 'gaussian'")


def estimate_translation(
    before: np.ndarray,
    after: np.ndarray,
    max_shift: int = 16,
) -> tuple[float, float]:
    """Estimate the integer (row, column) shift that aligns after to before."""
    first = np.asarray(before, dtype=np.float32)
    second = np.asarray(after, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Images must have the same H,W shape")
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    first = np.nan_to_num(first - np.nanmean(first))
    second = np.nan_to_num(second - np.nanmean(second))
    cross = np.fft.fft2(first) * np.conj(np.fft.fft2(second))
    cross /= np.maximum(np.abs(cross), EPSILON)
    correlation = np.fft.fftshift(np.abs(np.fft.ifft2(cross)))
    center = np.array(correlation.shape) // 2
    row_slice = slice(max(0, center[0] - max_shift), center[0] + max_shift + 1)
    col_slice = slice(max(0, center[1] - max_shift), center[1] + max_shift + 1)
    window = correlation[row_slice, col_slice]
    peak = np.array(np.unravel_index(np.argmax(window), window.shape))
    origin = np.array([row_slice.start, col_slice.start])
    shift = peak + origin - center
    return float(shift[0]), float(shift[1])


def robust_temporal_normalize(
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove per-date channelwise offset and scale using median and IQR."""
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("Embeddings must have the same C,H,W shape")
    normalized = []
    for embedding in (before, after):
        values = np.asarray(embedding, dtype=np.float32)
        median = np.nanmedian(values, axis=(1, 2), keepdims=True)
        lower = np.nanpercentile(values, 25, axis=(1, 2), keepdims=True)
        upper = np.nanpercentile(values, 75, axis=(1, 2), keepdims=True)
        scale = np.maximum(upper - lower, EPSILON)
        normalized.append(((values - median) / scale).astype(np.float32))
    return normalized[0], normalized[1]


def change_scores(
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cosine and normalized-Euclidean change maps plus a valid mask."""
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("Embeddings must have the same C,H,W shape")
    before = np.moveaxis(np.asarray(before, dtype=np.float32), 0, -1)
    after = np.moveaxis(np.asarray(after, dtype=np.float32), 0, -1)
    before_norm = np.linalg.norm(before, axis=-1)
    after_norm = np.linalg.norm(after, axis=-1)
    valid = (
        np.isfinite(before).all(axis=-1)
        & np.isfinite(after).all(axis=-1)
        & (before_norm > EPSILON)
        & (after_norm > EPSILON)
    )

    before_unit = before / np.maximum(before_norm[..., None], EPSILON)
    after_unit = after / np.maximum(after_norm[..., None], EPSILON)
    cosine = 1.0 - np.sum(before_unit * after_unit, axis=-1)
    euclidean = np.linalg.norm(before_unit - after_unit, axis=-1)
    cosine = np.clip(cosine, 0.0, 2.0).astype(np.float32)
    euclidean = np.clip(euclidean, 0.0, 2.0).astype(np.float32)
    cosine[~valid] = np.nan
    euclidean[~valid] = np.nan
    return cosine, euclidean, valid


def neighborhood_cosine_change(
    before: np.ndarray,
    after: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match each before pixel to the most similar after pixel nearby."""
    return _directional_neighborhood_cosine_change(
        before,
        after,
        radius=radius,
        displacement_penalty=0.0,
    )


def _directional_neighborhood_cosine_change(
    before: np.ndarray,
    after: np.ndarray,
    radius: int,
    displacement_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    if radius < 1:
        raise ValueError("radius must be at least 1")
    if displacement_penalty < 0:
        raise ValueError("displacement_penalty must be non-negative")
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("Embeddings must have the same C,H,W shape")

    before_pixels = np.moveaxis(np.asarray(before, dtype=np.float32), 0, -1)
    after_pixels = np.moveaxis(np.asarray(after, dtype=np.float32), 0, -1)
    before_norm = np.linalg.norm(before_pixels, axis=-1)
    after_norm = np.linalg.norm(after_pixels, axis=-1)
    before_valid = np.isfinite(before_pixels).all(axis=-1) & (before_norm > EPSILON)
    after_valid = np.isfinite(after_pixels).all(axis=-1) & (after_norm > EPSILON)
    before_unit = before_pixels / np.maximum(before_norm[..., None], EPSILON)
    after_unit = after_pixels / np.maximum(after_norm[..., None], EPSILON)

    height, width = before_valid.shape
    padded_after = np.pad(
        after_unit,
        ((radius, radius), (radius, radius), (0, 0)),
        mode="constant",
    )
    padded_valid = np.pad(
        after_valid,
        ((radius, radius), (radius, radius)),
        mode="constant",
        constant_values=False,
    )
    best_cost = np.full((height, width), np.inf, dtype=np.float32)
    has_match = np.zeros((height, width), dtype=bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            candidate = padded_after[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
            candidate_valid = padded_valid[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
            pair_valid = before_valid & candidate_valid
            similarity = np.sum(before_unit * candidate, axis=-1)
            row_distance = row_offset - radius
            column_distance = column_offset - radius
            normalized_distance = (
                row_distance * row_distance + column_distance * column_distance
            ) / float(radius * radius)
            cost = 1.0 - similarity + displacement_penalty * normalized_distance
            best_cost[pair_valid] = np.minimum(
                best_cost[pair_valid],
                cost[pair_valid],
            )
            has_match |= pair_valid

    change = np.clip(best_cost, 0.0, 2.0).astype(np.float32)
    change[~has_match] = np.nan
    return change, has_match


def symmetric_neighborhood_cosine_change(
    before: np.ndarray,
    after: np.ndarray,
    radius: int = 2,
    displacement_penalty: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine both temporal matching directions with spatial regularization."""
    return bidirectional_neighborhood_cosine_change(
        before,
        after,
        radius=radius,
        displacement_penalty=displacement_penalty,
        fusion="max",
    )


def bidirectional_neighborhood_cosine_change(
    before: np.ndarray,
    after: np.ndarray,
    radius: int = 2,
    displacement_penalty: float = 0.05,
    fusion: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse forward and backward neighborhood matching costs."""
    forward, forward_valid = _directional_neighborhood_cosine_change(
        before,
        after,
        radius=radius,
        displacement_penalty=displacement_penalty,
    )
    backward, backward_valid = _directional_neighborhood_cosine_change(
        after,
        before,
        radius=radius,
        displacement_penalty=displacement_penalty,
    )
    valid = forward_valid & backward_valid
    maximum = np.maximum(forward, backward)
    minimum = np.minimum(forward, backward)
    if fusion == "max":
        change = maximum
    elif fusion == "mean":
        change = (forward + backward) * 0.5
    elif fusion == "q75":
        change = minimum * 0.25 + maximum * 0.75
    else:
        raise ValueError("fusion must be 'max', 'mean', or 'q75'")
    change[~valid] = np.nan
    return change.astype(np.float32), valid


def global_limits(
    score_maps: Iterable[np.ndarray],
    low_quantile: float = 0.02,
    high_quantile: float = 0.98,
) -> tuple[float, float]:
    """Return finite global quantile limits across score maps."""
    finite = [
        np.asarray(score)[np.isfinite(score)]
        for score in score_maps
        if np.isfinite(score).any()
    ]
    if not finite:
        raise ValueError("No finite change scores available")
    values = np.concatenate(finite)
    low, high = np.quantile(values, [low_quantile, high_quantile])
    if high <= low:
        high = low + 1e-6
    return float(low), float(high)


def select_representative_patches(
    stats: list[dict],
    count: int = 12,
) -> list[str]:
    """Select evenly spaced patches from the cosine-P95 ranking."""
    if not stats or count <= 0:
        return []
    ranked = sorted(stats, key=lambda item: (item["cosine_p95"], item["patch_id"]))
    indexes = np.linspace(0, len(ranked) - 1, min(count, len(ranked)))
    indexes = np.rint(indexes).astype(int)
    return [ranked[index]["patch_id"] for index in dict.fromkeys(indexes)]


def robust_rgb(path: Path) -> np.ndarray:
    """Read Sentinel-2 B04/B03/B02 and apply robust per-band stretching."""
    with rasterio.open(path) as dataset:
        descriptions = list(dataset.descriptions)
        try:
            indexes = [descriptions.index(name) + 1 for name in ("B04", "B03", "B02")]
        except ValueError as exc:
            raise ValueError(f"RGB bands are missing from {path}") from exc
        bands = dataset.read(indexes).astype(np.float32)

    rgb = np.moveaxis(bands, 0, -1)
    output = np.zeros_like(rgb, dtype=np.uint8)
    for channel in range(3):
        values = rgb[..., channel]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        low, high = np.percentile(values[finite], [2, 98])
        if high <= low:
            continue
        stretched = np.clip((values - low) / (high - low), 0.0, 1.0)
        stretched[~finite] = 0.0
        output[..., channel] = np.rint(stretched * 255).astype(np.uint8)
    return output


def _best_optical(month: str, patch_id: str) -> Path:
    """Select the clearest available scene in a month, then prefer latest."""
    candidates = sorted(
        path
        for path in OPTICAL_ROOT.glob(f"s2_{month}??_{patch_id}.tif")
        if "_mask" not in path.name
    )
    if not candidates:
        raise FileNotFoundError(f"No optical image for {month}/{patch_id}")
    ranked = []
    for path in candidates:
        with rasterio.open(path) as dataset:
            rgb = dataset.read([1, 2, 3])
            valid_share = float(
                (
                    np.isfinite(rgb).all(axis=0)
                    & (np.abs(rgb).sum(axis=0) > 0)
                ).mean()
            )
            scl = dataset.read(12) if dataset.count >= 12 else None
            cloud_share = (
                float(np.isin(scl, [3, 8, 9, 10, 11]).mean())
                if scl is not None
                else 0.0
            )
        ranked.append((valid_share, -cloud_share, path.name, path))
    return max(ranked)[-1]


def _paired_patch_ids() -> list[str]:
    before = {path.stem for path in (EMBEDDING_ROOT / BEFORE_MONTH).glob("patch_*.npy")}
    after = {path.stem for path in (EMBEDDING_ROOT / AFTER_MONTH).glob("patch_*.npy")}
    return sorted(before & after)


def _summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "mean": float(finite.mean()),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def _colored_map(scores: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    low, high = limits
    scaled = np.clip((scores - low) / (high - low), 0.0, 1.0)
    rgba = CMAP(np.nan_to_num(scaled, nan=0.0), bytes=True)
    rgb = rgba[..., :3]
    rgb[~np.isfinite(scores)] = (238, 241, 244)
    return rgb


def threshold_colored_map(
    scores: np.ndarray,
    low: float,
    threshold: float,
) -> np.ndarray:
    """Use cool colors below a hard red threshold."""
    if threshold <= low:
        raise ValueError("threshold must be greater than low")
    values = np.asarray(scores, dtype=np.float32)
    scaled = np.clip((values - low) / (threshold - low), 0.0, 1.0)
    blue = np.array([49.0, 86.0, 166.0], dtype=np.float32)
    light_blue = np.array([213.0, 234.0, 247.0], dtype=np.float32)
    pale_yellow = np.array([255.0, 247.0, 188.0], dtype=np.float32)
    red = np.array([196.0, 52.0, 46.0], dtype=np.float32)

    first = np.clip(scaled / 0.75, 0.0, 1.0)[..., None]
    second = np.clip((scaled - 0.75) / 0.25, 0.0, 1.0)[..., None]
    below = blue + first * (light_blue - blue)
    below = below + second * (pale_yellow - light_blue)
    rgb = np.where((values >= threshold)[..., None], red, below)
    rgb = np.rint(np.nan_to_num(rgb, nan=0.0)).astype(np.uint8)
    rgb[~np.isfinite(values)] = (238, 241, 244)
    return rgb


def _overlay(optical: np.ndarray, heatmap: np.ndarray, valid: np.ndarray) -> np.ndarray:
    blended = optical.copy()
    blended[valid] = np.rint(
        optical[valid].astype(np.float32) * 0.52
        + heatmap[valid].astype(np.float32) * 0.48
    ).astype(np.uint8)
    return blended


def _save_image(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values).resize((512, 512), Image.Resampling.NEAREST).save(path)


def _write_html(output: Path, rows: list[dict], overall: dict) -> None:
    cards = []
    for row in rows:
        metrics = row["metrics"]
        figures = []
        for key, title in (
            ("before_rgb", "2025 年 12 月光学影像"),
            ("after_rgb", "2026 年 4 月光学影像"),
            ("cosine_heatmap", "余弦变化距离"),
            ("cosine_overlay", "余弦变化叠加"),
            ("euclidean_heatmap", "归一化欧氏距离"),
            ("euclidean_overlay", "欧氏变化叠加"),
        ):
            figures.append(
                f"<figure><button class='zoom' data-src='{row[key]}' "
                f"aria-label='放大{title}'><img src='{row[key]}' alt='{title}'></button>"
                f"<figcaption>{title}</figcaption></figure>"
            )
        cards.append(
            "<section class='patch'>"
            f"<h2>{row['patch_id']}</h2>"
            f"<p class='scene'>实际影像：{row['before_scene']} → {row['after_scene']}</p>"
            "<div class='metrics'>"
            f"<span>余弦 P95 <b>{metrics['cosine']['p95']:.4f}</b></span>"
            f"<span>余弦高变化 <b>{metrics['cosine']['high_share']:.1%}</b></span>"
            f"<span>欧氏 P95 <b>{metrics['euclidean']['p95']:.4f}</b></span>"
            f"<span>欧氏高变化 <b>{metrics['euclidean']['high_share']:.1%}</b></span>"
            "</div><div class='views'>" + "".join(figures) + "</div></section>"
        )

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>海淀 Embedding 变化检测对照实验</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f8;color:#18212a;font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1780px;margin:auto;padding:24px}}header,.patch{{background:#fff;border:1px solid #dce2e7;border-radius:8px;padding:20px;margin-bottom:18px}}
h1{{font-size:26px;margin:0 0 8px}}h2{{font-size:19px;margin:0}}p{{margin:6px 0}}.note{{color:#53606c}}code{{background:#eef2f5;padding:2px 5px;border-radius:3px}}
.legend{{height:14px;max-width:460px;background:linear-gradient(90deg,#3b4cc0,#f7f7f7,#b40426);border:1px solid #bcc5cd;margin-top:12px}}
.legend-labels{{display:flex;justify-content:space-between;max-width:460px;font-size:13px;color:#53606c}}
.metrics{{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0}}.metrics span{{border-left:3px solid #607d94;padding-left:8px}}
.scene{{color:#65717c;font-size:13px}}.views{{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));gap:12px}}
figure{{margin:0;min-width:0}}.zoom{{border:0;padding:0;background:none;cursor:zoom-in;width:100%}}img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#eef1f3;border:1px solid #d9dfe4}}
figcaption{{font-weight:600;text-align:center;margin-top:5px;font-size:13px}}dialog{{border:0;border-radius:8px;padding:12px;background:#111;max-width:94vw;max-height:94vh}}dialog img{{width:auto;max-width:90vw;max-height:88vh;border:0}}dialog::backdrop{{background:rgba(0,0,0,.78)}}
@media(max-width:1100px){{.views{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:620px){{main{{padding:10px}}.views{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<header><h1>海淀 Embedding 变化检测对照实验</h1>
<p><code>{BEFORE_MONTH}</code> 对比 <code>{AFTER_MONTH}</code>，P10C 64D embedding。</p>
<p class="note">蓝色表示变化小，红色表示变化大。所有 Patch 使用统一色标；高变化指超过全体像素 P95。</p>
<div class="legend"></div><div class="legend-labels"><span>变化小</span><span>变化中等</span><span>变化大</span></div>
<p>两种指标相关性：Pearson <b>{overall['pearson']:.6f}</b>，Spearman <b>{overall['spearman']:.6f}</b>。
余弦色标 {overall['cosine_limits'][0]:.4f}–{overall['cosine_limits'][1]:.4f}；
欧氏色标 {overall['euclidean_limits'][0]:.4f}–{overall['euclidean_limits'][1]:.4f}。</p></header>
{''.join(cards)}
</main><dialog id="viewer"><img alt="放大预览"></dialog>
<script>
const viewer=document.querySelector('#viewer'), image=viewer.querySelector('img');
document.querySelectorAll('.zoom').forEach(button=>button.addEventListener('click',()=>{{image.src=button.dataset.src;viewer.showModal();}}));
viewer.addEventListener('click',()=>viewer.close());
</script></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def run_experiment(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    patch_ids = _paired_patch_ids()
    maps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    stats = []
    failures = []
    for patch_id in patch_ids:
        try:
            before = np.load(EMBEDDING_ROOT / BEFORE_MONTH / f"{patch_id}.npy")
            after = np.load(EMBEDDING_ROOT / AFTER_MONTH / f"{patch_id}.npy")
            cosine, euclidean, valid = change_scores(before, after)
            maps[patch_id] = (cosine, euclidean, valid)
            stats.append(
                {
                    "patch_id": patch_id,
                    "cosine_p95": _summary(cosine)["p95"],
                    "euclidean_p95": _summary(euclidean)["p95"],
                }
            )
        except (OSError, ValueError) as exc:
            failures.append({"patch_id": patch_id, "error": str(exc)})

    cosine_limits = global_limits([item[0] for item in maps.values()])
    euclidean_limits = global_limits([item[1] for item in maps.values()])
    cosine_high = cosine_limits[1]
    euclidean_high = euclidean_limits[1]
    selected = select_representative_patches(stats)
    rows = []
    correlation_cosine = []
    correlation_euclidean = []

    for patch_id in selected:
        cosine, euclidean, valid = maps[patch_id]
        before_scene = _best_optical(BEFORE_MONTH, patch_id)
        after_scene = _best_optical(AFTER_MONTH, patch_id)
        before_rgb = robust_rgb(before_scene)
        after_rgb = robust_rgb(after_scene)
        cosine_heatmap = _colored_map(cosine, cosine_limits)
        euclidean_heatmap = _colored_map(euclidean, euclidean_limits)
        files = {
            "before_rgb": f"{patch_id}_202512_rgb.png",
            "after_rgb": f"{patch_id}_202604_rgb.png",
            "cosine_heatmap": f"{patch_id}_cosine.png",
            "cosine_overlay": f"{patch_id}_cosine_overlay.png",
            "euclidean_heatmap": f"{patch_id}_euclidean.png",
            "euclidean_overlay": f"{patch_id}_euclidean_overlay.png",
        }
        _save_image(output / files["before_rgb"], before_rgb)
        _save_image(output / files["after_rgb"], after_rgb)
        _save_image(output / files["cosine_heatmap"], cosine_heatmap)
        _save_image(output / files["cosine_overlay"], _overlay(after_rgb, cosine_heatmap, valid))
        _save_image(output / files["euclidean_heatmap"], euclidean_heatmap)
        _save_image(output / files["euclidean_overlay"], _overlay(after_rgb, euclidean_heatmap, valid))

        cosine_values = cosine[valid]
        euclidean_values = euclidean[valid]
        correlation_cosine.append(cosine_values)
        correlation_euclidean.append(euclidean_values)
        cosine_stats = _summary(cosine)
        cosine_stats["high_share"] = float((cosine_values >= cosine_high).mean())
        euclidean_stats = _summary(euclidean)
        euclidean_stats["high_share"] = float((euclidean_values >= euclidean_high).mean())
        rows.append(
            {
                "patch_id": patch_id,
                "before_scene": before_scene.stem,
                "after_scene": after_scene.stem,
                "metrics": {"cosine": cosine_stats, "euclidean": euclidean_stats},
                **files,
            }
        )

    cosine_values = np.concatenate(correlation_cosine)
    euclidean_values = np.concatenate(correlation_euclidean)
    overall = {
        "before_month": BEFORE_MONTH,
        "after_month": AFTER_MONTH,
        "paired_patch_count": len(patch_ids),
        "evaluated_patch_count": len(maps),
        "selected_patch_ids": selected,
        "cosine_limits": cosine_limits,
        "euclidean_limits": euclidean_limits,
        "pearson": float(pearsonr(cosine_values, euclidean_values).statistic),
        "spearman": float(spearmanr(cosine_values, euclidean_values).statistic),
        "failures": failures,
    }
    result = {"overall": overall, "patches": rows}
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_html(output, rows, overall)
    return result


def main() -> None:
    output = ROOT / "Tmp" / f"haidian_embedding_change_{datetime.now():%Y%m%d_%H%M%S}"
    result = run_experiment(output)
    print(output)
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
