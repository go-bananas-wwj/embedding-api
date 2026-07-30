#!/usr/bin/env python3
"""Build a visual WGS84 alignment audit for Haidian mosaic data sources."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Tmp" / "mosaic_multisensor_alignment_20260730"
API_BASE = "http://127.0.0.1:9061/regions/haidian/mosaic"
DATE = "202603"
PATCH_IDS = ["patch_000000", "patch_000319"]
SENSORS = [
    ("s2", "Sentinel-2 光学"),
    ("s1", "Sentinel-1 SAR"),
    ("landsat", "Landsat 光学"),
    ("highres", "高分辨率光学"),
]
COLORS = {
    "patch_000000": (0, 255, 80, 255),
    "patch_000319": (255, 40, 180, 255),
}


def _url(sensor: str, output_format: str) -> str:
    params = [
        ("date", DATE),
        ("sensor_type", sensor),
        ("format", output_format),
        *(("patch_ids", patch_id) for patch_id in PATCH_IDS),
    ]
    return f"{API_BASE}?{urlencode(params)}"


def _load_json(url: str) -> dict:
    with urlopen(url, timeout=180) as response:
        return json.load(response)


def _load_image(url: str) -> Image.Image:
    with urlopen(url, timeout=300) as response:
        return Image.open(BytesIO(response.read())).convert("RGBA")


def _pixel_ring(coordinates: list, bounds: list[float], size: tuple[int, int]):
    min_x, min_y, max_x, max_y = bounds
    width, height = size
    return [
        (
            (lon - min_x) / (max_x - min_x) * width,
            (max_y - lat) / (max_y - min_y) * height,
        )
        for lon, lat in coordinates
    ]


def _draw_footprints(image: Image.Image, metadata: dict) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    stroke = max(3, round(max(image.size) / 900))
    for patch in metadata["patches"]:
        geometry = patch["footprint_wgs84"]
        polygons = (
            [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry["coordinates"]
        )
        for polygon in polygons:
            ring = _pixel_ring(
                polygon[0], metadata["bounds_wgs84"], image.size
            )
            draw.line(ring + [ring[0]], fill=COLORS[patch["patch_id"]], width=stroke)
    return overlay


def _crop_patch(image: Image.Image, patch: dict, bounds: list[float]) -> Image.Image:
    geometry = shape(patch["footprint_wgs84"])
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    width, height = image.size
    min_x, min_y, max_x, max_y = bounds
    left = (min_lon - min_x) / (max_x - min_x) * width
    right = (max_lon - min_x) / (max_x - min_x) * width
    top = (max_y - max_lat) / (max_y - min_y) * height
    bottom = (max_y - min_lat) / (max_y - min_y) * height
    padding = max(12, round(max(right - left, bottom - top) * 0.08))
    box = (
        max(0, int(left) - padding),
        max(0, int(top) - padding),
        min(width, int(right) + padding),
        min(height, int(bottom) + padding),
    )
    return image.crop(box)


def _alignment_iou(image: Image.Image, metadata: dict) -> float:
    alpha = np.asarray(image.getchannel("A")) > 0
    transform = from_bounds(*metadata["bounds_wgs84"], *image.size)
    footprints = [
        (shape(patch["footprint_wgs84"]), 1) for patch in metadata["patches"]
    ]
    expected = rasterize(
        footprints,
        out_shape=(image.height, image.width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)
    intersection = np.logical_and(alpha, expected).sum()
    union = np.logical_or(alpha, expected).sum()
    return float(intersection / union) if union else 1.0


def _sensor_section(sensor: str, label: str, metadata: dict, iou: float) -> str:
    patches = {item["patch_id"]: item for item in metadata["patches"]}
    return f"""
    <section>
      <div class="section-head">
        <div>
          <h2>{label} <code>{sensor}</code></h2>
          <p>采集日期：首 Patch {patches[PATCH_IDS[0]]["source_date"]}，
             末 Patch {patches[PATCH_IDS[1]]["source_date"]}</p>
        </div>
        <dl>
          <div><dt>尺寸</dt><dd>{metadata["width"]} × {metadata["height"]}</dd></div>
          <div><dt>坐标系</dt><dd>{metadata["crs"]}</dd></div>
          <div><dt>边界对齐 IoU</dt><dd>{iou:.6f}</dd></div>
        </dl>
      </div>
      <div class="mosaic checker">
        <a href="{sensor}_overlay.png" target="_blank">
          <img src="{sensor}_overlay.png" alt="{label} WGS84 边界叠加图">
        </a>
      </div>
      <div class="patch-grid">
        <figure class="checker">
          <a href="{sensor}_{PATCH_IDS[0]}_zoom.png" target="_blank">
            <img src="{sensor}_{PATCH_IDS[0]}_zoom.png" alt="{label} 首 Patch">
          </a>
          <figcaption><i class="first"></i>首 Patch：{PATCH_IDS[0]}</figcaption>
        </figure>
        <figure class="checker">
          <a href="{sensor}_{PATCH_IDS[1]}_zoom.png" target="_blank">
            <img src="{sensor}_{PATCH_IDS[1]}_zoom.png" alt="{label} 末 Patch">
          </a>
          <figcaption><i class="last"></i>末 Patch：{PATCH_IDS[1]}</figcaption>
        </figure>
      </div>
      <p class="links"><a href="{sensor}_metadata.json">WGS84 元数据</a> ·
         <a href="{sensor}_mosaic.png">无边界原图</a></p>
    </section>"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sections = []
    report = {}
    for sensor, label in SENSORS:
        metadata = _load_json(_url(sensor, "json"))
        image = _load_image(_url(sensor, "png"))
        overlay = _draw_footprints(image, metadata)
        iou = _alignment_iou(image, metadata)

        image.save(OUTPUT / f"{sensor}_mosaic.png", optimize=True)
        overlay.save(OUTPUT / f"{sensor}_overlay.png", optimize=True)
        for patch in metadata["patches"]:
            _crop_patch(
                overlay, patch, metadata["bounds_wgs84"]
            ).save(
                OUTPUT / f"{sensor}_{patch['patch_id']}_zoom.png",
                optimize=True,
            )
        (OUTPUT / f"{sensor}_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report[sensor] = {
            "label": label,
            "alignment_iou": iou,
            "width": metadata["width"],
            "height": metadata["height"],
            "bounds_wgs84": metadata["bounds_wgs84"],
            "patches": [
                {
                    "patch_id": patch["patch_id"],
                    "source_date": patch["source_date"],
                }
                for patch in metadata["patches"]
            ],
        }
        sections.append(_sensor_section(sensor, label, metadata, iou))

    (OUTPUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>海淀多数据源 Mosaic WGS84 对齐检查</title>
  <style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f3f5f6;color:#172026;
    font-family:Arial,"Noto Sans CJK SC",sans-serif}}header{{background:#fff;
    border-bottom:1px solid #d6dcdf;padding:20px 28px}}header h1{{font-size:24px;
    margin:0 0 8px}}header p{{margin:0;color:#536069}}main{{max-width:1500px;
    margin:auto;padding:20px}}section{{background:#fff;border:1px solid #d6dcdf;
    border-radius:6px;margin-bottom:20px;padding:18px}}.section-head{{display:flex;
    align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:14px}}
    h2{{font-size:20px;margin:0 0 5px}}p{{margin:0}}code{{background:#eef1f2;
    padding:2px 5px}}dl{{display:flex;gap:22px;margin:0}}dl div{{min-width:100px}}
    dt{{font-size:12px;color:#68757e}}dd{{margin:4px 0 0;font-weight:700}}
    .checker{{background-color:#e7eaec;background-image:linear-gradient(45deg,
    #c8ced2 25%,transparent 25%),linear-gradient(-45deg,#c8ced2 25%,transparent 25%),
    linear-gradient(45deg,transparent 75%,#c8ced2 75%),linear-gradient(-45deg,
    transparent 75%,#c8ced2 75%);background-size:20px 20px;
    background-position:0 0,0 10px,10px -10px,-10px 0}}.mosaic{{height:440px;
    overflow:auto;padding:10px;border:1px solid #d6dcdf}}.mosaic img{{display:block;
    width:auto;height:auto;max-width:none}}.patch-grid{{display:grid;
    grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}}figure{{margin:0;
    border:1px solid #d6dcdf;padding:10px}}figure img{{display:block;width:100%;
    max-height:420px;object-fit:contain}}figcaption{{background:#fff;margin:10px -10px
    -10px;padding:10px}}i{{display:inline-block;width:13px;height:13px;
    margin-right:7px;vertical-align:-1px}}i.first{{background:#00ff50}}
    i.last{{background:#ff28b4}}.links{{margin-top:12px}}a{{color:#166c7a}}
    @media(max-width:760px){{.section-head{{display:block}}dl{{margin-top:12px;
    flex-wrap:wrap}}.patch-grid{{grid-template-columns:1fr}}main{{padding:10px}}
    section{{padding:12px}}}}
  </style>
</head>
<body>
  <header>
    <h1>海淀区 2026 年 3 月多数据源 WGS84 对齐检查</h1>
    <p>绿色为首 Patch，洋红色为末 Patch；棋盘格表示没有 Patch 的透明区域。点击图片查看原尺寸。</p>
  </header>
  <main>{''.join(sections)}</main>
</body>
</html>"""
    (OUTPUT / "index.html").write_text(html, encoding="utf-8")
    print(OUTPUT)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
