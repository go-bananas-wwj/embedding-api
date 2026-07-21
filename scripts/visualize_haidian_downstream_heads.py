"""Compare every Haidian downstream API result with source optical imagery."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio


ROOT = Path(__file__).resolve().parents[1]
S2_ROOT = ROOT / "data/haidian/archive/processed_training_data/extracted/patches/s2"
MONTH = "202603"
TASKS = [
    ("building_extraction", "建筑物提取"),
    ("road_extraction", "道路提取"),
    ("construction", "施工地检测"),
    ("land_use_classification", "土地利用分类"),
    ("land_cover_classification", "土地覆盖分类"),
    ("water_extraction", "水体提取"),
]
PATCHES = [
    ("patch_000205", "密集建成区"),
    ("patch_000264", "道路与混合地表"),
    ("patch_000198", "施工地与城市边缘"),
    ("patch_000276", "水体与农田"),
    ("patch_000089", "山地植被与混合地表"),
]


def request(base_url: str, path: str) -> tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(f"{base_url}{path}", timeout=20) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers.get("Content-Type", "")


def true_color_composite(patch_id: str, month: str) -> tuple[np.ndarray, list[str]]:
    scenes = sorted(S2_ROOT.glob(f"s2_{month}*_{patch_id}.tif"))
    stack = []
    used = []
    for scene in scenes:
        mask_path = scene.with_name(scene.stem + "_mask.tif")
        with rasterio.open(scene) as dataset:
            data = dataset.read([3, 2, 1]).astype(np.float32)
        valid = np.isfinite(data).all(axis=0) & (data.max(axis=0) > 0)
        if mask_path.exists():
            with rasterio.open(mask_path) as dataset:
                valid &= dataset.read(1) > 0
        if not valid.any():
            continue
        data[:, ~valid] = np.nan
        stack.append(data)
        used.append(scene.name)
    if not stack:
        raise FileNotFoundError(f"No valid Sentinel-2 source for {patch_id}/{month}")
    with np.errstate(all="ignore"):
        composite = np.nanmedian(np.stack(stack), axis=0)
    valid = np.isfinite(composite).all(axis=0)
    rgb = np.moveaxis(composite, 0, -1)
    output = np.full(rgb.shape, 150, dtype=np.float32)
    if valid.any():
        low, high = np.nanpercentile(rgb[valid], [2, 98], axis=0)
        stretched = np.clip((rgb - low) / np.maximum(high - low, 1), 0, 1)
        output[valid] = np.power(stretched[valid], 0.85) * 255
    return output.astype(np.uint8), used


def palette(path: Path) -> list[dict]:
    array = np.asarray(Image.open(path).convert("RGB"))
    colors, counts = np.unique(array.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [
        {
            "color": f"#{colors[i, 0]:02X}{colors[i, 1]:02X}{colors[i, 2]:02X}",
            "share": float(counts[i] / counts.sum()),
        }
        for i in order
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:9061")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "Tmp" / f"haidian_downstream_heads_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)

    tasks_code, tasks_body, _ = request(args.base_url, "/regions/haidian/tasks")
    advertised = {
        item["id"]: item for item in json.loads(tasks_body).get("tasks", [])
    } if tasks_code == 200 else {}

    tile_statuses = []
    for task_id, task_name in TASKS:
        path = f"/regions/haidian/tasks/{task_id}/tiles?version=v1&period={MONTH}"
        code, body, _ = request(args.base_url, path)
        data = json.loads(body) if code == 200 else {"tiles": []}
        tile_statuses.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "advertised": task_id in advertised,
                "status": code,
                "tile_count": len(data.get("tiles", [])),
                "note": (
                    "按单 Patch 实时推理，当前无整月预生成 tiles"
                    if task_id == "road_extraction" and not data.get("tiles")
                    else "整月预生成 tiles 可用"
                ),
            }
        )

    rows = []
    for patch_id, scene_name in PATCHES:
        optical, sources = true_color_composite(patch_id, MONTH)
        optical_name = f"{patch_id}_{MONTH}_sentinel2.png"
        Image.fromarray(optical).save(output / optical_name)
        task_results = []
        for task_id, task_name in TASKS:
            api_path = (
                f"/regions/haidian/patches/{patch_id}/tasks/{task_id}/result"
                f"?format=png&version=v1&month={MONTH}"
            )
            code, body, content_type = request(args.base_url, api_path)
            filename = f"{patch_id}_{MONTH}_{task_id}.png"
            if code == 200 and "image/png" in content_type:
                (output / filename).write_bytes(body)
                image_palette = palette(output / filename)
            else:
                filename = None
                image_palette = []
            task_results.append(
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "status": code,
                    "content_type": content_type,
                    "image": filename,
                    "palette": image_palette,
                    "api_path": api_path,
                }
            )
        rows.append(
            {
                "patch_id": patch_id,
                "scene_name": scene_name,
                "optical": optical_name,
                "sources": sources,
                "results": task_results,
            }
        )

    status_html = "".join(
        f"<tr><td>{item['task_name']}</td><td><code>{item['task_id']}</code></td>"
        f"<td>{'是' if item['advertised'] else '否'}</td>"
        f"<td class='{'ok' if item['status'] == 200 else 'bad'}'>{item['status']}</td>"
        f"<td>{item['tile_count']}</td><td>{item['note']}</td></tr>"
        for item in tile_statuses
    )

    row_html = []
    for row in rows:
        cards = [
            f"<figure class='source'><img src='{row['optical']}'><figcaption>"
            f"<strong>原始 Sentinel-2 光学影像</strong>"
            f"<span>{len(row['sources'])} 景月内中位数合成</span></figcaption></figure>"
        ]
        for result in row["results"]:
            swatches = "".join(
                f"<i style='background:{item['color']}' title='{item['color']} {item['share']:.1%}'></i>"
                for item in result["palette"][:10]
            )
            if result["image"]:
                media = f"<img src='{result['image']}'>"
            else:
                media = f"<div class='missing'>HTTP {result['status']}</div>"
            cards.append(
                f"<figure>{media}<figcaption><strong>{result['task_name']}</strong>"
                f"<span>HTTP {result['status']} · {len(result['palette'])} 种颜色</span>"
                f"<span class='swatches'>{swatches}</span></figcaption></figure>"
            )
        row_html.append(
            f"<section><h2>{row['patch_id']} <small>{row['scene_name']}</small></h2>"
            f"<div class='matrix'>{''.join(cards)}</div></section>"
        )

    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>海淀下游头对照</title><style>
body{{margin:0;background:#f3f5f7;color:#17202a;font:14px/1.5 system-ui,sans-serif}}main{{max-width:2300px;margin:auto;padding:24px}}section{{background:#fff;border:1px solid #dce2e6;border-radius:8px;padding:18px;margin:18px 0}}h1,h2{{margin-top:0}}h2 small{{font-weight:400;color:#657078;margin-left:10px}}
table{{border-collapse:collapse;width:100%;max-width:900px}}th,td{{padding:8px 10px;border-bottom:1px solid #e5e9ec;text-align:left}}.ok{{color:#137333;font-weight:700}}.bad{{color:#b3261e;font-weight:700}}code{{background:#eef1f3;padding:2px 5px}}
.matrix{{display:grid;grid-template-columns:repeat(7,minmax(180px,1fr));gap:12px;overflow-x:auto;padding-bottom:6px}}figure{{margin:0;min-width:180px}}figure img,.missing{{width:100%;aspect-ratio:1;display:block;border:1px solid #d6dce0;image-rendering:pixelated}}figure img{{cursor:zoom-in}}.source img{{border:3px solid #1f6f8b;box-sizing:border-box}}.missing{{display:grid;place-items:center;background:#f2f3f4;color:#b3261e}}
figcaption{{display:flex;flex-direction:column;gap:2px;margin-top:6px}}figcaption span{{color:#657078}}.swatches{{display:flex;gap:3px;margin-top:3px}}.swatches i{{width:18px;height:18px;border:1px solid #859099}}.note{{border-left:4px solid #1f6f8b;padding-left:12px}}
dialog{{border:0;padding:0;background:transparent}}dialog img{{max-width:95vw;max-height:93vh;image-rendering:pixelated}}dialog::backdrop{{background:rgba(8,12,16,.9)}}
</style></head><body><main><section><h1>海淀区全部下游头结果对照</h1>
<p>每行是同一 Patch：最左为 2026 年 3 月的原始 Sentinel-2 光学合成图，右侧六列均由运行中的 API <code>GET .../result</code> 现场请求。点击图片可放大。</p>
<p class='note'>原始影像为当月有效 Sentinel-2 景的中位数合成与 2%–98% 显示拉伸，只用于人工解读；下游头推理使用的是 P10C 64 维嵌入，不是这张拉伸后的 PNG。</p>
<h2>任务与图块状态</h2><table><thead><tr><th>任务</th><th>ID</th><th>任务列表</th><th>HTTP</th><th>202603 Patch数</th><th>说明</th></tr></thead><tbody>{status_html}</tbody></table></section>
{''.join(row_html)}</main><dialog id='viewer'><img></dialog><script>const d=document.querySelector('#viewer'),v=d.querySelector('img');document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{{v.src=i.src;d.showModal()}});d.onclick=()=>d.close();</script></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "audit.json").write_text(
        json.dumps(
            {
                "base_url": args.base_url,
                "month": MONTH,
                "task_list_status": tasks_code,
                "tile_statuses": tile_statuses,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
