"""Build a browser audit of the Haidian land-cover HTTP APIs."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MONTHS = ("202512", "202601", "202602", "202603", "202604", "202605")
REPRESENTATIVES = {
    "树木覆盖": "patch_000156",
    "灌木地": "patch_000101",
    "草地": "patch_000026",
    "耕地": "patch_000318",
    "建成区": "patch_000041",
    "裸地/稀疏植被": "patch_000059",
    "永久性水体": "patch_000146",
}
TIMELINE_PATCH = "patch_000089"


def request(base_url: str, path: str, method: str = "GET") -> tuple[int, bytes, str]:
    req = urllib.request.Request(f"{base_url}{path}", method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers.get("Content-Type", "")


def color_counts(image_path: Path) -> Counter[str]:
    pixels = np.asarray(Image.open(image_path).convert("RGB")).reshape(-1, 3)
    return Counter(f"#{r:02X}{g:02X}{b:02X}" for r, g, b in pixels)


def save_result(
    base_url: str, output: Path, patch_id: str, month: str, suffix: str
) -> tuple[Path, int, Counter[str]]:
    path = (
        f"/regions/haidian/patches/{patch_id}/tasks/"
        f"land_cover_classification/result?format=png&version=v1&month={month}"
    )
    status, body, content_type = request(base_url, path)
    if status != 200 or "image/png" not in content_type:
        raise RuntimeError(f"Result request failed: {path} -> {status} {body[:300]!r}")
    destination = output / f"{patch_id}_{month}_{suffix}.png"
    destination.write_bytes(body)
    return destination, len(body), color_counts(destination)


def status_row(label: str, method: str, path: str, status: int, expected: int, note: str):
    return {
        "label": label,
        "method": method,
        "path": path,
        "status": status,
        "expected": expected,
        "ok": status == expected,
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:9061")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "Tmp" / f"haidian_land_cover_api_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)

    statuses = []
    classes_path = "/system-models/land_cover_classification/classes?region_id=haidian&version=v1"
    code, body, _ = request(args.base_url, classes_path)
    classes = json.loads(body) if code == 200 else []
    statuses.append(status_row("七类图例", "GET", classes_path, code, 200, "应返回7个类别"))

    omitted_path = "/system-models/land_cover_classification/classes?region_id=haidian"
    omitted_code, omitted_body, _ = request(args.base_url, omitted_path)
    statuses.append(status_row("省略版本参数", "GET", omitted_path, omitted_code, 200, "应自动使用v1"))
    if omitted_code == 200 and json.loads(omitted_body) != classes:
        raise RuntimeError("Explicit and implicit v1 class responses differ")

    tile_counts = {}
    for month in MONTHS:
        tile_path = (
            "/regions/haidian/tasks/land_cover_classification/tiles"
            f"?version=v1&period={month}"
        )
        tile_code, tile_body, _ = request(args.base_url, tile_path)
        tile_data = json.loads(tile_body) if tile_code == 200 else {"tiles": []}
        tile_counts[month] = len(tile_data.get("tiles", []))
        statuses.append(
            status_row(
                f"{month}图块列表", "GET", tile_path, tile_code, 200,
                f"返回{tile_counts[month]}个Patch",
            )
        )

    representative_rows = []
    class_by_name = {item["name"]: item for item in classes}
    known_colors = {item["color"].upper() for item in classes}
    for class_name, patch_id in REPRESENTATIVES.items():
        image_path, size, counts = save_result(
            args.base_url, output, patch_id, "202603", "representative"
        )
        color = class_by_name[class_name]["color"].upper()
        share = counts[color] / sum(counts.values())
        unknown = sorted(set(counts) - known_colors)
        representative_rows.append(
            {
                "class_name": class_name,
                "patch_id": patch_id,
                "image": image_path.name,
                "size": size,
                "share": share,
                "colors": len(counts),
                "unknown": unknown,
            }
        )

    # Verify that the dedicated tile-file route returns the same PNG as result.
    compare_patch = REPRESENTATIVES["建成区"]
    tile_file_path = (
        "/regions/haidian/tasks/land_cover_classification/tiles/"
        f"{compare_patch}.png?version=v1&period=202603"
    )
    tile_code, tile_body, tile_type = request(args.base_url, tile_file_path)
    result_file = output / f"{compare_patch}_202603_representative.png"
    same_bytes = tile_code == 200 and tile_body == result_file.read_bytes()
    statuses.append(
        status_row(
            "图块文件", "GET", tile_file_path, tile_code, 200,
            f"Content-Type={tile_type}；与result内容一致={same_bytes}",
        )
    )

    timeline = []
    for month in MONTHS:
        image_path, size, counts = save_result(
            args.base_url, output, TIMELINE_PATCH, month, "timeline"
        )
        timeline.append(
            {
                "month": month,
                "image": image_path.name,
                "size": size,
                "colors": len(counts),
                "unknown": sorted(set(counts) - known_colors),
            }
        )

    unavailable = [
        (
            "实时推理",
            "POST",
            "/system-models/land_cover_classification/infer"
            "?region_id=haidian&patch_id=patch_000000&month=202603&version=v1",
            "当前只有预生成PNG，无在线checkpoint",
        ),
        (
            "摘要",
            "GET",
            "/regions/haidian/tasks/land_cover_classification/summary?version=v1",
            "当前缺少summary统计元数据",
        ),
        (
            "预测数组",
            "GET",
            "/regions/haidian/patches/patch_000000/tasks/"
            "land_cover_classification/prediction?version=v1",
            "当前未发布prediction文件",
        ),
        (
            "标签数组",
            "GET",
            "/regions/haidian/patches/patch_000000/tasks/"
            "land_cover_classification/label?version=v1",
            "当前未发布label文件",
        ),
    ]
    for label, method, path, note in unavailable:
        unavailable_code, _, _ = request(args.base_url, path, method)
        statuses.append(status_row(label, method, path, unavailable_code, 404, note))

    legend_html = "".join(
        f"<li><span style='background:{item['color']}'></span>"
        f"<code>{item['color']}</code><strong>{item['name']}</strong>"
        f"<small>{item['id']}</small></li>"
        for item in classes
    )
    status_html = "".join(
        f"<tr><td><b class='{'ok' if row['ok'] else 'bad'}'>{row['status']}</b></td>"
        f"<td>{row['label']}</td><td><code>{row['method']}</code></td>"
        f"<td><code>{row['path']}</code></td><td>{row['note']}</td></tr>"
        for row in statuses
    )
    representative_html = "".join(
        f"<figure><img src='{row['image']}'><figcaption><strong>{row['class_name']}</strong>"
        f"<code>{row['patch_id']}</code><span>目标颜色占比 {row['share']:.1%} · "
        f"图内 {row['colors']} 色 · 未知颜色 {len(row['unknown'])}</span></figcaption></figure>"
        for row in representative_rows
    )
    timeline_html = "".join(
        f"<figure><img src='{row['image']}'><figcaption><strong>{row['month']}</strong>"
        f"<span>{TIMELINE_PATCH} · {row['colors']} 色 · 未知颜色 {len(row['unknown'])}</span>"
        f"</figcaption></figure>"
        for row in timeline
    )
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>海淀土地覆盖API验收</title>
<style>body{{margin:0;background:#f4f6f7;color:#17202a;font:15px/1.55 system-ui,sans-serif}}main{{max-width:1760px;margin:auto;padding:24px}}
section{{background:white;border:1px solid #dce2e6;border-radius:8px;padding:20px;margin:18px 0}}h1,h2{{margin-top:0}}.legend{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;list-style:none;padding:0}}
.legend li{{display:grid;grid-template-columns:34px 76px 1fr;align-items:center;gap:8px}}.legend span{{width:30px;height:30px;border:1px solid #89939b}}.legend small{{grid-column:3;color:#647078}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #e4e8eb;padding:8px;text-align:left}}td:nth-child(4){{word-break:break-all}}.ok{{color:#137333}}.bad{{color:#b3261e}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}}figure{{margin:0}}figure img{{width:100%;aspect-ratio:1;image-rendering:pixelated;border:1px solid #d7dde1;cursor:zoom-in}}figcaption{{display:flex;flex-direction:column;gap:2px;margin-top:5px}}figcaption span,small{{color:#647078}}code{{background:#eef1f3;padding:2px 5px}}
.note{{border-left:4px solid #d6a100;padding-left:12px}}dialog{{border:0;padding:0;background:transparent}}dialog img{{max-width:94vw;max-height:92vh;image-rendering:pixelated}}dialog::backdrop{{background:rgba(8,12,16,.9)}}</style></head>
<body><main><section><h1>海淀土地覆盖 API 验收</h1><p>本页所有图片都通过运行中的 HTTP API 重新请求，不直接引用磁盘文件。</p>
<ul class='legend'>{legend_html}</ul></section><section><h2>接口状态</h2><table><thead><tr><th>HTTP</th><th>功能</th><th>方法</th><th>路径</th><th>验收结论</th></tr></thead><tbody>{status_html}</tbody></table>
<p class='note'>404 项是当前数据产品明确未提供的能力，不计为本次修复失败。当前可用能力是 classes、tiles 和 result PNG。</p></section>
<section><h2>7 类代表 Patch</h2><p>每张图来自 <code>GET .../result?month=202603</code>；未知颜色应为 0。</p><div class='grid'>{representative_html}</div></section>
<section><h2>同一 Patch 的 6 个月</h2><p>用于检查调色板在月度结果中是否保持一致。</p><div class='grid'>{timeline_html}</div></section>
</main><dialog id='viewer'><img></dialog><script>const d=document.querySelector('#viewer'),v=d.querySelector('img');document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{{v.src=i.src;d.showModal()}});d.onclick=()=>d.close();</script></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "audit.json").write_text(
        json.dumps(
            {
                "base_url": args.base_url,
                "classes": classes,
                "statuses": statuses,
                "tile_counts": tile_counts,
                "representatives": representative_rows,
                "timeline": timeline,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
