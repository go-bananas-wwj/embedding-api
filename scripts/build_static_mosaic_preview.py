#!/usr/bin/env python3
"""Build a compact browser audit for the generated static Mosaic package."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Tmp" / "static_mosaic_package_20260730"
STAGING = OUTPUT / "staging"
PREVIEW = OUTPUT / "preview"


def main() -> None:
    audit = json.loads((OUTPUT / "audit-report.json").read_text(encoding="utf-8"))
    PREVIEW.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for asset in audit["assets"]:
        grouped.setdefault((asset["region_id"], asset["sensor"]), []).append(asset)

    cards = []
    for (region_id, sensor), assets in sorted(grouped.items()):
        assets.sort(key=lambda item: item["date"])
        asset = assets[len(assets) // 2]
        source = STAGING / asset["path"]
        thumbnail_name = f"{region_id}_{sensor}_{asset['date']}.png"
        with Image.open(source) as image:
            thumbnail = image.copy()
            thumbnail.thumbnail((900, 650), Image.Resampling.LANCZOS)
            thumbnail.save(PREVIEW / thumbnail_name, optimize=True)
        full_path = f"../staging/{asset['path']}"
        cards.append(
            f"""
            <article>
              <header><strong>{region_id}</strong><span>{sensor}</span><time>{asset['date']}</time></header>
              <a href="{full_path}" target="_blank"><img src="{thumbnail_name}" alt="{region_id} {sensor} {asset['date']}"></a>
              <footer>{asset['width']} × {asset['height']} · {asset['size_bytes'] / 1024**2:.1f} MiB</footer>
            </article>"""
        )

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>静态区域大图抽样检查</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f6;color:#172026;font-family:Arial,"Noto Sans CJK SC",sans-serif}}
body>header{{padding:18px 24px;background:#fff;border-bottom:1px solid #d6dcdf}}h1{{font-size:22px;margin:0 0 6px}}
p{{margin:0;color:#5b6870}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px;padding:18px}}
article{{background:#fff;border:1px solid #d6dcdf;border-radius:6px;overflow:hidden}}article header{{display:grid;grid-template-columns:1fr 1fr auto;gap:8px;padding:10px 12px}}
article img{{display:block;width:100%;height:320px;object-fit:contain;background-color:#e7eaec;background-image:linear-gradient(45deg,#c8ced2 25%,transparent 25%),linear-gradient(-45deg,#c8ced2 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#c8ced2 75%),linear-gradient(-45deg,transparent 75%,#c8ced2 75%);background-size:20px 20px;background-position:0 0,0 10px,10px -10px,-10px 0}}
footer{{padding:9px 12px;color:#5b6870}}@media(max-width:520px){{main{{grid-template-columns:1fr;padding:10px}}article img{{height:260px}}}}
</style></head><body><header><h1>海淀区与哈尔滨新区静态大图抽样检查</h1>
<p>每个传感器抽取中间月份；点击缩略图查看原始 PNG。棋盘格表示透明背景。</p></header><main>{''.join(cards)}</main></body></html>"""
    (PREVIEW / "index.html").write_text(html, encoding="utf-8")
    print(PREVIEW / "index.html")


if __name__ == "__main__":
    main()
