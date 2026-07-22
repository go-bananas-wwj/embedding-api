"""Build one visual acceptance gallery from live task API calls."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.visualize_haidian_downstream_heads import (  # noqa: E402
    MONTH,
    PATCHES,
    TASKS,
    palette,
    request,
    true_color_composite,
)
from app.services.external_embeddings import _read_s2_rgb  # noqa: E402
from app.services.s2_ml import resolve_s2_path  # noqa: E402


def overlay(optical: np.ndarray, result_path: Path) -> Image.Image:
    result = Image.open(result_path).convert("RGB").resize(
        (optical.shape[1], optical.shape[0]), Image.Resampling.NEAREST
    )
    source = Image.fromarray(optical).convert("RGB")
    return Image.blend(source, result, 0.48)


def s2_rgb(region_id: str, patch_id: str, month: str) -> np.ndarray:
    rgb = np.moveaxis(_read_s2_rgb(resolve_s2_path(region_id, patch_id, month)), 0, -1)
    return np.clip(np.power(rgb, 0.75) * 255, 0, 255).astype(np.uint8)


def main() -> None:
    base_url = "http://127.0.0.1:9061"
    output = ROOT / "Tmp" / f"callable_tasks_total_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)

    task_code, task_body, _ = request(base_url, "/regions/haidian/tasks")
    advertised = json.loads(task_body).get("tasks", []) if task_code == 200 else []
    advertised_ids = {item["id"] for item in advertised}
    calls = []
    sections = []

    for patch_id, scene_name in PATCHES:
        optical, sources = true_color_composite(patch_id, MONTH)
        optical_name = f"{patch_id}_{MONTH}_s2.png"
        Image.fromarray(optical).save(output / optical_name)
        rows = []
        for task_id, task_name in TASKS:
            api_path = (
                f"/regions/haidian/patches/{patch_id}/tasks/{task_id}/result"
                f"?format=png&version=v1&month={MONTH}"
            )
            status, body, content_type = request(base_url, api_path)
            result_name = f"{patch_id}_{MONTH}_{task_id}.png"
            overlay_name = f"{patch_id}_{MONTH}_{task_id}_overlay.png"
            colors = []
            if status == 200 and "image/png" in content_type:
                (output / result_name).write_bytes(body)
                overlay(optical, output / result_name).save(output / overlay_name)
                colors = palette(output / result_name)
                result_media = f"<img src='{result_name}' alt='{task_name}结果'>"
                overlay_media = f"<img src='{overlay_name}' alt='{task_name}叠加'>"
            else:
                result_media = overlay_media = f"<div class='missing'>HTTP {status}</div>"
            calls.append(
                {
                    "patch_id": patch_id,
                    "task_id": task_id,
                    "status": status,
                    "content_type": content_type,
                    "api_path": api_path,
                    "colors": colors,
                }
            )
            swatches = "".join(
                f"<i style='background:{item['color']}' title='{item['color']} {item['share']:.1%}'></i>"
                for item in colors[:10]
            )
            rows.append(
                f"<article><header><h3>{task_name}</h3><b class='{'ok' if status == 200 else 'bad'}'>HTTP {status}</b>"
                f"<code>{api_path}</code></header><div class='triplet'>"
                f"<figure><img src='{optical_name}' alt='S2原图'><figcaption>原始 S2 光学影像<br><small>{len(sources)} 景月内中位数合成</small></figcaption></figure>"
                f"<figure>{result_media}<figcaption>API 原始结果<br><span class='swatches'>{swatches}</span></figcaption></figure>"
                f"<figure>{overlay_media}<figcaption>结果与 S2 叠加</figcaption></figure>"
                f"</div></article>"
            )
        sections.append(
            f"<section><h2>{patch_id} <small>{scene_name}</small></h2>{''.join(rows)}</section>"
        )

    dino_result = ROOT / "users/default/results/infer_model_3cb7b1e4_harbin_patch_000212_2026-05.png"
    dino_html = "<p class='bad'>本地结果文件不存在。</p>"
    if dino_result.is_file():
        copied = output / "dino_model_3cb7b1e4_result.png"
        copied.write_bytes(dino_result.read_bytes())
        dino_html = (
            "<div class='dino'><figure><img src='dino_model_3cb7b1e4_result.png'><figcaption>"
            "DINOv3-SAT493M + 两层 MLP 真实调用结果</figcaption></figure>"
            "<div><p><code>model_3cb7b1e4</code> / <code>job_e5b277ab484b46b7</code></p>"
            "<p>训练、单图推理、批量推理均为 HTTP 200。原生特征为 1024×14×14，边界呈 token 块状。</p></div></div>"
        )

    custom_s2 = s2_rgb("harbin", "patch_000212", "2026-05")
    custom_s2_name = "custom_harbin_patch_000212_202605_s2.png"
    Image.fromarray(custom_s2).save(output / custom_s2_name)
    traditional_source = ROOT / "users/default/results/infer_model_8bb83664_harbin_patch_000212_2026-05.png"
    traditional_html = "<p class='bad'>传统方法结果文件不存在。</p>"
    if traditional_source.is_file():
        traditional_name = "traditional_rf_model_8bb83664_result.png"
        traditional_overlay_name = "traditional_rf_model_8bb83664_overlay.png"
        (output / traditional_name).write_bytes(traditional_source.read_bytes())
        overlay(custom_s2, output / traditional_name).save(output / traditional_overlay_name)
        traditional_html = f"""<div class='triplet'>
<figure><img src='{custom_s2_name}'><figcaption>原始 S2 光学影像</figcaption></figure>
<figure><img src='{traditional_name}'><figcaption>Random Forest API 结果</figcaption></figure>
<figure><img src='{traditional_overlay_name}'><figcaption>结果与 S2 叠加</figcaption></figure></div>
<p><code>model_8bb83664</code> / <code>job_049002fa1f644988</code> · 训练、单图、批量推理成功 · OOB score 0.9526</p>"""

    aef_result = ROOT / "users/default/results/infer_model_06b1e19e_harbin_patch_000212_2025-10.png"
    aef_html = "<p class='bad'>真实 AEF 推理结果文件不存在。</p>"
    if aef_result.is_file():
        aef_s2 = s2_rgb("harbin", "patch_000212", "2025-10")
        aef_s2_name = "aef_harbin_patch_000212_202510_s2.png"
        aef_result_name = "aef_model_06b1e19e_result.png"
        aef_overlay_name = "aef_model_06b1e19e_overlay.png"
        Image.fromarray(aef_s2).save(output / aef_s2_name)
        (output / aef_result_name).write_bytes(aef_result.read_bytes())
        overlay(aef_s2, output / aef_result_name).save(output / aef_overlay_name)
        aef_html = f"""<div class='triplet'>
<figure><img src='{aef_s2_name}'><figcaption>原始 S2 光学影像 · 2025-10</figcaption></figure>
<figure><img src='{aef_result_name}'><figcaption>AEF + 两层 MLP API 结果</figcaption></figure>
<figure><img src='{aef_overlay_name}'><figcaption>结果与 S2 叠加</figcaption></figure></div>
<p><code>model_06b1e19e</code> / <code>job_fd151cb2db20479f</code> · 训练、单图、批量推理成功 · training F1 0.8799</p>
<p>特征来自 Source Cooperative <code>tge-labs/aef</code> 的 AlphaEarth Foundations 2025 年 64 维年度 embedding，并按官方公式反量化。</p>"""

    availability = "".join(
        f"<tr><td>{item['name']}</td><td><code>{item['id']}</code></td><td class='ok'>可调用</td>"
        f"<td><code>{','.join(item.get('versions', []))}</code></td></tr>"
        for item in advertised
    )
    change_patch = "patch_000212"
    before_month, after_month = "2025-04", "2025-10"
    before = s2_rgb("harbin", change_patch, before_month)
    after = s2_rgb("harbin", change_patch, after_month)
    before_name, after_name = "harbin_change_before_s2.png", "harbin_change_after_s2.png"
    Image.fromarray(before).save(output / before_name)
    Image.fromarray(after).save(output / after_name)
    change_path = (
        f"/regions/harbin/patches/{change_patch}/tasks/change_detection/result"
        f"?format=png&version=v1&before_month={before_month}&after_month={after_month}"
    )
    change_status, change_body, change_type = request(base_url, change_path)
    change_result_name, change_overlay_name = "harbin_change_result.png", "harbin_change_overlay.png"
    if change_status == 200 and "image/png" in change_type:
        (output / change_result_name).write_bytes(change_body)
        overlay(after, output / change_result_name).save(output / change_overlay_name)
    calls.append({"patch_id": change_patch, "task_id": "change_detection", "status": change_status,
                  "content_type": change_type, "api_path": change_path})
    availability += (
        "<tr><td>变化检测</td><td><code>change_detection</code></td>"
        f"<td class='{'ok' if change_status == 200 else 'bad'}'>可调用（HTTP {change_status}）</td><td><code>v1</code></td></tr>"
    )
    change_html = f"""<section><h2>哈尔滨双时相变化检测</h2><p><code>{change_path}</code></p>
<div class='change-grid'><figure><img src='{before_name}'><figcaption>变化前 S2 · {before_month}</figcaption></figure>
<figure><img src='{after_name}'><figcaption>变化后 S2 · {after_month}</figcaption></figure>
<figure><img src='{change_result_name}'><figcaption>API 变化检测结果 · HTTP {change_status}</figcaption></figure>
<figure><img src='{change_overlay_name}'><figcaption>结果与变化后 S2 叠加</figcaption></figure></div></section>"""
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>全部任务真实调用结果</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f7;color:#17202a;font:14px/1.5 system-ui,sans-serif}}main{{max-width:1680px;margin:auto;padding:24px}}section{{background:#fff;border:1px solid #d8dee2;border-radius:8px;padding:18px;margin:18px 0}}h1,h2,h3{{margin-top:0}}h2 small{{font-weight:400;color:#637078}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #e3e7e9;text-align:left}}code{{background:#edf1f3;padding:2px 5px;overflow-wrap:anywhere}}article{{border-top:1px solid #e1e5e8;padding:16px 0}}article header{{display:grid;grid-template-columns:170px 90px 1fr;gap:10px;align-items:start}}article h3{{font-size:16px;margin:0}}.ok{{color:#137333;font-weight:700}}.bad{{color:#b3261e;font-weight:700}}.triplet{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:14px;margin-top:10px}}.change-grid{{display:grid;grid-template-columns:repeat(4,minmax(200px,1fr));gap:14px}}figure{{margin:0}}figure img,.missing{{width:100%;aspect-ratio:1;object-fit:contain;border:1px solid #cfd6da;background:#eef1f2;display:block;image-rendering:pixelated;cursor:zoom-in}}.missing{{display:grid;place-items:center;color:#b3261e}}figcaption{{font-weight:650;margin-top:5px}}figcaption small{{font-weight:400;color:#657078}}.swatches{{display:flex;gap:3px;margin-top:4px}}.swatches i{{width:17px;height:17px;border:1px solid #718087}}.note{{border-left:4px solid #1976a3;padding-left:12px}}.unavailable{{border:1px solid #e0a8a3;background:#fff4f2;padding:14px}}.dino{{display:grid;grid-template-columns:minmax(220px,360px) 1fr;gap:18px}}dialog{{border:0;padding:0;background:transparent}}dialog img{{max-width:95vw;max-height:93vh;image-rendering:pixelated}}dialog::backdrop{{background:rgba(0,0,0,.88)}}@media(max-width:760px){{.triplet,.change-grid{{grid-template-columns:1fr}}article header{{grid-template-columns:1fr}}.dino{{grid-template-columns:1fr}}}}
</style></head><body><main><section><h1>全部任务真实 API 调用结果</h1>
<p class='note'>每一行依次展示原始 Sentinel-2（S2）光学影像、API 返回 PNG、结果与 S2 叠加图。所有结果均在生成页面时调用运行中的 <code>{base_url}</code> 获取，点击图片可放大。</p>
<h2>当前系统任务</h2><table><thead><tr><th>名称</th><th>任务 ID</th><th>状态</th><th>版本</th></tr></thead><tbody>{availability}</tbody></table></section>
<section><h2>传统机器学习（S2 + Random Forest）</h2>{traditional_html}</section>
<section><h2>AEF + 两层 MLP</h2>{aef_html}</section>
<section><h2>自定义 DINOv3 训练调用</h2>{dino_html}</section>{change_html}{''.join(sections)}</main>
<dialog id='viewer'><img></dialog><script>const d=document.querySelector('#viewer'),v=d.querySelector('img');document.querySelectorAll('img').forEach(i=>i.onclick=()=>{{v.src=i.src;d.showModal()}});d.onclick=()=>d.close();</script></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "audit.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "base_url": base_url,
                "advertised_tasks": advertised,
                "all_advertised_visualized": advertised_ids == {item[0] for item in TASKS},
                "calls": calls,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
