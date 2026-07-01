# 替换海淀区任务结果为 monthly_osm_assisted_patch_tiles 诊断图归档

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 ModelScope `WeijieWu/xuannv_embdding_api/haidian/v1/reports/monthly_osm_assisted_patch_tiles/archives/` 下载 6 个月诊断图 tar 包，提取/转换后替换 `data/haidian/tasks/` 下的旧结果，使海淀区 4 类任务接口可用并重新生成全域可视化大图。

**Architecture:** 诊断图是 1312×342 的 5 面板拼接图，本次采用“提取第 4 面板（后处理结果）并缩放回 128×128 tile”的方案替换旧结果，保持 API 输出尺寸与前端 `xuannv_show` 兼容；按月份分目录存放，并设置默认月份 fallback。新增 `water_extraction` 配置。

**Tech Stack:** Python 3.9, PIL, tarfile, ModelScope SDK, FastAPI, config.yaml

---

## 已确认的前提信息

- ModelScope 路径：`WeijieWu/xuannv_embdding_api/haidian/v1/reports/monthly_osm_assisted_patch_tiles/archives/`
- 归档文件（共 6 个 tar，约 2.12 GB）：
  - `haidian_v1_202512_monthly_osm_assisted_patch_tiles.tar` (335 MB)
  - `haidian_v1_202601_monthly_osm_assisted_patch_tiles.tar` (339 MB)
  - `haidian_v1_202602_monthly_osm_assisted_patch_tiles.tar` (338 MB)
  - `haidian_v1_202603_monthly_osm_assisted_patch_tiles.tar` (345 MB)
  - `haidian_v1_202604_monthly_osm_assisted_patch_tiles.tar` (351 MB)
  - `haidian_v1_202605_monthly_osm_assisted_patch_tiles.tar` (310 MB)
- 每个 tar 内部结构：`{YYYYMM}/{task}/haidian_{YYYYMM}_{task}_patch_{XXXXXX}_osm_diagnostic.png`
- 任务目录（新）：`building_extraction`, `road_extraction`, `construction_site_extraction`, `water_extraction`
- 诊断图尺寸：1312×342 RGB，第 4 面板为“后处理结果”（推荐作为结果 tile）
- 旧结果位置：`data/haidian/tasks/{task}/v1/results/tiles/{patch_id}.png`（128×128）

---

## 需要用户在执行前确认的决策

1. **construction_site_extraction 映射到哪个现有任务？**
   - 推荐：**`construction`**（因为语义为施工工地，与旧 `construction` 一致）
   - 备选：`construction_joint`（如果你希望替换掉旧 `construction_joint`）
   - 本计划默认按 **`construction_site_extraction → construction`** 执行。

2. **是否把诊断图完整 1312×342 直接作为结果？**
   - 推荐：**否**。提取第 4 面板并缩放回 128×128，保持与 `xuannv_show`/旧接口兼容。
   - 若选“是”，则跳过 Task 5 的裁剪缩放，直接把诊断图按 `results/tiles/{patch_id}_{period}.png` 存放（API 会返回 1312×342，前端 tile 显示会异常）。

3. **默认展示哪个月份？**
   - 推荐：**202605**（最新一期），同时保留 202512-202605 全部月份供接口按 `before_month`/`after_month` 查询。

---

## 文件变更清单

- **新增脚本：** `scripts/replace_haidian_results_from_osm_archives.py`
- **修改配置：** `config.yaml`（新增 `water_extraction` v1 结果路径；确认 `construction` 路径不变）
- **修改文档：** `docs/API.md`（更新海淀区示例，说明为 OSM-assisted 诊断图后处理面板）
- **修改记录：** `progress.md`
- **数据替换：** `data/haidian/tasks/{building_extraction,road_extraction,construction,water_extraction}/v1/results/**`
- **生成图片：** `test_output/hd_v1/full_region_*.png`（不进入 git）

---

### Task 1: 下载并校验 6 个月归档

**Files:**
- Create: `/tmp/hd_osm_archives/`（临时下载目录，执行后可删除）
- Read: `haidian/v1/reports/monthly_osm_assisted_patch_tiles/archives/checksums.sha256`

- [ ] **Step 1: 创建临时目录并下载 6 个 tar**

```bash
mkdir -p /tmp/hd_osm_archives
cd /tmp/hd_osm_archives
python - <<'PY'
from modelscope.hub.api import HubApi
import requests, os
api = HubApi()
api.login('ms-399d1804-1cb3-446a-a3f7-dfc4dc70d977')
months = ['202512','202601','202602','202603','202604','202605']
for m in months:
    path = f'haidian/v1/reports/monthly_osm_assisted_patch_tiles/archives/haidian_v1_{m}_monthly_osm_assisted_patch_tiles.tar'
    url = api.get_dataset_file_url(path, 'xuannv_embdding_api', 'WeijieWu', revision='master')
    out = f'/tmp/hd_osm_archives/{os.path.basename(path)}'
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print('exists', out)
        continue
    print('downloading', path)
    r = requests.get(url, stream=True, timeout=600)
    r.raise_for_status()
    with open(out, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print('saved', out, os.path.getsize(out))
PY
```

- [ ] **Step 2: 下载并校验 checksums.sha256**

```bash
cd /tmp/hd_osm_archives
python - <<'PY'
from modelscope.hub.api import HubApi
import requests, hashlib, os
api = HubApi(); api.login('ms-399d1804-1cb3-446a-a3f7-dfc4dc70d977')
url = api.get_dataset_file_url(
    'haidian/v1/reports/monthly_osm_assisted_patch_tiles/archives/checksums.sha256',
    'xuannv_embdding_api', 'WeijieWu', revision='master')
open('checksums.sha256','wb').write(requests.get(url, timeout=60).content)
expected = {}
for line in open('checksums.sha256'):
    h,p = line.strip().split('  ')
    expected[os.path.basename(p)] = h
ok=0
for name,h in expected.items():
    data = open(name,'rb').read()
    actual = hashlib.sha256(data).hexdigest()
    if actual == h:
        print('OK', name)
        ok+=1
    else:
        print('FAIL', name, 'expected', h, 'actual', actual)
print(f'checksum {ok}/{len(expected)} passed')
PY
```

**Expected output:** `checksum 6/6 passed`

---

### Task 2: 编写替换/转换脚本

**Files:**
- Create: `scripts/replace_haidian_results_from_osm_archives.py`

- [ ] **Step 1: 写出数据转换脚本**

```python
# scripts/replace_haidian_results_from_osm_archives.py
"""
Extract post-processed panel from Haidian monthly OSM-assisted diagnostic
strips and replace the old 128x128 result tiles.
"""
import argparse
import hashlib
import io
import os
import re
import tarfile
from pathlib import Path

from PIL import Image

# New task dir in archive -> existing task id in config.yaml
TASK_MAP = {
    'building_extraction': 'building_extraction',
    'road_extraction': 'road_extraction',
    'construction_site_extraction': 'construction',
    'water_extraction': 'water_extraction',
}

# Geometry of the 1312x342 diagnostic strip (verified on sample):
# - usable horizontal strip starts at y=64 and is 256 px high
# - 5 panels, each 256 px wide, separated by ~6 px gaps
PANEL_LEFTS = [16, 278, 540, 802, 1064]
PANEL_TOP = 64
PANEL_SIZE = 256
DEFAULT_PERIOD = '202605'


def extract_postprocessed_panel(img: Image.Image) -> Image.Image:
    """Return the 4th panel (post-processed result) as a 128x128 RGB image."""
    left = PANEL_LEFTS[3]
    panel = img.crop((left, PANEL_TOP,
                      left + PANEL_SIZE, PANEL_TOP + PANEL_SIZE))
    return panel.resize((128, 128), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archives-dir', default='/tmp/hd_osm_archives')
    parser.add_argument('--out-root', default='data/haidian/tasks')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    archives = sorted(Path(args.archives_dir).glob('haidian_v1_*_monthly_osm_assisted_patch_tiles.tar'))
    if not archives:
        raise SystemExit(f'No tar files found in {args.archives_dir}')

    out_root = Path(args.out_root)

    for tar_path in archives:
        period = tar_path.stem.split('_')[2]  # e.g. 202512
        print(f'\nProcessing {tar_path.name} -> period {period}')
        with tarfile.open(tar_path, 'r') as tf:
            members = [m for m in tf.getmembers() if m.name.endswith('_osm_diagnostic.png')]
            print(f'  members: {len(members)}')
            for member in members:
                # path: 202512/building_extraction/haidian_202512_building_extraction_patch_000141_osm_diagnostic.png
                parts = member.name.split('/')
                archive_task = parts[1]
                task = TASK_MAP.get(archive_task)
                if task is None:
                    continue
                m = re.search(r'patch_(\d{6})', member.name)
                if not m:
                    continue
                patch_id = f"patch_{m.group(1)}"

                # destination: data/haidian/tasks/{task}/v1/results/{period}/tiles/{patch_id}.png
                out_dir = out_root / task / 'v1' / 'results' / period / 'tiles'
                out_path = out_dir / f'{patch_id}.png'

                if args.dry_run:
                    print(f'  would write {out_path}')
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)
                data = tf.extractfile(member).read()
                img = Image.open(io.BytesIO(data)).convert('RGB')
                tile = extract_postprocessed_panel(img)
                tile.save(out_path)

        # After processing all periods, create per-patch fallback from the default period
        print(f'\nCreating fallback tiles from period {DEFAULT_PERIOD} ...')
        for task in TASK_MAP.values():
            fallback_src = out_root / task / 'v1' / 'results' / DEFAULT_PERIOD / 'tiles'
            fallback_dst = out_root / task / 'v1' / 'results' / 'tiles'
            if not fallback_src.exists():
                continue
            fallback_dst.mkdir(parents=True, exist_ok=True)
            for src in sorted(fallback_src.glob('patch_*.png')):
                dst = fallback_dst / src.name
                if args.dry_run:
                    print(f'  would fallback {dst}')
                    continue
                # hardlink if possible, otherwise copy
                try:
                    os.link(src, dst)
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)

    print('\nDone.')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 先 dry-run 验证输出路径**

```bash
cd /workspace/embedding-api
python scripts/replace_haidian_results_from_osm_archives.py --dry-run
```

**Expected:** 打印 320 patches × 4 tasks × 6 months ≈ 7680 条 `would write` 路径，无异常。

---

### Task 3: 备份旧结果并执行替换

**Files:**
- Modify: `data/haidian/tasks/{task}/v1/results/tiles/`（数据目录，被 .gitignore 忽略）

- [ ] **Step 1: 备份旧 tiles**

```bash
cd /workspace/embedding-api
python - <<'PY'
from pathlib import Path
import shutil, datetime
suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
for task in ['building_extraction','road_extraction','construction','construction_joint','water_extraction']:
    src = Path(f'data/haidian/tasks/{task}/v1/results/tiles')
    if src.exists():
        dst = Path(f'data/haidian/tasks/{task}/v1/results/tiles_backup_{suffix}')
        shutil.copytree(src, dst)
        print('backup', src, '->', dst)
PY
```

- [ ] **Step 2: 删除旧结果（保留备份）**

```bash
cd /workspace/embedding-api
python - <<'PY'
from pathlib import Path
for task in ['building_extraction','road_extraction','construction','construction_joint','water_extraction']:
    p = Path(f'data/haidian/tasks/{task}/v1/results/tiles')
    if p.exists():
        for f in p.glob('patch_*.png'):
            f.unlink()
        print('cleared', p)
PY
```

- [ ] **Step 3: 正式运行转换脚本**

```bash
cd /workspace/embedding-api
python scripts/replace_haidian_results_from_osm_archives.py
```

**Expected:** 无报错；最终每个任务有 `results/{period}/tiles/patch_*.png`（6 期 × 320）以及 `results/tiles/patch_*.png`（默认 202605）。

- [ ] **Step 4: 统计验证**

```bash
cd /workspace/embedding-api
for task in building_extraction road_extraction construction water_extraction; do
  echo -n "$task fallback: "
  ls data/haidian/tasks/$task/v1/results/tiles/*.png 2>/dev/null | wc -l
  echo -n "$task 202605: "
  ls data/haidian/tasks/$task/v1/results/202605/tiles/*.png 2>/dev/null | wc -l
done
```

**Expected:** fallback 和 202605 均为 320。

---

### Task 4: 更新 config.yaml

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: 为 water_extraction 增加 v1 版本配置**

在 `regions.haidian.tasks.water_extraction` 下添加：

```yaml
water_extraction:
  name: 水体提取
  description: 基于海淀 V1 OSM-assisted 诊断图的水体提取
  versions:
    v1:
      results: data/haidian/tasks/water_extraction/v1/results
      predictions: data/haidian/tasks/water_extraction/v1/predictions
      labels: data/haidian/tasks/water_extraction/v1/labels
```

- [ ] **Step 2: 确认/更新 construction 描述**

```yaml
construction:
  name: 施工地检测
  description: 基于海淀 V1 OSM-assisted 诊断图的施工地检测（原 construction_site_extraction）
  versions:
    v1:
      results: data/haidian/tasks/construction/v1/results
      predictions: data/haidian/tasks/construction/v1/predictions
      labels: data/haidian/tasks/construction/v1/labels
```

- [ ] **Step 3: 决定 construction_joint 的处理**

- 若用户选择把 `construction_site_extraction` 同时用于 `construction_joint`：软链或复制 `construction/v1/results` 到 `construction_joint/v1/results`。
- 若用户选择只替换 `construction`，则保留旧 `construction_joint` 或删除其 v1 配置。

本计划默认：**保留 `construction_joint` 旧结果不变**；如需删除，执行：

```bash
cd /workspace/embedding-api
rm -rf data/haidian/tasks/construction_joint/v1/results/tiles
```

---

### Task 5: 重启服务并验证接口

**Files:**
- Read/verify: `app/services/data_service.py`（已有 period 和 per-patch fallback，应兼容新目录）

- [ ] **Step 1: 重启服务**

```bash
cd /workspace/embedding-api
# 停止旧进程
kill $(cat server.pid) 2>/dev/null || true
sleep 2
# 启动
nohup uvicorn app.main:app --host 0.0.0.0 --port 9061 --workers 1 > server.log 2>&1 &
echo $! > server.pid
sleep 3
curl -s http://127.0.0.1:9061/health
```

**Expected:** `{"status":"ok"}`

- [ ] **Step 2: 逐个任务采样请求**

```bash
cd /workspace/embedding-api
for task in building_extraction road_extraction construction water_extraction; do
  curl -s "http://127.0.0.1:9061/regions/haidian/patches/patch_000000/tasks/${task}/result?format=png&before_month=202605&after_month=202605" \
       -o /tmp/hd_sample_${task}.png
  file /tmp/hd_sample_${task}.png
done
```

**Expected:** 4 个文件均为 `PNG image data, 128 x 128, 8-bit/color RGB`。

- [ ] **Step 3: 跑单元测试**

```bash
cd /workspace/embedding-api
pytest -q -m "not slow"
```

**Expected:** `100 passed, 5 deselected`（或等价通过数；若新增 water_extraction 测试需同步更新）。

---

### Task 6: 重新生成全域可视化大图

**Files:**
- Create: `test_output/hd_v2/full_region_*.png`（被 .gitignore 忽略）

- [ ] **Step 1: 运行 mosaic 脚本**

```bash
cd /workspace/embedding-api
python - <<'PY'
import json, pickle, os, requests
from PIL import Image, ImageDraw, ImageFont

# fetch grid mapping from API
pages = []
for p in range(1,5):
    pages.extend(requests.get(f'http://127.0.0.1:9061/regions/haidian/patches?page={p}&page_size=100').json()['patches'])
xs=sorted({p['bounds'][0] for p in pages})
ys=sorted({p['bounds'][1] for p in pages}, reverse=True)
col={x:i for i,x in enumerate(xs)}
row={y:i for i,y in enumerate(ys)}
W=(max(col.values())+1)*128
H=(max(row.values())+1)*128

font_path='/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'
font=ImageFont.truetype(font_path,36)
small=ImageFont.truetype(font_path,24)

tasks=[('building_extraction','海淀区 - 建筑物提取'),
       ('road_extraction','海淀区 - 道路提取'),
       ('construction','海淀区 - 施工地检测'),
       ('water_extraction','海淀区 - 水体提取')]
thumb=1536
canvas=Image.new('RGB',(thumb*2,thumb*2+60),(255,255,255))
draw=ImageDraw.Draw(canvas)
for i,(task,title) in enumerate(tasks):
    big=Image.new('RGB',(W,H),(255,255,255))
    tiles_dir=f'data/haidian/tasks/{task}/v1/results/tiles'
    for pid in [p['patch_id'] for p in pages]:
        c,r=col[[p for p in pages if p['patch_id']==pid][0]['bounds'][0]], row[[p for p in pages if p['patch_id']==pid][0]['bounds'][1]]
        tile=Image.open(f'{tiles_dir}/{pid}.png').convert('RGB')
        big.paste(tile,(c*128,r*128))
    big=big.resize((thumb,thumb),Image.Resampling.NEAREST)
    x=(i%2)*thumb; y=(i//2)*(thumb+30)
    canvas.paste(big,(x,y+30))
    bbox=draw.textbbox((0,0),title,font=font); tw=bbox[2]-bbox[0]
    draw.text((x+(thumb-tw)//2,y),title,fill=(0,0,0),font=font)
footer='320 patches × 128×128 合并 | 空白为无数据区域 | 来源：OSM-assisted 诊断图后处理面板'
bbox=draw.textbbox((0,0),footer,font=small); tw=bbox[2]-bbox[0]
draw.text(((canvas.width-tw)//2,canvas.height-28),footer,fill=(100,100,100),font=small)
os.makedirs('test_output/hd_v2',exist_ok=True)
canvas.save('test_output/hd_v2/full_region_all_tasks.png')
print('saved test_output/hd_v2/full_region_all_tasks.png', canvas.size)
PY
```

- [ ] **Step 2: 读取并检查最终图片**

```bash
file test_output/hd_v2/full_region_all_tasks.png
```

**Expected:** `PNG image data, 3072 x 3132, 8-bit/color RGB`

---

### Task 7: 更新文档与提交

**Files:**
- Modify: `docs/API.md`, `progress.md`
- Modify: `.gitignore` 无变化（test_output 仍忽略）

- [ ] **Step 1: 更新 `docs/API.md`**

在“海淀区 V1 示例”小节中：

```markdown
> **海淀区 V1 示例**（OSM-assisted 诊断图后处理面板，已按月份归档）：
> ```bash
> curl -s "http://60.31.21.42:22065/regions/haidian/patches/patch_000000/tasks/building_extraction/result?format=png&before_month=202605&after_month=202605" -o /tmp/hd_be.png
> curl -s "http://60.31.21.42:22065/regions/haidian/patches/patch_000000/tasks/road_extraction/result?format=png&before_month=202605&after_month=202605" -o /tmp/hd_re.png
> curl -s "http://60.31.21.42:22065/regions/haidian/patches/patch_000000/tasks/construction/result?format=png&before_month=202605&after_month=202605" -o /tmp/hd_con.png
> curl -s "http://60.31.21.42:22065/regions/haidian/patches/patch_000000/tasks/water_extraction/result?format=png&before_month=202605&after_month=202605" -o /tmp/hd_water.png
> ```
> 返回结果为 128×128 PNG，内容取自 OSM-assisted 诊断图的“后处理结果”面板。
```

- [ ] **Step 2: 更新 `progress.md`**

追加一段，记录：
- 替换来源与 6 个月 tar 大小；
- 提取后处理面板并缩放回 128×128；
- 新增 `water_extraction`；
- 默认月份 202605；
- 验证与可视化路径。

- [ ] **Step 3: 提交代码/文档变更**

```bash
cd /workspace/embedding-api
git add config.yaml docs/API.md progress.md scripts/replace_haidian_results_from_osm_archives.py
git commit -m "feat(haidian): replace Haidian V1 results with monthly OSM-assisted diagnostic archives

- Download 6 monthly tar archives (2.1 GB) from ModelScope.
- Extract post-processed panel from 1312x342 diagnostic strips and
  resize to 128x128 to keep API tile contract.
- Map construction_site_extraction -> construction.
- Add water_extraction v1 config and results.
- Default fallback month set to 202605.
- Regenerate full-region mosaic in test_output/hd_v2/.
- Update docs/API.md and progress.md."
git push origin main
```

---

## 回滚方案

如果替换后效果不如预期：

```bash
cd /workspace/embedding-api
# 恢复旧 tiles（假设备份名为 tiles_backup_YYYYMMDD_HHMMSS）
python - <<'PY'
from pathlib import Path
import shutil
for task in ['building_extraction','road_extraction','construction','construction_joint','water_extraction']:
    src = sorted(Path(f'data/haidian/tasks/{task}/v1/results').glob('tiles_backup_*'), reverse=True)
    if src:
        dst = Path(f'data/haidian/tasks/{task}/v1/results/tiles')
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src[0], dst)
        print('restored', task, 'from', src[0])
PY
# 重启服务
cat server.pid | xargs kill
nohup uvicorn app.main:app --host 0.0.0.0 --port 9061 --workers 1 > server.log 2>&1 &
echo $! > server.pid
```

---

## 风险与注意事项

1. **诊断图不是纯模型输出**：图中文字明确“OSM-assisted/oracle 诊断图，不是纯模型输出”。作为 API 结果返回时需向用户说明。
2. **construction_joint**：新归档中无此任务数据，默认保留旧结果；若用户希望删除，请在执行前说明。
3. **磁盘占用**：临时 tar 2.1 GB + 提取后的多期 PNG 约 2–3 GB；`/workspace` 剩余 198 GB，足够。
4. **显存/内存**：本次为文件搬运与图像裁剪，不触发模型推理，峰值内存 < 2 GB。
5. **git 范围**：`data/` 和 `test_output/` 被 `.gitignore` 忽略，只有 `config.yaml`、`docs/API.md`、`progress.md` 和新增脚本会进入 git。
