# 完整复现手册

本文说明如何在一台新 Linux 机器上恢复
`embedding-api-20260730-stable`，包括源码、Python 环境、模型、数据和区域大图。

## 1. 稳定版本

| 项目 | 值 |
|---|---|
| GitHub | `go-bananas-wwj/embedding-api` |
| Git tag | `embedding-api-20260730-stable` |
| Git commit | `22921a71569c2ee6f03dc0e27e67cc51339d58ba` |
| ModelScope | `WeijieWu/xuannv_embdding_backup` |
| ModelScope 版本目录 | `embedding-api-20260730-stable/` |
| 正式载荷 | 36 个文件，57.02 GiB |

ModelScope 数据集是私有仓库。下载前必须由仓库所有者授予访问权限。

## 2. 机器要求

制作稳定快照时的环境：

- Linux x86_64
- Python 3.9.18
- PyTorch 2.8.0 + CUDA 12.8
- NVIDIA 驱动 580.159.03
- RTX 4090 24 GB

建议至少准备：

| 资源 | 最低建议 | 说明 |
|---|---:|---|
| 磁盘可用空间 | 150 GB | 同时容纳下载包、解压文件和运行缓存 |
| 内存 | 32 GB | 大图与批量推理建议 64 GB 或更多 |
| NVIDIA 显存 | 16 GB | SAM3 和部分基座模型需要 GPU |
| CPU | 8 核 | 解压、数据读取和普通 API 请求 |

只阅读代码或运行不加载 GPU 模型的接口时，可以使用 CPU；完整功能复现建议使用
NVIDIA GPU。

基础工具：

```bash
sudo apt-get update
sudo apt-get install -y git curl zstd
```

还需要安装 Miniconda 或 Anaconda。

## 3. 获取源码

### 方式 A：从 GitHub 获取，推荐

```bash
git clone https://github.com/go-bananas-wwj/embedding-api.git
cd embedding-api
git checkout embedding-api-20260730-stable
git rev-parse HEAD
```

最后一条命令应输出：

```text
22921a71569c2ee6f03dc0e27e67cc51339d58ba
```

### 方式 B：GitHub 不可用时从 Git bundle 恢复

先从 ModelScope 下载：

```text
embedding-api-20260730-stable/source/repository.bundle
```

再恢复仓库：

```bash
git clone repository.bundle embedding-api
cd embedding-api
git checkout embedding-api-20260730-stable
git bundle verify ../repository.bundle
```

`source/source.tar.zst` 是不依赖 Git 的源码快照，适合审计或应急读取；正常开发优先
使用 GitHub 或 `repository.bundle`。

## 4. 下载 ModelScope 备份

不要把 Token 写入脚本或文档。推荐在当前终端临时设置：

```bash
read -rsp "ModelScope Token: " MODELSCOPE_API_TOKEN
echo
export MODELSCOPE_API_TOKEN
```

安装 ModelScope CLI：

```bash
python -m pip install "modelscope==1.37.1"
```

完整下载稳定版本：

```bash
mkdir -p /srv/xuannv-backup
modelscope download \
  --dataset WeijieWu/xuannv_embdding_backup \
  --token "$MODELSCOPE_API_TOKEN" \
  --include "embedding-api-20260730-stable/**" \
  --local_dir /srv/xuannv-backup
```

下载后的版本根目录是：

```bash
export RELEASE_DIR=/srv/xuannv-backup/embedding-api-20260730-stable
```

空间有限时可以使用多个 `--include` 分别下载：

```text
embedding-api-20260730-stable/source/**
embedding-api-20260730-stable/environment/**
embedding-api-20260730-stable/models/**
embedding-api-20260730-stable/data/**
embedding-api-20260730-stable/mosaics/**
embedding-api-20260730-stable/restore/**
embedding-api-20260730-stable/manifest.json
```

只下载源码不足以运行完整服务。至少还需要环境、所调用区域的数据和对应模型。

## 5. 下载后先校验

在项目目录执行：

```bash
python scripts/verify_backup_manifest.py "$RELEASE_DIR"
```

成功时输出：

```text
backup verification passed
```

也可以使用备份自带的独立脚本：

```bash
python "$RELEASE_DIR/restore/verify_backup_manifest.py" "$RELEASE_DIR"
```

出现 `missing`、`size mismatch` 或 `sha256 mismatch` 时不要继续解压，应重新下载
对应文件。`SHA256SUMS` 可用于人工审计；`manifest.json` 是自动校验的权威清单。

## 6. 恢复 Python 环境

### 方式 A：恢复打包环境，最接近原服务器

```bash
sudo mkdir -p /opt/conda/envs/pyseims
sudo tar --use-compress-program=unzstd \
  -xf "$RELEASE_DIR/environment/pyseims-environment.tar.zst" \
  -C /opt/conda/envs/pyseims
sudo /opt/conda/envs/pyseims/bin/conda-unpack
source /opt/conda/envs/pyseims/bin/activate
```

如果没有 `/opt/conda` 写权限，可以恢复到用户目录：

```bash
mkdir -p "$HOME/.conda/envs/pyseims"
tar --use-compress-program=unzstd \
  -xf "$RELEASE_DIR/environment/pyseims-environment.tar.zst" \
  -C "$HOME/.conda/envs/pyseims"
"$HOME/.conda/envs/pyseims/bin/conda-unpack"
source "$HOME/.conda/envs/pyseims/bin/activate"
```

### 方式 B：从 Conda 清单重建

```bash
conda create -n pyseims \
  --file deploy/environment/conda-linux-64-explicit.txt
conda activate pyseims
```

### 方式 C：只搭建开发环境

```bash
conda create -n embedding-api python=3.9 -y
conda activate embedding-api
python -m pip install -r requirements.txt
```

方式 C 适合阅读代码和普通开发，不保证与稳定服务器逐包一致。生产复现优先使用
方式 A，其次使用方式 B。

确认环境：

```bash
python - <<'PY'
import fastapi
import rasterio
import torch

print("Python/Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("FastAPI:", fastapi.__version__)
print("Rasterio:", rasterio.__version__)
PY
```

## 7. 恢复模型与数据

以下命令都在项目根目录执行。普通 `.tar.zst` 直接解压；带 `.part-*` 的文件先按
文件名排序合并。

定义辅助函数：

```bash
extract_zst() {
  local archive="$1"
  tar --use-compress-program=unzstd -xf "$archive" -C .
}

join_and_extract() {
  local prefix="$1"
  local output="/tmp/$(basename "$prefix").tar.zst"
  cat "${prefix}".part-* > "$output"
  tar --use-compress-program=unzstd -xf "$output" -C .
  rm -f "$output"
}
```

恢复模型：

```bash
extract_zst "$RELEASE_DIR/models/sam3.tar.zst"
extract_zst "$RELEASE_DIR/models/haidian.tar.zst"
extract_zst "$RELEASE_DIR/models/harbin.tar.zst"
extract_zst "$RELEASE_DIR/models/dinov3-sat493m.tar.zst"
```

恢复区域数据和缓存：

```bash
join_and_extract "$RELEASE_DIR/data/haidian.tar.zst"
join_and_extract "$RELEASE_DIR/data/harbin.tar.zst"
join_and_extract "$RELEASE_DIR/data/external-embeddings.tar.zst"
join_and_extract "$RELEASE_DIR/data/feature-cache.tar.zst"
join_and_extract "$RELEASE_DIR/data/visualization-cache.tar.zst"
```

分卷必须按 `part-000`、`part-001`、`part-002` 的顺序完整合并，不能单独解压某一
分卷。解压后应在项目根目录看到：

```text
models/sam3/
models/haidian/
models/harbin/
models/dinov3_sat493m/
data/haidian/
data/harbin/
```

区域大图是前端静态交付物，不是启动 API 的硬性依赖：

```bash
mkdir -p static-mosaics
unzip "$RELEASE_DIR/mosaics/regional-mosaics.zip" -d static-mosaics
```

## 8. 检查配置

默认服务读取项目根目录的 `config.yaml`。恢复后重点检查：

```bash
test -f config.yaml
test -f models/sam3/sam3.pt
test -f models/haidian/v1/task_heads/building_conv3x3_best.pt
test -f data/haidian/patches_meta_v1.json
test -f data/harbin/patches_meta.json
```

`config.yaml` 中少量哈尔滨原始影像目录可能是部署机绝对路径。如果新机器目录不同，
请修改 `regions.harbin.*_dir`，或将数据挂载到配置中的路径。不要把本机 Token 或
密码写入该文件。

DINOv3 模型放在非默认位置时：

```bash
export DINOV3_SAT493M_MODEL_DIR=/absolute/path/to/dinov3_sat493m
```

Rasterio/PROJ 报数据库路径错误时：

```bash
export PROJ_DATA="$(python - <<'PY'
import pyproj
print(pyproj.datadir.get_data_dir())
PY
)"
```

## 9. 启动与验收

先运行测试：

```bash
python -m pytest tests -q
```

该稳定版本制作时的结果为：

```text
287 passed
```

启动带自动拉起能力的服务：

```bash
python service_watchdog.py start
python service_watchdog.py status
```

验收基础入口：

```bash
curl -fsS http://127.0.0.1:9061/health
curl -fsS http://127.0.0.1:9061/regions
curl -fsS http://127.0.0.1:9061/openapi.json >/dev/null
```

验收海淀 Embedding：

```bash
curl -fsS \
  "http://127.0.0.1:9061/regions/haidian/patches/patch_000000/embedding?format=json&version=v1&month=202512" \
  >/tmp/haidian-embedding.json
```

浏览器打开：

```text
http://127.0.0.1:9061/docs
http://127.0.0.1:9061/logs
```

## 10. 常见问题

### API 可以启动，但接口返回 `not found`

通常是只恢复了源码，没有恢复对应的 `models/` 或 `data/`；也可能是月份、区域、
Patch ID 与该区域可用范围不匹配。先查看 `/regions` 和 Swagger 参数说明。

### SAM3 显存不足

确认 GPU 显存没有被其他进程占满，并检查 `config.yaml` 中 `sam3.device`。SAM3
首次加载会占用较多显存。

### `conda-unpack` 不存在

说明环境归档没有完整解压，或使用了错误目录。重新校验环境包并解压，不要改用
系统 Python 执行该命令。

### ModelScope 下载中断

保留已经下载的目录，重新执行同一条 `modelscope download`。完成后必须重新运行
manifest 校验。

### 测试出现 scikit-learn `InconsistentVersionWarning`

部分历史系统模型由 scikit-learn 1.8.0 保存，而稳定运行环境使用 1.6.1，因此加载
时会出现兼容性警告。当前稳定版本的完整测试仍然通过。复现时应保留打包环境，不要
随意升级或降级 scikit-learn；如果推理结果异常，再用原训练版本重新导出对应模型。

### 公网地址在服务器本机打不开

公网端口可能不支持网络回环。本机健康检查使用 `127.0.0.1:9061`；外部前端使用
部署方提供的公网映射地址。

## 11. 复现完成标准

满足以下条件才算完整复现：

- Git commit 与稳定版本记录一致；
- manifest 校验通过；
- 测试通过，或已记录与硬件相关的跳过项；
- Watchdog 显示 `running`，API 显示 `healthy`；
- `/health`、`/regions`、`/openapi.json` 返回成功；
- 至少成功读取一个海淀 Embedding；
- 实际需要使用的 SAM3、系统模型和自定义模型接口通过业务样例验证。
