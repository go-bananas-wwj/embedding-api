# Harbin ModelScope Asset Pipeline

本目录包含把哈尔滨新区静态资产上传到 ModelScope，以及后续从 ModelScope
下载这些资产的工具脚本。

## 为什么用 tar 归档？

ModelScope 数据集对单个目录树下的文件/目录总数有限制（约 50,000 个）。
哈尔滨新区共有约 15 万个静态文件，因此上传前把它们按类别打包成少量 tar
归档，下载后再自动解压。

## 资产归档

| 归档文件名 | 本地源路径 | 说明 |
|------------|------------|------|
| `data_harbin.tar` | `data/harbin` | embedding、任务结果/预测/标签、patches_meta |
| `models_harbin.tar` | `models/harbin` | 哈尔滨系统模型 checkpoint |
| `models_sam3.tar` | `models/sam3` | SAM3 交互式分割权重 |
| `raw_harbin.tar` | `/workspace/raw/harbin` | mosaic 大图接口使用的 S2/S1/Landsat 原始 TIFF |
| `raw_harbin_scenes.tar` | `/workspace/raw/harbin_scenes` | SAM3/embedding 使用的多时相 S2 场景 |

ModelScope 数据集地址：`https://www.modelscope.cn/datasets/WeijieWu/xuannv_embdding_api`  
数据集内前缀：`harbin/v1/api_ready`

## 打包上传（维护者）

```bash
# 1. 先 dry run 查看文件数量和总体积
python pipelines/harbin/prepare_harbin_api_assets.py \
  --output-root /workspace/modelscope_upload/harbin/v1 \
  --dry-run

# 2. 生成 tar 归档（默认输出到 /workspace/modelscope_upload/harbin/v1/api_ready）
python pipelines/harbin/prepare_harbin_api_assets.py \
  --output-root /workspace/modelscope_upload/harbin/v1

# 3. 上传到 ModelScope（Token 从环境变量读取，不要写死在命令历史中）
export MODELSCOPE_TOKEN="..."
modelscope upload --repo-type dataset --token "$MODELSCOPE_TOKEN" \
  --max-workers 8 \
  --commit-message "Add Harbin V1 static assets" \
  WeijieWu/xuannv_embdding_api \
  /workspace/modelscope_upload/harbin/v1/api_ready \
  harbin/v1/api_ready
```

> 如果只需要上传不带原始卫星场景的小包，可加上 `--skip-raw-scenes`。

## 下载部署（使用者）

```bash
export MODELSCOPE_TOKEN="..."  # 私有数据集需要
python pipelines/harbin/download_modelscope_assets.py \
  --repo WeijieWu/xuannv_embdding_api \
  --prefix harbin/v1/api_ready \
  --target . \
  --verify-checksums
```

脚本会把归档下载到 `.modelscope_cache/harbin_v1`，校验 `checksums.sha256` 后
自动解压：`data/harbin`、`models/harbin`、`models/sam3` 放到项目根目录，
原始卫星场景放到真实的 `/workspace/raw/...` 路径，与 `config.yaml` 中的绝对
路径保持一致。

## 校验

上传包中生成 `manifest.json` 与 `checksums.sha256`。下载时加 `--verify-checksums`
可对归档做完整性校验后再解压。
