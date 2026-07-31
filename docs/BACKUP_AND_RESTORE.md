# 稳定版本备份与恢复

> 新机器的逐步恢复命令请直接阅读
> [`docs/REPRODUCTION.md`](REPRODUCTION.md)。本文主要说明备份维护规则。

本项目使用两处互补备份：

- GitHub 保存源码、配置、文档、测试和环境清单。
- ModelScope 私有数据集保存同版本源码副本、模型、数据、环境包和区域大图。

每次稳定版备份必须在 `manifest.json` 中记录 Git commit、Git tag、文件大小和
SHA256。ModelScope 中的源码归档与 Git bundle 必须由该 commit 生成，不能包含
未提交文件。

## 恢复顺序

1. 从 GitHub 按稳定版 tag 克隆代码；GitHub 不可用时，从 ModelScope 的
   `source/repository.bundle` 恢复。
2. 下载 ModelScope 对应版本目录，并运行：

   ```bash
   python scripts/verify_backup_manifest.py /path/to/release
   ```

3. 按归档内的相对路径解压 `models` 和 `data`。
4. 优先使用 `environment/pyseims-environment.tar.zst` 恢复完整环境；也可以用
   `deploy/environment/conda-linux-64-explicit.txt` 重建。
5. 设置 Rasterio 使用的 `PROJ_DATA`，运行 `pytest -q`。
6. 使用 `python service_watchdog.py start` 启动服务，确认 `/health`、`/regions`、
   `/openapi.json` 和 `/docs` 均返回 HTTP 200。

当前稳定备份：

| 项目 | 值 |
|---|---|
| Git tag | `embedding-api-20260730-stable` |
| Git commit | `22921a71569c2ee6f03dc0e27e67cc51339d58ba` |
| ModelScope 数据集 | `WeijieWu/xuannv_embdding_backup` |
| 版本目录 | `embedding-api-20260730-stable/` |

## 安全约束

- ModelScope token 只能通过临时进程环境变量传入。
- token、`.env`、日志、PID、缓存和临时训练文件不得进入任何归档。
- 上传临时分片只能放在 `backup-staging/`，远端校验成功后删除。
- 不删除或移动正在被 API 使用的原始模型和数据。

## 环境文件

`deploy/environment/` 保存本次运行环境的多种描述：

- `conda-linux-64-explicit.txt`：最精确的 Conda 包地址和构建版本。
- `conda-environment.yml`：便于人工查看和跨机器重建。
- `pip-freeze.txt`：Python 包快照。
- `runtime-info.txt`：Python、PyTorch、CUDA 和关键库版本。
- `gpu-info.txt`：制作快照时的 GPU 和驱动信息。

这些文件用于恢复和审计，不包含凭据。
