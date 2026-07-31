# Reproduction README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 GitHub 和 ModelScope 提供清晰、可执行、彼此分工明确的中文阅读与完整复现说明。

**Architecture:** GitHub 首页作为简洁入口，完整流程下沉到 `docs/REPRODUCTION.md`；ModelScope 根 README 作为灾备总入口，稳定版本目录 README 固定记录该版本恢复信息。两边共同引用同一个 Git commit/tag 和备份清单。

**Tech Stack:** Markdown、Git、ModelScope Hub SDK、Conda、zstd、pytest、FastAPI/Uvicorn

## Global Constraints

- 不把 Token、密码、`.env` 或私密服务器配置写入文档或 Git。
- 当前稳定版本固定为 `embedding-api-20260730-stable` 和 commit `22921a71569c2ee6f03dc0e27e67cc51339d58ba`。
- ModelScope 私有数据集固定为 `WeijieWu/xuannv_embdding_backup`。
- 文档更新不得中断当前 API 和 Watchdog。

---

### Task 1: GitHub 阅读入口与复现手册

**Files:**
- Modify: `README.md`
- Create: `docs/REPRODUCTION.md`
- Modify: `docs/BACKUP_AND_RESTORE.md`

**Interfaces:**
- Consumes: 当前仓库目录、环境清单、备份 manifest 约定和 Watchdog 命令。
- Produces: 面向普通读者的首页及面向部署人员的完整复现流程。

- [ ] **Step 1: 重写根 README**

保留项目定位、在线入口、能力、快速启动和文档导航；增加稳定版本、三种阅读/复现路径和硬件空间提示。

- [ ] **Step 2: 编写完整复现手册**

覆盖 GitHub 获取代码、ModelScope 获取资产、校验、环境恢复、分卷解压、配置、启动、测试和故障排查。

- [ ] **Step 3: 对齐备份恢复文档**

让 `docs/BACKUP_AND_RESTORE.md` 指向完整复现手册，并保留备份维护者需要的安全约束。

- [ ] **Step 4: 校验本地链接与命令**

运行 Markdown 相对链接检查、Shell 代码块语法检查、备份校验脚本帮助/错误路径测试。

### Task 2: ModelScope 数据集说明

**Files:**
- Create temporarily: `backup-readme-staging/README.md`
- Create temporarily: `backup-readme-staging/release-README.md`

**Interfaces:**
- Consumes: ModelScope 稳定版本文件清单、Git commit/tag 和 GitHub 复现手册。
- Produces: 数据集根 README 与 `embedding-api-20260730-stable/README.md`。

- [ ] **Step 1: 生成 ModelScope 根 README**

写明私有访问、备份结构、57.02 GiB 正式载荷、按需下载、完整恢复和安全要求。

- [ ] **Step 2: 生成稳定版本 README**

写明该版本的 40 个远端文件、36 个正式载荷文件、commit/tag、各类别体积和恢复顺序。

- [ ] **Step 3: 上传两份 README**

使用临时进程环境变量传入 Token，分别上传到数据集根目录和稳定版本目录。

- [ ] **Step 4: 远端回读核验**

通过 ModelScope Hub API 回读文件，检查标题、commit、tag、恢复命令和 SHA256 说明。

- [ ] **Step 5: 删除临时文件**

删除 `backup-readme-staging/`，不触碰后端使用的模型、数据和环境。

### Task 3: 发布与运行状态验收

**Files:**
- Modify: Git index only for documentation files

**Interfaces:**
- Consumes: 已验证的 GitHub 与 ModelScope README。
- Produces: GitHub `main` 上的新文档提交和保持健康的线上服务。

- [ ] **Step 1: 运行文档与项目测试**

运行链接检查和相关测试；确认工作区没有凭据。

- [ ] **Step 2: 提交并推送**

只提交 README、复现手册、设计和计划，不提交运行日志。

- [ ] **Step 3: 最终服务检查**

确认 `/health`、`/openapi.json`、Watchdog 和 Git 远端 commit 正常。
