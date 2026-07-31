# GitHub 与 ModelScope 复现文档设计

## 目标

让第一次接触项目的人能够根据自己的目的选择最短路径：

1. 只阅读代码和接口设计；
2. 在已有模型与数据的机器上启动服务；
3. 从 ModelScope 私有备份完整恢复代码、环境、模型和数据；
4. GitHub 不可用时，仅依赖 ModelScope 恢复稳定版本。

## 文档分工

### GitHub `README.md`

GitHub 首页保持简洁，负责回答：

- 这个项目做什么；
- 当前稳定版本是什么；
- 在线服务和文档在哪里；
- 如何快速启动已有部署；
- 如何选择轻量安装或完整灾备恢复；
- 更详细的接口、训练、运维和复现文档在哪里。

首页不展开几十 GB 数据的逐文件恢复步骤，避免主要信息被淹没。

### GitHub `docs/REPRODUCTION.md`

完整复现手册负责回答：

- 推荐的硬件、操作系统、Python、CUDA 和磁盘要求；
- 如何按稳定 tag 获取代码；
- 如何从 ModelScope 下载指定稳定版本；
- 如何验证 `manifest.json` 和 SHA256；
- 如何恢复打包的 Conda 环境，或从环境清单重建；
- 如何恢复模型、数据、区域大图；
- 配置、启动、测试和接口验收；
- 常见错误与排查顺序。

### ModelScope 根 `README.md`

ModelScope 数据集首页面向灾备恢复，负责回答：

- 备份与 GitHub commit/tag 的对应关系；
- 私有仓库访问方式和 Token 安全要求；
- 目录、文件类别、体积和用途；
- 完整下载与按需下载的区别；
- 分卷文件的合并和解压规则；
- 恢复顺序及验收命令；
- GitHub 不可用时如何从 `repository.bundle` 恢复。

### ModelScope 稳定版本目录 `README.md`

版本目录 README 固定描述该版本的 commit、tag、备份统计、目录结构和恢复入口，
避免未来新增备份后根 README 与旧版本内容混淆。

## 安全与准确性

- 文档不保存 ModelScope Token、密码、`.env` 或服务器私密配置。
- Token 只通过 `MODELSCOPE_API_TOKEN` 临时环境变量传入。
- 所有路径、文件名、commit、tag 和体积以实际稳定备份为准。
- 不承诺 `requirements.txt` 能完整替代锁定环境；优先推荐打包环境或 Conda
  explicit 清单。
- 明确 ModelScope 数据集是私有仓库，读者必须先获得权限。

## 验证标准

- README 中引用的本地文档和脚本全部存在。
- Markdown 链接无失效的仓库内相对路径。
- Shell 示例通过语法检查，关键命令与脚本当前参数一致。
- GitHub 推送后 `main` 能看到更新内容。
- ModelScope 根 README 和版本 README 均能从远端读取到新内容。
- 文档修改不重启、不停止当前 API 和 Watchdog。
