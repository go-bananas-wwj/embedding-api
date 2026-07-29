# 海淀 AEF 2025 Embedding PCA 可视化 API 实施计划

## 1. 最终需求

新增一个只面向海淀区的 AEF 可视化接口：

- 数据使用本地已有的 AEF 2025 年年度 embedding
- 调用方只提交一个 `patch_id`
- 后端读取该 Patch 的 AEF embedding
- 返回 PCA 三通道彩色可视化 PNG
- 不返回原始 `.npy`
- 不支持哈尔滨
- 不支持其他年份
- 不新增独立 TCP 服务，继续使用现有 API 端口 `9061`

现有数据：

- 目录：`data/external_embeddings/aef/haidian/2025`
- 数量：320 个 Patch
- 总大小：约 642 MB
- 文件名示例：`patch_000106.npy`
- AEF 是 2025 年年度特征，不是月度特征

## 2. API 设计

### 2.1 接口路径

```http
GET /regions/haidian/patches/{patch_id}/embeddings/aef/pca
```

示例：

```http
GET /regions/haidian/patches/patch_000106/embeddings/aef/pca
```

这里把 `haidian` 固定在路径中，不提供 `region_id` 参数，避免前端误以为哈尔滨也支持。

### 2.2 参数

| 参数 | 位置 | 必填 | 格式 | 示例 | 说明 |
|---|---|---:|---|---|---|
| `patch_id` | path | 是 | `patch_` 加六位数字 | `patch_000106` | 要查看的海淀 Patch |

不提供以下参数：

- `region_id`
- `year`
- `month`
- `version`
- `format`

原因是数据范围已经固定为：

```text
region = haidian
source = AEF
year = 2025
format = PCA PNG
```

### 2.3 成功响应

```http
HTTP/1.1 200 OK
Content-Type: image/png
Content-Disposition: inline; filename="haidian_patch_000106_aef_2025_pca.png"
Cache-Control: public, max-age=86400
ETag: "..."
X-Embedding-Source: AEF
X-Embedding-Year: 2025
X-Patch-Id: patch_000106
X-PCA-Version: aef-haidian-2025-global-v1
```

浏览器可以直接显示，前端可直接使用：

```html
<img src="/regions/haidian/patches/patch_000106/embeddings/aef/pca">
```

## 3. PCA 可视化方法

### 3.1 不对每个 Patch 独立拟合 PCA

如果每个 Patch 分别拟合 PCA：

- 同一种颜色在不同 Patch 中不代表同一种特征方向
- Patch 之间无法比较
- 区域拼图会产生明显色差和割裂感

因此不采用逐 Patch PCA。

### 3.2 使用海淀全域统一 PCA

离线扫描 320 个 AEF embedding：

1. 从每个 Patch 均匀随机采样固定数量像素。
2. 合并采样特征。
3. 使用固定随机种子拟合一个全域三维 PCA。
4. 保存 PCA 均值与三个主成分。
5. 计算全域统一的每通道显示分位数。
6. 所有 Patch 使用同一套 PCA 参数和颜色范围。

输出含义：

- 红、绿、蓝分别对应全域 PCA 的前三个主成分
- 相同颜色在不同 Patch 中具有一致的相对特征含义
- Patch 之间可横向比较

### 3.3 归一化

每个 PCA 通道使用全域统一的稳健分位数，例如：

```text
low = 2%
high = 98%
```

再映射到 `0~255`。

不对单个 Patch 独立拉伸，避免颜色漂移。

### 3.4 PCA 模型文件

新增：

```text
models/haidian/aef/2025/pca_global_v1.npz
```

内容：

- `mean`
- `components`
- `display_low`
- `display_high`
- `sample_count`
- `random_seed`
- `source_year`
- `source_patch_count`

同时生成：

```text
models/haidian/aef/2025/pca_global_v1.json
```

用于人类可读地记录训练来源、维度、样本数和生成时间。

## 4. PNG 缓存

PCA 结果是静态年度产品，不需要每次请求重复计算。

缓存目录：

```text
data/visualization_cache/aef/haidian/2025/pca_global_v1/
```

文件示例：

```text
patch_000106.png
```

处理流程：

1. 请求到达。
2. 校验 `patch_id`。
3. 检查对应 AEF `.npy` 是否存在。
4. 如果 PNG 缓存存在，直接返回。
5. 如果不存在，使用全域 PCA 参数生成 PNG。
6. 使用临时文件写入后原子重命名，避免并发产生半张图片。
7. 返回 PNG。

可在部署时预生成全部 320 张 PNG，使第一次访问也无需等待。

## 5. 后端代码调整

### 5.1 新增 PCA 服务

新增：

```text
app/services/aef_pca_service.py
```

职责：

- 固定读取海淀 AEF 2025 目录
- 校验 Patch ID
- 安全加载 `.npy`
- 加载全域 PCA 参数
- 投影为三个通道
- 使用统一分位数映射 RGB
- 生成并缓存 PNG
- 为并发生成提供文件锁

### 5.2 新增路由

新增：

```text
app/routers/aef_embeddings.py
```

只包含：

```text
GET /regions/haidian/patches/{patch_id}/embeddings/aef/pca
```

在 `app/main.py` 中注册 Swagger 分组：

```text
AEF 可视化
```

### 5.3 新增离线 PCA 脚本

新增：

```text
scripts/fit_haidian_aef_2025_pca.py
```

功能：

- 扫描本地 320 个 AEF 文件
- 校验 shape 和 dtype
- 拟合全域 PCA
- 保存 PCA 参数
- 输出数据质量报告

### 5.4 新增批量缓存脚本

新增：

```text
scripts/precompute_haidian_aef_2025_pca.py
```

功能：

- 读取全域 PCA 参数
- 为 320 个 Patch 生成 PNG
- 跳过已经生成且版本一致的文件
- 输出成功、失败和缺失 Patch 清单

## 6. 输入校验与错误响应

### 6.1 Patch ID 校验

只允许：

```regex
^patch_\d{6}$
```

非法示例：

- `000106`
- `patch_106`
- `../patch_000106`
- `/etc/passwd`

### 6.2 错误响应

`404`：

```json
{
  "detail": "AEF 2025 embedding not found for Haidian patch 'patch_999999'"
}
```

`422`：

```json
{
  "detail": "Invalid patch_id. Use format patch_000000"
}
```

`500`：

```json
{
  "detail": "AEF embedding exists but could not be visualized"
}
```

服务日志记录真实内部错误，但响应不暴露服务器文件路径。

## 7. Swagger UI

Swagger 文档明确写出：

- 仅支持海淀区
- 数据固定为 AEF 2025 年年度 embedding
- AEF 不是月度数据
- 返回 PCA 彩色 PNG
- 相同颜色在不同 Patch 中使用同一个全域 PCA 基准
- `patch_id` 应填写什么

默认示例：

```text
patch_000106
```

Response 文档：

- `200 image/png`
- `404` Patch 没有 AEF 数据
- `422` Patch ID 格式错误
- `500` 文件损坏或 PCA 生成失败

## 8. 测试计划

### 8.1 PCA 单元测试

- 全域 PCA 输出恰好三个通道
- 同一个 Patch 重复生成结果完全一致
- 固定随机种子时 PCA 参数可复现
- 使用的是全域分位数，不是单 Patch 分位数
- 输出像素范围为 `0~255`
- 输入含 NaN/Inf 时按既定规则处理或拒绝

### 8.2 路由测试

- `patch_000106` 返回 `200 image/png`
- PNG 能被 Pillow 正常打开
- 响应头包含 AEF、2025、Patch ID 和 PCA 版本
- 非法 Patch ID 返回 422
- 不存在 Patch 返回 404
- 不支持通过参数切换区域、年份或原始格式

### 8.3 全量数据检查

对 320 个本地文件检查：

- 文件能安全加载
- shape 一致
- dtype 合法
- 没有 object/pickle 数组
- 没有异常空数组
- 记录 NaN/Inf 数量

### 8.4 视觉检查

随机选择至少 12 个 Patch：

- 建筑密集区
- 道路密集区
- 水体
- 山地与植被
- 城乡交界

生成 HTML 对照画廊，检查：

- 是否存在黑图或全白图
- 是否有明显 Patch 颜色漂移
- PCA 结构是否与光学影像空间位置一致
- 相邻 Patch 颜色是否连续

### 8.5 并发测试

- 10 个客户端同时请求同一未缓存 Patch
- 最终只产生一个完整 PNG
- 不出现损坏图片
- API 健康检查仍可响应

## 9. 部署步骤

1. 写失败测试，固定 API 契约。
2. 检查全部 320 个 AEF 文件。
3. 实现全域 PCA 拟合脚本。
4. 拟合并保存 `pca_global_v1`。
5. 实现 PCA 服务和 PNG 缓存。
6. 实现固定海淀路由。
7. 完善中文 Swagger。
8. 预生成全部 320 张 PNG。
9. 生成视觉审查画廊。
10. 运行单元测试和 API 回归测试。
11. 通过 Watchdog 重启 `9061`。
12. 验证本地接口与公网映射 `22065`。

## 10. 验收标准

- 只支持海淀区
- 只使用本地 AEF 2025 年年度 embedding
- 调用方只需提供 `patch_id`
- 返回浏览器可直接显示的 PCA PNG
- 不返回或暴露原始 `.npy`
- 320 个有效 Patch 均可访问
- 所有 Patch 使用同一套全域 PCA 与颜色范围
- 缓存命中时快速返回
- Swagger 中文说明清晰
- 现有 P10C、任务、自定义训练和 SAM3 接口不受影响
