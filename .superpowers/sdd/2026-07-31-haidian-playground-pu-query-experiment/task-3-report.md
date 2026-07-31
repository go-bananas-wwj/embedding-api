# Task 3 Report - `playground_xuannv` 全量误检实验

## 状态

DONE

> 2026-07-31 范围更新：320 Patch 全域结果仅保留为诊断证据；最终展示与
> 对照实验聚焦于 8 个典型 Patch，并加入真实高分辨率光学纹理边界约束。

## P1 数据质量结论

**三个原训练 Polygon 与高分辨率光学影像对齐后均疑似错标，这是当前结果的
首要根因，`playground_xuannv` 当前不可上线。**

- `patch_000059`：标注主要落在院落和建筑区域，疑似不是操场。
- `patch_000060`：标注位于建筑旁的小块区域，疑似不是操场。
- `patch_000064`：标注落在左侧建成区；真实蓝绿操场位于影像下方，原标注
  疑似错位。

因此，本文中的训练参考指标不能解释为操场识别精度。纹理边界、严格阈值和
面积筛选只用于观察**错误标注模型**的误检风险能否被压制，不是模型质量提升
方案，也不能作为生产部署依据。本任务不重训模型、不修改生产 API。

## 实现

- 新增 `scripts/experiment_playground_xuannv.py`。
- 新增 `tests/test_playground_xuannv_experiment.py`。
- 直接加载注册模型 `model_756ed870` / `playground_xuannv`。
- 直接调用生产 `app.services.pu_query.score_pu_query`，没有复制或重训推理头。
- 从 `logs/request_audit.jsonl` 恢复原始三个训练 Polygon，并栅格化到
  `patch_000059`、`patch_000060`、`patch_000064`。
- 将 OSM `patch_000076` 保持为完全隔离的独立评估样本；该 Patch 未进入
  严格阈值或面积保护参数选择。
- 对海淀 `202604` 全部 320 个 64 维 P10C embedding 同时计算：
  - 生产 Query adaptation 分数；
  - 禁用 Query adaptation 的生产基础分数；
  - checkpoint 原始阈值；
  - 仅由训练 Polygon 校准的严格阈值；
  - 训练 Polygon 校准的双阈值连通扩张和面积保护。
- 所有 Precision、F1、IoU 均明确标注为“相对于不完整参考标签”，未把
  未标注像素当作可靠负样本。

## 冻结参数

- checkpoint 原始阈值：`0.2470571715072547`
- 严格阈值：`1.1431439236419174`
- 面积保护：
  - 高阈值：`1.2267825305461884`
  - 低阈值：`0.5899610471708814`
  - 最小连通域：`4` 像素
  - 最大连通域：`256` 像素
  - 单 Patch 总面积上限：`3%`

参数只使用三个训练 Patch 选择。训练 Polygon 的参考相对召回下限为
`0.8138`，最终面积保护在训练参考上的召回为 `0.8972`。

## 全域结果

### 生产 Query 分数

| 方案 | 非空 Patch | 平均预测面积 | 中位预测面积 | P95 预测面积 |
| --- | ---: | ---: | ---: | ---: |
| 原始阈值 | 309 / 320 | 24.73% | 22.50% | 55.75% |
| 严格阈值 | 186 / 320 | 0.61% | 0.02% | 3.45% |
| 面积保护 | 46 / 320 | 0.17% | 0.00% | 1.31% |

原始阈值产生 17,300 个连通域。`patch_000232` 为 100% 命中，
`patch_000249` 为 99.96%，`patch_000154` 为 79.61%，确认存在严重的
跨 Patch 分数漂移和大面积误检候选。

### Query adaptation 是否是主因

- Query adaptation 在 295 / 320 个 Patch 上被采用。
- 关闭 Query 后，原始阈值平均预测面积从 24.73% 降至 23.78%。
- Query 带来的平均面积增量为 0.96 个百分点，中位增量为 0.61 个百分点，
  P95 增量为 2.96 个百分点。

结论：Query adaptation 会放大误检，但不是根因。即使关闭 Query，原始头
仍在 309 / 320 个 Patch 上产生非空结果，平均面积仍接近四分之一。

## 独立 OSM 操场

`patch_000076` 没有参与任何参数选择：

| 方案 | 已知 OSM Polygon 召回 | Patch 预测面积 |
| --- | ---: | ---: |
| 原始阈值 | 71.65% | 10.14% |
| 严格阈值 | 0% | 0% |
| 面积保护 | 0% | 0% |

独立 OSM Polygon 分数中位数为 `0.4754`，明显低于训练 Polygon 的
`1.0517`；而高面积未标注误检候选连通域的分数中位数为 `0.6340`。
这说明训练操场、独立操场和某些背景地物的分数发生交叠，单一全局高阈值
无法同时保留新样式操场并消除高分背景。

## 原因判断

1. **首要根因：三个原训练 Polygon 疑似将院落、建筑旁小块和建成区标成
   操场，模型学习目标本身可能错误。**
2. 当前头只有三个 Polygon，且没有可信操场正标注覆盖不同外观。
3. 生产 PU 训练只从前景相似度最低 30% 的未标注像素中构建背景原型，
   主要学到“容易背景”，缺少屋顶、道路、裸地等高相似度难负样本。
4. 支持 Patch 统计量用于全域标准化，跨 Patch 光谱和场景变化导致分数基线漂移。
5. Query adaptation 是次要放大因素，不是误检的决定性来源。

## 建议

现有纹理和面积保护只适合做“错误标注模型的误检风险控制”对照，不能直接
替换生产结果，更不能上线。旧面积保护在独立操场上出现 0 召回；相对种子
方案即使恢复部分 OSM 区域，也不能消除训练目标疑似错误这一前提问题。

若后续允许修改训练逻辑，优先方案是：

1. 用户补充少量不同外观、不同 Patch 的操场正标注；
2. 引入人工确认的难负样本或交互式误检反馈，不能自动把全部未标注区当负类；
3. 使用 Patch 内相对分数、候选连通域排序和“无目标 Patch”拒识分支，而不是
   继续提高一个全域固定阈值；
4. 在至少多个独立操场上冻结参数后再考虑上线。

## 输出

- `Tmp/playground_pu_query_20260731/experiment_manifest.json`
- `Tmp/playground_pu_query_20260731/metrics.json`
- `Tmp/playground_pu_query_20260731/score_groups.json`
- `Tmp/playground_pu_query_20260731/training_annotations.geojson`
- `Tmp/playground_pu_query_20260731/arrays/*.npy`

已保存训练 Patch、独立 OSM Patch、三个指定高误检 Patch、动态高面积 Patch
和八个固定随机 Patch 的连续分数、原始/严格/面积保护掩膜与参考标签数组。

## 测试

```text
9 passed in 5.42s
```

验证命令：

```bash
/opt/conda/envs/pyseims/bin/python -m pytest \
  tests/test_playground_xuannv_experiment.py -q

/opt/conda/envs/pyseims/bin/python scripts/experiment_playground_xuannv.py \
  --month 202604 \
  --output Tmp/playground_pu_query_20260731
```

## 聚焦纹理实验补充

最终页面只展示原训练标注 `patch_000059`、`patch_000060`、
`patch_000064`、
独立 OSM `patch_000076`，以及高误检代表 `patch_000232`、
`patch_000249`、`patch_000154`、`patch_000139`。选择依据、中心坐标和
基线预测面积均保存在聚焦实验 manifest 中。

实验使用真实 202604 高分辨率光学 RGB 的灰度梯度、通道梯度和局部纹理
构造边界，没有使用建筑、道路、操场掩膜或其他类别先验。公平对照使用相同
的 Patch 内 P99 高置信种子和面积筛选参数，纹理方案只额外限制区域扩张
跨越强纹理边界。由于三个训练 Polygon 疑似错标，这些方案仅用于错误标注
模型上的风险抑制对照，不代表操场识别效果，不可上线。

独立 OSM Patch 上，相对种子加面积筛选的参考相对 F1 为 `0.4023`，
纹理边界加面积筛选为 `0.4039`，Recall 均为 `0.5309`。高误检代表的
平均预测面积均由基线的 `81.47%` 降为 `0%`。因此本次结论是：
**纹理边界未证明有实质增益，主要改善来自 Patch 内相对种子阈值。**

聚焦输出：

- `Tmp/playground_texture_20260731/experiment_manifest.json`
- `Tmp/playground_texture_20260731/metrics.json`
- `Tmp/playground_texture_20260731/arrays/*.npy`
- `Tmp/playground_texture_20260731/index.html`
- `Tmp/playground_texture_20260731/assets/*`

最终验证：聚焦测试 `8 passed`，完整测试 `334 passed`。
