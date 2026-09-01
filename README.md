# LatentGeometry-GWR

Research repository for **Latent-Geometry Geographically Weighted Regression (LG-GWR)**.

本仓库用于 LG-GWR 方法论文的算法研究、理论推导、仿真、消融、对比与实证实验。它不是 pyGWRx 的替代品，也不是成熟软件包；pyGWRx 中现有 `LGGWR` 实现仅作为本仓库的初始 research baseline。

## Start here

新的对话或新的研究阶段请按以下顺序恢复项目：

1. `00_PROJECT_HANDOFF.md`
2. `ARCHITECTURE_INDEX.md`
3. `docs/project/CURRENT_STATUS.md`
4. `docs/design/LGGWR_BASELINE_SPEC.md`
5. `docs/design/RESEARCH_QUESTIONS.md`
6. `docs/experiments/EXPERIMENT_PLAN.md`
7. `docs/decisions/` 最新 ADR

## Current baseline

当前 pyGWRx baseline 的核心思想是：

1. 将地理坐标与上下文属性构成 geometry input；
2. joint 模式学习线性潜在映射 `z = A u`；
3. separable 模式保留 geographic channel，并学习 attribute map `zeta = B a`；
4. 在学习后的距离几何中构造局部核权重；
5. 使用局部加权最小二乘进行回归；
6. 以 leave-one-out prediction error 训练潜在几何；
7. 使用解析梯度与约束优化稳定训练；
8. 可进一步用 AICc 选择报告带宽。

**以上不是最终论文算法。任何组件都允许通过理论、文献和实验被修改或删除。**

## Planned minimal code layout

```text
LatentGeometry-GWR/
├─ 00_PROJECT_HANDOFF.md
├─ ARCHITECTURE_INDEX.md
├─ pyproject.toml
├─ src/latentgeometry_gwr/
│  ├─ gwr.py
│  └─ lggwr.py
├─ tests/
│  ├─ test_gwr.py
│  └─ test_lggwr.py
├─ experiments/
├─ results/
└─ docs/
   ├─ project/
   ├─ design/
   ├─ experiments/
   ├─ literature/
   └─ decisions/
```

## Research rule

代码存在不等于理论已经确认。每个关键算法组件都应经过：

**source behavior -> mathematical statement -> literature check -> targeted simulation -> keep/modify/remove decision**。

当模型定义发生变化时，必须同步更新 `CURRENT_STATUS.md`、设计规范、ADR 和 `SESSION_LOG.md`，确保以后任何对话都能准确接手。
