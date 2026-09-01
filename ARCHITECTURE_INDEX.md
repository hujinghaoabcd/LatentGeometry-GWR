# ARCHITECTURE_INDEX

本文件是仓库结构索引，不定义最终算法。跨对话恢复时先读 `00_PROJECT_HANDOFF.md`，再读本文件。

## 1. 代码

- `src/latentgeometry_gwr/gwr.py`
  - 最小 standard GWR baseline；
  - 负责地理距离核、固定/自适应带宽、局部加权最小二乘；
  - 不承担 latent geometry 学习逻辑。

- `src/latentgeometry_gwr/lggwr.py`
  - 当前 LG-GWR research baseline；
  - 从 pyGWRx 的 `src/pygwrx/models/lg_gwr.py` 提取；
  - 目标是更小、更容易数学核对与修改；
  - **不是最终论文算法。**

## 2. 项目状态

- `docs/project/CURRENT_STATUS.md`
  - 当前阶段、已完成事项、正在讨论的问题、下一步任务。

## 3. 模型设计

- `docs/design/LGGWR_BASELINE_SPEC.md`
  - 当前 baseline 的逐步数学与代码对应关系。

- `docs/design/METRIC_BANDWIDTH_IDENTIFIABILITY.md`
  - `A/B`、PSD metric 与 bandwidth 的尺度/旋转可识别性审查；
  - 当前候选 canonical representation：normalized metric shape + effective bandwidth。

- `docs/design/JOINED_VS_GEOGRAPHY_PRESERVING_PATHS.md`
  - **当前最重要的双路线设计记录之一**；
  - 同时保留 joint LG-GWR 与 geography-preserving/separable 两条候选主线；
  - joint 路线强调 non-separable geographic-context geometry 与 `M_sc` cross-coupling，算法创新更强；
  - geography-preserving 路线强调 physical geography anchor、process-conditioned locality 与更强地学可解释性；
  - **尚未裁决，后续必须通过文献、理论、仿真和地理案例决定。**

- `docs/design/RESEARCH_QUESTIONS.md`
  - 当前所有尚未冻结的模型设计问题。

## 4. 实验

- `docs/experiments/EXPERIMENT_PLAN.md`
  - 仿真、消融、对比、实证的计划和优先级。

建议目录：

- `experiments/validation/`
- `experiments/simulations/`
- `experiments/ablations/`
- `experiments/benchmarks/`
- `experiments/case_studies/`
- `results/`

## 5. 决策记录

- `docs/decisions/ADR-0001-research-repository-scope.md`
- 后续每次改变论文方法定义时新增 ADR，不覆盖历史。

## 6. 测试

- `tests/test_gwr.py`
- `tests/test_lggwr.py`

测试目标不是成熟软件包 API 覆盖，而是保护算法数学性质，包括：

- standard GWR baseline 数值一致性；
- analytical gradient 与 finite difference 一致；
- separable attribute channel 关闭时退化为 geographic GWR；
- geometry scaling / identification constraints；
- deterministic optimisation behavior；
- bandwidth / geometry final-state consistency。

## 7. 文档优先级

发生冲突时：

1. 最新已接受 ADR
2. `docs/project/CURRENT_STATUS.md`
3. 当前正式模型设计规范
4. `00_PROJECT_HANDOFF.md`
5. README

README 只负责项目入口，不作为算法真相来源。
