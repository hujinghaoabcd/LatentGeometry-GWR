# 00_PROJECT_HANDOFF

> **跨对话最高优先级入口。任何新的 ChatGPT / Codex 对话继续本仓库前，必须先读本文。**

## 1. 项目定位

`LatentGeometry-GWR` 是 **Latent-Geometry Geographically Weighted Regression (LG-GWR)** 方法论文的轻量研究仓库，不是成熟软件包。

用途：

- 从 pyGWRx 独立理解、重构和验证 LG-GWR；
- 保留可信的最小 standard GWR baseline；
- 快速修改论文算法；
- 做 synthetic / real-data / ablation / benchmark 实验；
- 生成论文结果；
- 通过设计文档、ADR 和交接文档保证跨对话连续开发。

## 2. 最高原则

### 2.1 当前 LG-GWR 不是最终算法

pyGWRx 中 `src/pygwrx/models/lg_gwr.py` 只是 baseline snapshot。后续完全允许修改或替换：

- joint / separable geometry；
- linear latent map；
- geometry standardisation；
- kernel definition；
- bandwidth policy；
- leave-one-out training objective；
- scale constraint；
- AICc reselection；
- Adam optimisation；
- 整个 geometry-learning framework。

不要因为现有代码已经实现就把它当作论文最终定义。

### 2.2 研究顺序

每个组件遵循：

`source behavior -> mathematical statement -> literature check -> targeted simulation -> keep / modify / remove`

### 2.3 方法定义变化必须留痕

至少同步：

- `docs/project/CURRENT_STATUS.md`
- 对应设计规范
- `docs/decisions/` 新增或更新 ADR
- 若影响整体流程，更新本文

## 3. pyGWRx 当前 baseline 已确认的核心行为

当前源实现定义 `LGGWR` 为 Latent-Geometry Geographically Weighted Regression。

### 3.1 Joint geometry

对输入 `u_i = [s_i, a_i]` 学习线性映射：

`z_i = A u_i`

并用 latent-space distance 构造核权重：

`w_ij = K(||z_i-z_j|| / h)`

当前实现以 leave-one-out prediction error 训练 `A`。

### 3.2 Separable geometry

保留 geographic distance channel，同时学习属性映射：

`zeta_i = B a_i`

权重写为两个通道乘积：

`w_ij = K(d_geo_ij / h_g) * K(||zeta_i-zeta_j|| / h_a)`

当 `h_a = infinity` 时，当前代码设计为退化到相同 geographic bandwidth 下的 ordinary geographic GWR。

### 3.3 当前实现中的重要工程组件

- Gaussian / bisquare / exponential kernel；
- analytical LOO gradient；
- Frobenius / orthogonal / none scale constraint；
- deterministic restarts；
- geometry standardisation；
- AICc bandwidth reselection；
- joint 与 separable 两套训练路径；
- fitted coefficients / predictions / latent coordinates / metric contributions 输出。

这些只是需要被验证的源行为，不代表论文最终应全部保留。

## 4. 当前第一阶段目标

Phase A — baseline extraction and validation：

1. 从 pyGWRx 提取 LGGWR 的最小算法依赖；
2. 建立独立的 standard GWR baseline；
3. 将 pyGWRx LGGWR 转化为更小、更可读的 research baseline；
4. 搬迁并精简关键数学测试，尤其是 analytical gradient tests；
5. 验证 joint / separable 的关键退化性质；
6. 明确 source behavior 与数学表达一一对应；
7. 在任何新算法修改前冻结 baseline validation evidence。

## 5. 当前最重要的问题

不要直接开始“调参数提升精度”。必须先回答：

- 为什么 GWR 的邻近关系应该从固定地理空间变成可学习几何？
- latent geometry 的统计含义到底是 metric learning、contextual neighbourhood，还是一种空间非平稳性的重参数化？
- coordinates 与 attributes 是否应该 joint embedding？
- geographic proximity 是否应保留不可消除的基础约束？
- LOO prediction loss 是否是论文上最合理的 geometry-learning objective？
- `A` / `B` 的可识别性如何正式处理？
- latent dimension 如何选择？
- bandwidth 与 metric 是否存在尺度混淆？
- 如何证明模型不是单纯通过额外自由度过拟合？
- 应该设计哪些 synthetic regimes 才能真正验证“learned neighbourhood”而不是只比较拟合优度？

## 6. 跨对话恢复顺序

1. `00_PROJECT_HANDOFF.md`
2. `ARCHITECTURE_INDEX.md`
3. `docs/project/CURRENT_STATUS.md`
4. `docs/design/LGGWR_BASELINE_SPEC.md`
5. `docs/design/RESEARCH_QUESTIONS.md`
6. 最新 ADR
7. 当前实验结果
