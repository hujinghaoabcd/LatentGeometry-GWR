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

当前 `src/latentgeometry_gwr/lggwr.py` 是从 pinned pyGWRx source 抽离并完成数值 parity 的 **validated baseline snapshot**。后续完全允许修改或替换：

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

不要因为现有代码已经实现并通过 parity 就把它当作论文最终定义。Parity 只证明“独立仓库忠实复现了源算法”，不证明“源算法理论上正确或最优”。

### 2.2 研究顺序

每个组件遵循：

`source behavior -> mathematical statement -> literature check -> targeted simulation -> keep / modify / remove`

### 2.3 方法定义变化必须留痕

至少同步：

- `docs/project/CURRENT_STATUS.md`
- 对应设计规范
- `docs/decisions/` 新增或更新 ADR
- 若影响整体流程，更新本文

## 3. Pinned source baseline

pyGWRx source anchor：

- repository: `hujinghaoabcd/pyGWRx`
- source: `src/pygwrx/models/lg_gwr.py`
- pinned commit: `ee26988a0c5b7ed15edf2d6065f538ed0d4d5429`
- class: `LGGWR`

### 3.1 Joint geometry

对输入 `u_i = [s_i, a_i]` 学习线性映射：

`z_i = A u_i`

并用 latent-space distance 构造核权重：

`w_ij = K(||z_i-z_j|| / h)`

当前 baseline 以 leave-one-out prediction error 训练 `A`。

### 3.2 Separable geometry

保留 geographic distance channel，同时学习属性映射：

`zeta_i = B a_i`

权重写为两个通道乘积：

`w_ij = K(d_geo_ij / h_g) * K(||zeta_i-zeta_j|| / h_a)`

当 `h_a = infinity` 时，当前代码设计为退化到相同 geographic bandwidth 下的 ordinary geographic GWR。

### 3.3 当前 baseline 的工程组件

- Gaussian / bisquare / exponential kernel；
- analytical LOO gradient；
- Frobenius / orthogonal / none scale constraint；
- deterministic restarts；
- geometry standardisation；
- AICc bandwidth reselection；
- joint 与 separable 两套训练路径；
- fitted coefficients / predictions / latent coordinates / metric contributions 输出。

这些只是已经被忠实复现和验证的 source behavior，不代表论文最终应全部保留。

## 4. 当前阶段 — Phase A.3

**Standalone baseline numerically anchored; theory audit next.**

已完成：

1. 从 pyGWRx 抽离 LGGWR 最小研究实现；
2. 建立独立 standard GWR baseline；
3. 去除 pyGWRx package-wide 软件层依赖；
4. 建立 joint / separable analytical-gradient tests；
5. 建立 unit invariance、restart reproducibility、failed-refit atomicity 等关键不变量测试；
6. 验证 separable `h_a = infinity` 的 geographic-GWR 退化性质；
7. 建立 `LGGWR_BASELINE_SPEC.md` source-to-math 对应；
8. 建立 Python 3.10 / 3.12 永久测试 CI；
9. 建立 pinned pyGWRx strict numerical parity CI；
10. strict parity 已在 4 个配置上通过：joint fixed、joint+AICc、separable fixed、separable+AICc。

Strict parity evidence：

- harness: `experiments/validation/lggwr_vs_pygwrx.py`
- result: `results/validation/lggwr_vs_pygwrx/summary.json`
- GitHub Actions run: `33531288453`
- tolerance: `atol = rtol = 1e-10`
- bandwidth difference: `0`
- AICc difference: `0`
- largest local-coefficient difference: about `3.11e-15`
- overall largest recorded floating-point difference: about `2.84e-14`

### 4.1 Parity audit discovered and corrected one extraction drift

首次 strict parity run 失败并发现：

- source joint AICc grid = `16`，抽取版误写为 `12`；
- source separable AICc grid = `7`，抽取版误写为 `6`。

已经恢复为 `16` / `7`，**没有放宽任何 parity tolerance**。修复后 4 组 strict parity 全部通过。

## 5. 现在最重要的研究问题

**不要直接开始“调参数提升精度”。** 当前下一阶段是理论审查，不是 benchmark chasing。

必须先回答：

- 为什么 GWR 的邻近关系应该从固定地理空间变成可学习几何？
- latent geometry 的统计含义到底是 metric learning、contextual neighbourhood，还是空间非平稳性的重参数化？
- coordinates 与 attributes 是否应该 joint embedding？
- geographic proximity 是否应保留不可消除的基础约束？
- LOO prediction loss 是否是论文上最合理的 geometry-learning objective？
- 真正可识别、可解释的对象是 `A`、`B`，还是 `A^T A` / `B^T B`？
- latent dimension 如何选择？
- bandwidth 与 metric 是否存在尺度混淆，当前 Frobenius constraint 是否足够？
- AICc bandwidth reselection 与 geometry-learning objective 的耦合是否统计上自洽？
- 如何证明模型不是单纯通过额外自由度过拟合？
- 应该设计哪些 synthetic scenarios 才能真正验证“learned neighbourhood”而不是只比较拟合优度？

## 6. 下一步优先级

1. 加强 `BasicGWR` external/reference validation，使 standard-GWR baseline 与 GeoRegime-GWR 已验证结果对齐；
2. 对 LG-GWR 做逐组件理论审查；
3. 第一优先研究 **metric / bandwidth identification**；
4. 第二优先研究 **joint vs geography-preserving/separable geometry**；
5. 第三优先研究 **LOO objective + AICc bandwidth coupling**；
6. 再研究 latent dimension、constraints、initialisation、restart policy；
7. 完成理论取舍后再设计系统 synthetic experiments、benchmark 和 real-data experiments。

## 7. 跨对话恢复顺序

1. `00_PROJECT_HANDOFF.md`
2. `ARCHITECTURE_INDEX.md`
3. `docs/project/CURRENT_STATUS.md`
4. `docs/design/LGGWR_BASELINE_SPEC.md`
5. `docs/design/RESEARCH_QUESTIONS.md`
6. `results/validation/lggwr_vs_pygwrx/summary.json`
7. 最新 ADR
8. 当前实验结果
