# 00_PROJECT_HANDOFF

> **跨对话最高优先级入口。任何新的 ChatGPT / Codex 对话继续本仓库前，必须先读本文。**

## 1. 项目定位

`LatentGeometry-GWR` 是 **Latent-Geometry Geographically Weighted Regression (LG-GWR)** 方法论文的轻量研究仓库，不是成熟软件包。

用途：独立理解、验证和重构 LG-GWR；保留可信 standard-GWR baseline；开展 synthetic / real-data / ablation / benchmark；生成论文结果；通过设计文档、ADR 和验证证据保持跨对话连续开发。

## 2. 最高原则

### 2.1 当前 LG-GWR 是 validated baseline，不是最终论文算法

`src/latentgeometry_gwr/lggwr.py` 已与 pinned pyGWRx source 做严格数值 parity。Parity 只证明独立仓库忠实复现源算法，不证明源算法理论上最终正确。

后续允许修改/替换 joint/separable geometry、linear latent map、standardisation、kernel、bandwidth policy、LOO objective、scale constraint、AICc reselection、Adam 以及整个 geometry-learning framework。

### 2.2 每个组件遵循

`source behavior -> mathematical statement -> literature check -> targeted simulation -> keep / modify / remove`

### 2.3 方法定义变化必须留痕

至少同步：

- `docs/project/CURRENT_STATUS.md`
- 对应设计规范
- `docs/decisions/` ADR
- 若影响整体流程，更新本文

## 3. Pinned source anchors

### LG-GWR

- repository: `hujinghaoabcd/pyGWRx`
- source: `src/pygwrx/models/lg_gwr.py`
- pinned commit: `ee26988a0c5b7ed15edf2d6065f538ed0d4d5429`
- class: `LGGWR`

### Standard GWR

- repository: `hujinghaoabcd/GeoRegime-GWR`
- pinned commit: `428336399da87eb4ada4f97dfc5cc1993fa4b7e9`
- external reference: `mgwr==2.2.1`

## 4. 当前阶段 — Phase B.1

**Theory audit started: metric–bandwidth identifiability.**

### 4.1 LG-GWR extraction/parity — COMPLETED

已完成：

1. 从 pyGWRx 抽离 LGGWR 最小研究实现；
2. 去除 package-wide 软件层依赖；
3. joint/separable analytical-gradient tests；
4. unit invariance、restart reproducibility、prediction consistency、failed-refit atomicity；
5. separable `h_a = infinity` geographic-GWR reduction；
6. `LGGWR_BASELINE_SPEC.md`；
7. Python 3.10 / 3.12 permanent CI；
8. pinned pyGWRx strict parity CI。

Strict parity evidence：

- `experiments/validation/lggwr_vs_pygwrx.py`
- `results/validation/lggwr_vs_pygwrx/summary.json`
- Actions run `33531288453`
- `atol = rtol = 1e-10`
- joint fixed / joint+AICc / separable fixed / separable+AICc 全部通过
- bandwidth difference = `0`
- AICc difference = `0`
- largest local-coefficient difference ≈ `3.11e-15`
- overall largest floating-point difference ≈ `2.84e-14`

Parity 首轮还抓出并修复了 extraction drift：joint AICc grid `12 -> 16`，separable `6 -> 7`；未放宽 tolerance。

### 4.2 Standard GWR anchor — COMPLETED / FROZEN

新仓库 `BasicGWR` 已切换为 GeoRegime-GWR 中已经验证的标准 GWR 基线，并建立永久外部 reference CI。

Evidence：

- `experiments/validation/basicgwr_reference_anchor.py`
- `results/validation/basicgwr_reference_anchor/summary.json`
- Actions run `33532293971`

Canonical Georgia：

- research-default exhaustive adaptive AICc: standalone `k=116` = GeoRegime `k=116`；
- research-default 参数、fitted、residual、hat matrix 对 GeoRegime 均为 **0 difference**；
- mgwr-compatible: standalone `117` = GeoRegime `117` = `mgwr 2.2.1` `117`；
- standalone vs mgwr max parameter/fitted difference = `5.55e-16`；
- max hat difference = `5.55e-17`；
- AICc = `299.0508086830288`，difference = `0`。

**Standard GWR 现在视为 frozen infrastructure，不再作为 LG-GWR 创新对象。**

## 5. 当前理论审查：metric–bandwidth identifiability

核心设计文档：

`docs/design/METRIC_BANDWIDTH_IDENTIFIABILITY.md`

Joint LG-GWR：

`r_ij = ||A(u_i-u_j)|| / h`

因此

`r_ij^2 = delta_ij^T [A^T A / h^2] delta_ij`。

真正由 neighbourhood weights 识别的是：

`H = A^T A / h^2`

而不是 `A` 和 `h` 分别。

现已用永久测试验证两种 exact symmetry：

1. scale symmetry: `(A,h) -> (cA,ch)`；
2. latent rotation: `A -> Q A`, `Q^T Q = I`。

因此：

- `A` / `B` 应视为计算因子，不应直接作为唯一科学 estimand；
- `M=A^T A` 去除了 latent rotation ambiguity，但仍受 global scale convention 影响；
- 当前 `diag(M)/trace(M)` contributions 对 scale/rotation 不变，但忽略 off-diagonal metric structure；
- Frobenius constraint 是数值 scale-fixing convention，不是 `A` 可识别性的证明。

当前领先的 paper-facing canonical representation：

`C = A^T A / trace(A^T A)`, `trace(C)=1`

与 effective bandwidth

`b = h / ||A||_F`。

于是

`r_ij = sqrt(delta_ij^T C delta_ij) / b`。

Separable attribute channel 同理使用 `C_a` 与 `b_a`。

**尚未修改 optimizer，也尚未接受最终 ADR。**

## 6. 下一步优先级

1. 量化 random restarts 的 initialization-dependent Frobenius norm 对 fixed-bandwidth / selected-bandwidth 结果的实际影响；
2. 先添加 canonical `(C,b)` / `(C_a,b_a)` experimental diagnostics，不改变 fitted weights；
3. controlled simulation 比较 current factor parameterization 与 explicit canonicalization；
4. 决定论文最终是“factor optimization + canonical reporting”还是直接 normalized PSD metric-shape formulation；
5. identification 决策后审查 LOO objective + AICc bandwidth coupling；
6. 再审 joint vs geography-preserving/separable；
7. 最后才进入系统 benchmark / real-data performance experiments。

## 7. 当前禁止事项

- 不要直接调参追 benchmark；
- 不要解释 `A` 的行/latent axes 为唯一物理含义；
- 不要无条件把 raw `metric_matrix_` 当成唯一可识别 metric；
- 不要为了精度直接引入 nonlinear embedding；
- 不要提前把 joint geometry 冻结成最终论文算法。

## 8. 跨对话恢复顺序

1. `00_PROJECT_HANDOFF.md`
2. `ARCHITECTURE_INDEX.md`
3. `docs/project/CURRENT_STATUS.md`
4. `docs/design/LGGWR_BASELINE_SPEC.md`
5. `docs/design/METRIC_BANDWIDTH_IDENTIFIABILITY.md`
6. `docs/design/RESEARCH_QUESTIONS.md`
7. `results/validation/lggwr_vs_pygwrx/summary.json`
8. `results/validation/basicgwr_reference_anchor/summary.json`
9. 最新 ADR
10. 当前实验结果
