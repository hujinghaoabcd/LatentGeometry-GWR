# RESEARCH_QUESTIONS

本文件记录 LG-GWR 尚未冻结的理论与算法问题。问题被关闭前，不应把对应实现写成论文最终定义。

## RQ1. What exactly is being learned?

当前 joint LG-GWR 学习 `z = A[s,a]`。需要明确论文中的统计含义：

- metric learning；
- contextual neighbourhood learning；
- spatial neighbourhood deformation；
- 或局部回归中的 learned similarity kernel。

不同表述对应不同理论依据与实验设计。

## RQ2. Should geography be deformable or protected?

joint geometry 允许属性信息与坐标共同决定 latent distance，理论上可能使地理上很远的位置在 latent space 中变近。

需要比较至少：

- fully joint geometry；
- geography-preserving separable geometry；
- geographic base distance + learned correction；
- constrained deformation where geographic proximity cannot be completely removed。

## RQ3. Identification of metric and bandwidth

`A` 的整体尺度与 bandwidth `h` 存在天然混淆：

`||c A(u_i-u_j)|| / (c h) = ||A(u_i-u_j)|| / h`。

必须明确 scale constraint 或其他 identification convention，并解释其统计含义。

## RQ4. Rotation / representation non-identifiability

距离只依赖 `A^T A`，latent coordinates 本身可能在正交旋转下不唯一。

论文是否应把核心参数定义为 metric matrix `M = A^T A`，而把 `A` 仅视为因子化参数？

## RQ5. Is a linear latent map sufficient?

当前模型是线性 map。需要先证明线性 metric learning 已能解决目标科学问题，再讨论 nonlinear extension。非线性模型不能仅以精度提升作为依据。

## RQ6. What is the correct training objective?

当前使用 leave-one-out prediction error。需要比较或论证：

- LOO MSE；
- local likelihood / Gaussian log-likelihood；
- AICc-related objective；
- penalized likelihood；
- nested validation objective。

训练 geometry 和选择 bandwidth 是否应使用同一目标，也需要明确。

## RQ7. Joint optimisation versus alternating optimisation

当前存在 geometry training 与 AICc bandwidth reselection 的阶段式更新。

需要研究：

- alternating optimisation 是否收敛稳定；
- joint optimisation 是否必要；
- bandwidth selection 是否会改变 geometry gradient 的定义；
- reported final state 是否对应统一目标函数。

## RQ8. Latent dimension selection

`latent_dim` 不应仅作为手工超参数。候选策略：

- fixed interpretable dimension；
- CV；
- AICc / complexity penalty；
- eigenvalue / effective-rank criterion。

## RQ9. Complexity and overfitting

LG-GWR 比 standard GWR 增加 metric-learning degrees of freedom。

必须设计：

- no-learning control；
- shuffled-attribute control；
- irrelevant-attribute robustness；
- sample-size sensitivity；
- effective complexity accounting；
- out-of-sample validation。

## RQ10. What simulation can uniquely validate LG-GWR?

不能只做 smooth spatial coefficient surface。

至少需要设计：

- geography-only nonstationarity: LG-GWR should collapse toward GWR；
- context-driven neighbourhood: geographic GWR is intentionally misspecified；
- mixed geography + context；
- irrelevant context；
- correlated but non-causal context；
- discontinuous / manifold-like similarity；
- varying sample density and noise。

## RQ11. Interpretation

需要明确可解释对象：

- metric matrix；
- feature contributions to distance；
- latent coordinates；
- neighbourhood changes；
- local coefficients。

其中哪些具有旋转不变性、尺度不变性和可比较性必须严格区分。

## RQ12. Relationship to existing methods

需要系统比较相关理论：

- generalized / anisotropic GWR distance metrics；
- spatially varying kernels；
- metric learning；
- supervised manifold learning；
- spatial regimes / context-aware local regression；
- geographically neural / embedding-based local models。

文献比较必须围绕“邻近关系如何定义与学习”，而不只是列 GWR 变体。
