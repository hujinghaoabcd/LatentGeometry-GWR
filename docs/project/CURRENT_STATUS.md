# CURRENT_STATUS

## Current phase

**Phase A — baseline extraction and validation**

当前尚未进入最终论文算法设计。首要任务是从 pyGWRx 提取现有 LGGWR，建立独立、可读、可验证的 research baseline。

## Confirmed source baseline

pyGWRx 当前实现：

- source: `src/pygwrx/models/lg_gwr.py`
- class: `LGGWR`
- model name: Latent-Geometry Geographically Weighted Regression
- geometry modes: `joint`, `separable`
- kernels: Gaussian, bisquare, exponential
- geometry training: analytical leave-one-out gradient
- optimisation: NumPy Adam + clipping + early stopping
- optional bandwidth reselection by AICc
- scale constraints: Frobenius, orthogonal, none
- prediction and local coefficient outputs

## Existing tests worth preserving

pyGWRx `tests/test_lg_gwr.py` already contains several research-critical properties that should be migrated or rewritten rather than discarded:

1. analytical gradient vs finite difference for joint geometry;
2. analytical gradient vs finite difference for separable geometry;
3. fixed norm constraint and ordinary L2 regularisation incompatibility;
4. attribute-driven synthetic geometry learning;
5. learned geometry vs no-learning baseline;
6. deterministic restart reproducibility;
7. geometry standardisation / unit invariance;
8. final loss and bandwidth history consistency;
9. bandwidth selection should not worsen same-geometry AICc;
10. separable form recovers geographic GWR when attribute channel is off;
11. heavy-tail numerical stability.

## Immediate next tasks

1. Extract a minimal `BasicGWR` baseline, preferably sharing the already validated logic used by GeoRegime-GWR where appropriate.
2. Extract LGGWR core while removing pyGWRx software-layer dependencies such as generic summaries and package-wide API helpers.
3. Create `tests/test_lggwr.py` with only research-critical invariants.
4. Create a source-to-math mapping for every equation and optimisation step.
5. Run baseline synthetic tests before changing the model.
6. Only after baseline validation begin theory-driven LG-GWR redesign.

## Do not do yet

- Do not claim current latent map is theoretically final.
- Do not optimize benchmark accuracy before identification questions are resolved.
- Do not add nonlinear neural embeddings simply because they improve fit.
- Do not conflate learned latent distance with physical geographic distance.
- Do not freeze joint geometry as the paper's main model before comparison with geography-preserving alternatives.

## Main unresolved theory questions

See `docs/design/RESEARCH_QUESTIONS.md`.
