# CURRENT_STATUS

## Current phase

**Phase A.3 — standalone baseline numerically anchored; theory audit next**

当前仍未进入最终论文算法设计。pyGWRx 中的 LGGWR 已经抽离成独立 research baseline，软件层依赖已经移除，核心数学不变量与 pinned-source numerical parity 均已建立。

## Baseline extraction — COMPLETED

已建立：

- `src/latentgeometry_gwr/lggwr.py` — 独立 LG-GWR research baseline；
- `src/latentgeometry_gwr/gwr.py` — 最小标准 GWR 对照引擎；
- `src/latentgeometry_gwr/core.py` — 仅保留本研究需要的输入、诊断与 summary helper；
- `tests/test_lggwr.py` — joint geometry 核心不变量；
- `tests/test_lggwr_separable_gradient.py` — separable analytical-gradient 验证；
- `tests/test_gwr.py` — standard GWR smoke baseline；
- `.github/workflows/tests.yml` — Python 3.10 / 3.12 永久 CI；
- `.github/workflows/parity-pygwrx.yml` — pinned pyGWRx numerical-parity CI；
- `docs/design/LGGWR_BASELINE_SPEC.md` — source-to-math baseline specification；
- `results/validation/lggwr_vs_pygwrx/summary.json` — strict parity evidence。

## Source anchor

pyGWRx source baseline：

- repository: `hujinghaoabcd/pyGWRx`
- source: `src/pygwrx/models/lg_gwr.py`
- pinned source snapshot: `ee26988a0c5b7ed15edf2d6065f538ed0d4d5429`
- class: `LGGWR`

当前独立实现不是成熟软件包复制品。它刻意删除 pyGWRx 的 package-wide API、文档和通用软件层，只保留论文研究所需数学行为。

## Validation status

### Standalone invariant tests

当前研究测试覆盖：

1. joint analytical gradient vs central finite difference — Gaussian；
2. joint analytical gradient vs central finite difference — bisquare；
3. joint analytical gradient vs central finite difference — exponential；
4. separable analytical gradient vs central finite difference — Gaussian；
5. separable analytical gradient vs central finite difference — bisquare；
6. synthetic training improves/preserves best LOO state；
7. geometry standardisation / unit invariance；
8. deterministic restart reproducibility；
9. DataFrame prediction reproduces training fitted values；
10. separable `h_a = infinity` reduces to geographic GWR；
11. failed refit clears fitted state；
12. standalone standard GWR smoke behaviour。

Permanent CI has passed on Python 3.10 and Python 3.12.

### Strict pyGWRx parity — PASSED

Validation harness:

- `experiments/validation/lggwr_vs_pygwrx.py`
- GitHub Actions run: `33531288453`
- tolerance: `atol = rtol = 1e-10`

Compared cases:

1. joint geometry + fixed bandwidth；
2. joint geometry + AICc bandwidth reselection；
3. separable geometry + fixed bandwidth；
4. separable geometry + AICc bandwidth reselection。

Compared state/output includes:

- `A_` / `B_`；
- bandwidth and bandwidth history；
- latent coordinates；
- full local parameter matrices；
- fitted values and residuals；
- hat matrix；
- learned metric matrix and contributions；
- LOO loss history / best loss / final loss；
- Gaussian GWR diagnostics including AICc；
- prediction results；
- iteration and stopping state。

Final result:

- all 4 cases passed；
- bandwidth differences = `0`；
- AICc differences = `0`；
- largest local-coefficient difference ≈ `3.11e-15`；
- overall largest recorded floating-point difference ≈ `2.84e-14`；
- therefore standalone behaviour matches the pinned pyGWRx source to machine precision under the tested configurations。

### Extraction bug found and corrected during parity audit

The first strict parity run deliberately failed and exposed an extraction drift:

- pyGWRx joint bandwidth-search grid: `n_grid = 16`；
- extracted standalone value had been reduced to `12`；
- pyGWRx separable bandwidth-search grid: `n_grid = 7`；
- extracted standalone value had been reduced to `6`。

These defaults were restored to `16` and `7`. No validation tolerance was relaxed. After correction, all strict parity cases passed.

## Current baseline definition

The extracted LG-GWR retains the pinned pyGWRx research structure:

- geometry modes: `joint`, `separable`；
- kernels: Gaussian, bisquare, exponential；
- geometry training: analytical leave-one-out gradient；
- optimiser: NumPy Adam + gradient clipping + early stopping；
- optional bandwidth reselection by Gaussian GWR AICc；
- scale constraints: Frobenius, orthogonal, none；
- final local weighted least-squares refit；
- prediction and metric outputs。

**This is a validated baseline snapshot, not the final paper algorithm.**

## Immediate next tasks

1. Expand the standalone `BasicGWR` validation so standard-GWR behaviour is anchored to the already validated GeoRegime-GWR / external mgwr results rather than only a smoke test.
2. Begin component-by-component LG-GWR theory audit using:
   `source behavior -> mathematical statement -> literature check -> targeted simulation -> keep/modify/remove`.
3. Resolve metric/bandwidth scale identification before interpreting learned metric contributions.
4. Audit whether the identifiable object should be `A`, `A^T A`, or an explicitly parameterised metric matrix.
5. Compare joint geometry with geography-preserving/separable alternatives before selecting the paper's primary formulation.
6. Audit LOO objective, AICc bandwidth coupling, latent dimension, standardisation, constraints and restart policy before benchmark optimisation.
7. Only after the above begin theory-driven algorithm redesign and final paper experiments.

## Do not do yet

- Do not claim current latent map is theoretically final.
- Do not optimize benchmark accuracy before identification questions are resolved.
- Do not add nonlinear neural embeddings simply because they improve fit.
- Do not conflate learned latent distance with physical geographic distance.
- Do not freeze joint geometry as the paper's main model before comparison with geography-preserving alternatives.

## Main unresolved theory questions

See:

- `docs/design/LGGWR_BASELINE_SPEC.md`
- `docs/design/RESEARCH_QUESTIONS.md`
