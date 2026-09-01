# CURRENT_STATUS

## Current phase

**Phase A.2 — standalone baseline established; parity and theory audit next**

当前仍未进入最终论文算法设计。pyGWRx 中的 LGGWR 已经抽离成独立 research baseline，软件层依赖已经移除，第一组数学不变量已经由永久 CI 保护。

## Baseline extraction — COMPLETED

已建立：

- `src/latentgeometry_gwr/lggwr.py` — 独立 LG-GWR research baseline；
- `src/latentgeometry_gwr/gwr.py` — 最小标准 GWR 对照引擎；
- `src/latentgeometry_gwr/core.py` — 仅保留本研究需要的输入、诊断与 summary helper；
- `tests/test_lggwr.py` — joint geometry 核心不变量；
- `tests/test_lggwr_separable_gradient.py` — separable analytical-gradient 验证；
- `tests/test_gwr.py` — standard GWR smoke baseline；
- `.github/workflows/tests.yml` — Python 3.10 / 3.12 永久 CI；
- `docs/design/LGGWR_BASELINE_SPEC.md` — source-to-math baseline specification。

## Source anchor

pyGWRx source baseline：

- repository: `hujinghaoabcd/pyGWRx`
- source: `src/pygwrx/models/lg_gwr.py`
- source snapshot observed during extraction: `ee26988a0c5b7ed15edf2d6065f538ed0d4d5429`
- class: `LGGWR`

当前独立实现不是成熟软件包复制品。它刻意删除 pyGWRx 的 package-wide API、文档和通用软件层，只保留论文研究所需数学行为。

## Validation status

### Local extraction validation

`pytest`：**11 passed**。

覆盖：

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
12. standard GWR smoke behaviour is also protected within the 11-test suite through the standalone GWR test module.

### GitHub Actions

Permanent CI run passed on both:

- Python 3.10 — success；
- Python 3.12 — success。

## Current baseline definition

The extracted LG-GWR retains the pyGWRx research structure:

- geometry modes: `joint`, `separable`；
- kernels: Gaussian, bisquare, exponential；
- geometry training: analytical leave-one-out gradient；
- optimiser: NumPy Adam + gradient clipping + early stopping；
- optional bandwidth reselection by Gaussian GWR AICc；
- scale constraints: Frobenius, orthogonal, none；
- final local weighted least-squares refit；
- prediction and metric outputs。

**This is a baseline snapshot, not the final paper algorithm.**

## Immediate next tasks

1. Build a strict numerical parity harness against the pinned pyGWRx source snapshot, using identical synthetic inputs and fixed settings.
2. Expand the standalone `BasicGWR` validation so its standard-GWR behaviour is anchored to the already validated GeoRegime-GWR / external mgwr results rather than only a smoke test.
3. Audit every baseline component using `source behavior -> mathematical statement -> literature check -> targeted simulation -> keep/modify/remove`.
4. Resolve metric/bandwidth scale identification before interpreting learned metric contributions.
5. Compare joint geometry with geography-preserving/separable alternatives before selecting the paper's primary formulation.
6. Only after the above begin theory-driven algorithm redesign and benchmark optimisation.

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
