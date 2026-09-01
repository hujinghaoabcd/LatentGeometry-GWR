# CURRENT_STATUS

## Current phase

**Phase B.1 — theory audit started: metric–bandwidth identifiability**

The standalone baseline is now numerically anchored both to the pinned pyGWRx LGGWR source and to an external standard-GWR reference chain. The project has therefore left extraction/ground-truthing and entered component-by-component theory audit. No final LG-GWR redesign has yet been accepted.

## Baseline extraction — COMPLETED

Established:

- `src/latentgeometry_gwr/lggwr.py` — standalone LG-GWR research baseline;
- `src/latentgeometry_gwr/gwr.py` — trusted standard-GWR baseline;
- `src/latentgeometry_gwr/bandwidth.py` — validated standard-GWR bandwidth policies reused from GeoRegime-GWR;
- `src/latentgeometry_gwr/core.py` — research-only helpers;
- mathematical invariant tests, including analytical-gradient and identifiability tests;
- permanent Python 3.10 / 3.12 test CI;
- pinned pyGWRx parity CI;
- standard-GWR external-reference CI.

## Source anchors

### LG-GWR source baseline

- repository: `hujinghaoabcd/pyGWRx`
- source: `src/pygwrx/models/lg_gwr.py`
- pinned snapshot: `ee26988a0c5b7ed15edf2d6065f538ed0d4d5429`

### Standard-GWR reference baseline

- repository: `hujinghaoabcd/GeoRegime-GWR`
- pinned snapshot: `428336399da87eb4ada4f97dfc5cc1993fa4b7e9`
- external implementation: `mgwr==2.2.1`
- canonical data: Georgia GWR example, 159 counties

## Validation status

### 1. Standalone LG-GWR invariants

Protected properties include:

- joint analytical gradient vs central finite difference for Gaussian / bisquare / exponential;
- separable analytical gradient checks;
- geometry unit invariance;
- deterministic restarts;
- prediction consistency;
- separable `h_a = infinity` geographic-GWR reduction;
- failed-refit state clearing;
- metric/bandwidth scale and latent-rotation invariances.

### 2. Strict pyGWRx parity — PASSED

Harness: `experiments/validation/lggwr_vs_pygwrx.py`

GitHub Actions run: `33531288453`

Tolerance: `atol = rtol = 1e-10`.

Cases:

1. joint + fixed bandwidth;
2. joint + AICc reselection;
3. separable + fixed bandwidth;
4. separable + AICc reselection.

Result:

- all four cases passed;
- bandwidth differences = 0;
- AICc differences = 0;
- largest local-coefficient difference ≈ `3.11e-15`;
- overall largest recorded floating-point difference ≈ `2.84e-14`.

Evidence: `results/validation/lggwr_vs_pygwrx/summary.json`.

The audit caught and corrected extraction drift in bandwidth-grid defaults (`joint 12 -> 16`, `separable 6 -> 7`) without relaxing tolerance.

### 3. Standard-GWR reference anchor — PASSED

Harness: `experiments/validation/basicgwr_reference_anchor.py`

GitHub Actions run: `33532293971`.

Evidence: `results/validation/basicgwr_reference_anchor/summary.json`.

Canonical Georgia results:

- research-default exhaustive adaptive AICc: standalone `k=116`, GeoRegime-GWR `k=116`;
- all research-default parameters, fitted values, residuals and hat matrix: exact zero difference against pinned GeoRegime-GWR;
- mgwr-compatible search: standalone `k=117`, GeoRegime-GWR `k=117`, `mgwr 2.2.1 k=117`;
- standalone vs mgwr max parameter difference: `5.55e-16`;
- standalone vs mgwr max fitted difference: `5.55e-16`;
- standalone vs mgwr max hat-matrix difference: `5.55e-17`;
- standalone AICc = mgwr AICc = `299.0508086830288`;
- fixed-distance path is exactly equal to the pinned GeoRegime implementation, but is not claimed as independently externally validated.

**Standard GWR is now treated as frozen infrastructure rather than an LG-GWR research target.**

## Current LG-GWR baseline definition

The extracted LG-GWR still retains the pinned pyGWRx structure:

- geometry: `joint`, `separable`;
- kernels: Gaussian, bisquare, exponential;
- analytical LOO geometry gradient;
- NumPy Adam + gradient clipping + early stopping;
- optional AICc bandwidth reselection;
- Frobenius / orthogonal / none scale constraints;
- local weighted least-squares refit;
- prediction and metric outputs.

**This remains a validated baseline snapshot, not the final paper algorithm.**

## Theory audit B.1 — metric–bandwidth identifiability

Design note: `docs/design/METRIC_BANDWIDTH_IDENTIFIABILITY.md`.

For joint geometry,

`r_ij = ||A(u_i-u_j)|| / h`

implies

`r_ij^2 = (u_i-u_j)^T [A^T A / h^2] (u_i-u_j)`.

Thus the kernel neighbourhood system identifies the combined PSD object

`H = A^T A / h^2`,

not `A` and `h` separately.

Two exact symmetries are now explicitly recognized and tested:

1. global scale: `(A,h) -> (cA,ch)`, `c>0`;
2. latent rotation: `A -> Q A`, where `Q^T Q = I`.

Consequences:

- `A` is a computational factor, not a unique scientific estimand;
- raw `M=A^T A` removes rotation ambiguity but remains scale-dependent unless a scale convention is imposed;
- current `diag(M)/trace(M)` contributions are scale/rotation invariant but omit off-diagonal geometry;
- current Frobenius projection is a numerical scale-fixing convention, not an identification theorem.

Leading paper-facing candidate parameterization:

`C = (A^T A) / trace(A^T A)`, with `trace(C)=1`,

and effective bandwidth

`b = h / ||A||_F`.

Then

`r_ij = sqrt(delta_ij^T C delta_ij) / b`.

The analogous separable attribute objects are `C_a` and `b_a` from `(B,h_a)`.

No optimizer change has yet been accepted.

## Immediate next tasks

1. quantify the practical effect of initialization-dependent Frobenius norm across random restarts, especially under fixed bandwidth;
2. expose canonical `(C,b)` and separable `(C_a,b_a)` as experimental diagnostics without changing fitted weights;
3. compare current factor parameterization vs explicit canonicalization in controlled synthetic regimes;
4. decide whether the paper should optimize a factor but report a canonical metric, or optimize an explicitly normalized PSD metric shape;
5. after identification is settled, audit the LOO objective and geometry–bandwidth alternating optimization;
6. only then compare joint vs geography-preserving/separable formulations as candidate final algorithms.

## Do not do yet

- do not interpret rows of `A` or `B` as unique latent axes;
- do not claim raw `metric_matrix_` is uniquely identified without stating the scale convention;
- do not benchmark-optimize before identification and objective coupling are resolved;
- do not introduce nonlinear embeddings just for accuracy;
- do not freeze joint geometry as the paper's final model.

## Key documents

- `docs/design/LGGWR_BASELINE_SPEC.md`
- `docs/design/METRIC_BANDWIDTH_IDENTIFIABILITY.md`
- `docs/design/RESEARCH_QUESTIONS.md`
- `results/validation/lggwr_vs_pygwrx/summary.json`
- `results/validation/basicgwr_reference_anchor/summary.json`
