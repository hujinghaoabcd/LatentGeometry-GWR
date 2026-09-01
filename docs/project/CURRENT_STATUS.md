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

## Candidate paper/model paths — BOTH RETAINED

Design note: `docs/design/JOINED_VS_GEOGRAPHY_PRESERVING_PATHS.md`.

The project now explicitly retains two competing candidate directions. **Neither is the accepted final model.**

### Path A — joint LG-GWR / learned geographic geometry

Joint formulation uses `u_i=[s_i,c_i]`, `z_i=A u_i`, so geography and process-relevant context form one coupled geometry. With `M=A^T A` partitioned into geographic and contextual blocks, the cross block `M_sc` permits non-separable space–context coupling.

Provisional strength:

- stronger algorithmic novelty;
- clearer distinction from SGWR-style “geographic weight + attribute-similarity weight” fusion;
- candidate interpretation as **process-conditioned deformation/reconstruction of geographic geometry** rather than generic attribute similarity.

Provisional risk:

- physical geography may be excessively deformed or suppressed;
- burden of geographic interpretation is higher;
- must distinguish the method from generic supervised metric learning and newer contextual-geometry GWR variants such as CEGWR.

### Path B — geography-preserving / separable process-conditioned locality

Physical geographic distance remains explicit while a learned contextual component modifies the effective local weighting structure.

Provisional strength:

- stronger immediate geographic interpretation;
- easier alignment with GWR, spatial proximity, geographic similarity/configuration, and process-conditioned locality.

Provisional risk:

- greater novelty overlap with SGWR, SGWR-GD and related similarity-GWR methods;
- a simple product of geographic and contextual kernels is not sufficient as a final innovation.

### Current decision rule

Do **not** decide between Path A and Path B by predictive accuracy alone. Final selection must consider geographic faithfulness, null behavior, process sensitivity, boundaries, stability/identifiability, interpretability, novelty against recent literature, and real-data geographic plausibility.

Terminology: avoid “空间借样 / 信息借用”. Prefer “局地信息整合”, “邻域贡献”, “局部样本权重”, “有效邻域构成”, or “局地权重分配”.

## Immediate next tasks

1. quantify the practical effect of initialization-dependent Frobenius norm across random restarts, especially under fixed bandwidth;
2. expose canonical `(C,b)` and separable `(C_a,b_a)` as experimental diagnostics without changing fitted weights;
3. compare current factor parameterization vs explicit canonicalization in controlled synthetic regimes;
4. decide whether the paper should optimize a factor but report a canonical metric, or optimize an explicitly normalized PSD metric shape;
5. after identification is settled, audit the LOO objective and geometry–bandwidth alternating optimization;
6. design a controlled **Path A vs Path B** comparison that can distinguish non-separable joint geometry from geography-preserving contextual modulation;
7. explicitly compare against SGWR / SGWR-GD / EDSGWR / CEGWR / supervised spatial metric learning where technically appropriate;
8. only after these audits select the paper's primary formulation and begin full benchmark / real-data experiments.

## Do not do yet

- do not interpret rows of `A` or `B` as unique latent axes;
- do not claim raw `metric_matrix_` is uniquely identified without stating the scale convention;
- do not benchmark-optimize before identification and objective coupling are resolved;
- do not introduce nonlinear embeddings just for accuracy;
- do not freeze joint geometry as the paper's final model;
- do not discard joint geometry merely because geography-preserving formulations are easier to interpret;
- do not discard geography-preserving formulations merely because joint geometry appears more algorithmically novel.

## Key documents

- `docs/design/LGGWR_BASELINE_SPEC.md`
- `docs/design/METRIC_BANDWIDTH_IDENTIFIABILITY.md`
- `docs/design/JOINED_VS_GEOGRAPHY_PRESERVING_PATHS.md`
- `docs/design/RESEARCH_QUESTIONS.md`
- `results/validation/lggwr_vs_pygwrx/summary.json`
- `results/validation/basicgwr_reference_anchor/summary.json`
