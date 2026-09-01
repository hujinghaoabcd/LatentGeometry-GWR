# LG-GWR metric–bandwidth identifiability audit

Status: **theory audit; no model-definition change has been accepted yet**.

## 1. Question

In joint LG-GWR the current baseline forms

`z_i = A u_i`

and kernel weights from

`r_ij = ||A(u_i-u_j)|| / h`.

What is actually identifiable from the weights: `A`, `A^T A`, the bandwidth `h`, or some combined object?

## 2. Exact algebra

Let

`delta_ij = u_i - u_j`

and

`M = A^T A`.

Then

`r_ij^2 = delta_ij^T M delta_ij / h^2`.

Therefore define

`H = M / h^2 = A^T A / h^2`.

The kernel argument is exactly

`r_ij = sqrt(delta_ij^T H delta_ij)`.

For any kernel whose weights are a deterministic function of `r_ij`, the fitted neighbourhood system depends on `(A, h)` only through `H`.

### Consequence 1 — global scale symmetry

For any scalar `c > 0`,

`(A, h) -> (c A, c h)`

leaves `H`, every pairwise kernel argument, every weight, every local weighted least-squares fit, the hat matrix, LOO predictions and Gaussian GWR diagnostics unchanged.

Thus unconstrained `A` and `h` are not separately identifiable.

### Consequence 2 — factor rotation symmetry

For any orthogonal matrix `Q` acting on latent coordinates,

`A -> Q A`

leaves

`A^T Q^T Q A = A^T A`.

Therefore `A` is not unique even when its global scale is fixed. A learned latent axis or row of `A` must not be interpreted as a unique estimand.

This is the standard factorization ambiguity of a Mahalanobis metric. Distance-metric-learning literature commonly treats the positive-semidefinite matrix `M = A^T A` as the metric object; the factor `A` is a computational parameterization and is unique only up to latent-space rotation.

## 3. More useful canonical parameterization

Decompose the metric into shape and scale:

`M = s^2 C`

where

`s = sqrt(trace(M)) = ||A||_F`

and

`C = M / trace(M)`.

Then `C` is positive semidefinite and `trace(C)=1`.

The kernel argument becomes

`r_ij = sqrt(delta_ij^T C delta_ij) / b`

with the **effective bandwidth**

`b = h / s`.

This yields two invariant quantities:

1. `C` — metric shape / anisotropy / feature-coupling structure;
2. `b` — smoothing scale measured in the normalized metric.

Both are invariant to `(A,h)->(cA,ch)` and to `A->QA`.

Equivalently, one can work directly with `H=A^T A/h^2`; however, the `(C,b)` decomposition is usually easier to interpret because it separates metric shape from smoothing scale.

## 4. Implication for the current baseline

### 4.1 `A_` should not be the scientific estimand

`A_` is a factor of the learned metric and is non-unique under latent rotations. It can remain an optimisation variable, but paper interpretation should focus on invariant metric objects.

### 4.2 raw `metric_matrix_ = A_.T @ A_` is still scale-dependent

The current output `metric_matrix_` removes rotation ambiguity but not the global `A/h` scaling ambiguity. If the norm of `A` is externally fixed, the ambiguity is numerically broken, but the chosen norm becomes part of the parameterization convention.

### 4.3 current `metric_contributions_` are better behaved but incomplete

The code reports

`diag(M) / trace(M)`.

This is equal to `diag(C)`, so it is invariant to global scaling and latent rotation. However, it ignores off-diagonal terms of `C`, so it cannot represent feature interactions / rotations in the input metric. It should not be described as a complete explanation of the learned geometry.

### 4.4 current Frobenius constraint is a scale-fixing device, not an identification theorem

The baseline projects `A` to the Frobenius norm of its initialization. This removes one numerical scaling degree of freedom during a given optimization run, but:

- it does not remove `A -> Q A` rotational ambiguity;
- the target norm is a parameterization convention rather than a data-identified quantity;
- random restarts can begin with different raw Frobenius norms;
- with a fixed external bandwidth, an arbitrary norm convention changes effective smoothing unless the bandwidth is transformed consistently.

Therefore the Frobenius constraint should currently be described as an optimisation/scale-fixing convention, not as proof that `A` itself is identifiable.

## 5. Separable geometry

The same issue appears in the attribute channel:

`K(d_geo/h_g) * K(||B(a_i-a_j)||/h_a)`.

The attribute component depends on

`H_a = B^T B / h_a^2`.

Thus `(B,h_a)->(cB,c h_a)` and `B->QB` are exact symmetries. The geographic bandwidth `h_g` is a distinct scale and is not part of this particular ambiguity.

A normalized attribute-metric shape

`C_a = B^T B / trace(B^T B)`

and effective attribute bandwidth

`b_a = h_a / ||B||_F`

provide the analogous invariant representation.

## 6. Targeted numerical tests now protecting the mathematics

`tests/test_metric_bandwidth_identifiability.py` verifies:

1. exact joint weight invariance under `(A,h)->(cA,ch)`;
2. exact joint weight invariance under `A->QA`;
3. invariance of `H=A^T A/h^2`;
4. invariance of canonical `(C,b)`;
5. identical local coefficients / hat matrix for scale-equivalent parameterizations;
6. the analogous `(B,h_a)` symmetry in separable geometry.

These tests describe mathematical equivalence classes; they do not freeze the current optimizer design.

## 7. Literature connection

Relevant prior work establishes two separate foundations:

- Distance metric learning commonly represents a learned linear transform through the PSD Mahalanobis matrix `M=L^T L`; the factor `L` is only determined up to rotation. See Weinberger & Saul (2009), *Distance Metric Learning for Large Margin Nearest Neighbor Classification*, JMLR 10:207–244.
- GWR literature has explicitly shown that distance-metric choice and bandwidth choice materially affect neighbourhood definition and model calibration, including Minkowski/non-Euclidean and parameter-specific distance metrics. See Lu et al. (2016/2017) and subsequent PSDM-GWR calibration work.

LG-GWR differs by learning the metric continuously from regression prediction loss, but it inherits the same need to distinguish metric shape from smoothing scale.

## 8. Current recommendation — provisional

Do **not** redesign the optimizer yet. For the next audit stage:

1. treat `A` / `B` as computational factors, not interpretable estimands;
2. add canonical reported quantities `C` and `b` (and separable `C_a`, `b_a`) in an experimental branch before changing the fitted weights;
3. quantify whether current restart ranking or fixed-bandwidth experiments are affected by initialization-dependent Frobenius scale;
4. compare three formulations under controlled simulations:
   - current Frobenius-factor parameterization;
   - factor optimization with explicit canonicalization to `trace(C)=1`;
   - direct/PSD metric-shape formulation with an explicit effective bandwidth;
5. only then decide whether the paper algorithm should retain the current constraint or move to a canonical metric-shape parameterization.

## 9. Decision status

**KEEP FOR NOW, BUT REINTERPRET.**

The current `A/B` factorization remains useful computationally. The scientific object should be treated as an invariant metric representation, with `C + b` currently the leading candidate for the paper-facing parameterization. No final ADR has yet been accepted.
