# LGGWR BASELINE SPEC

> This document freezes the **extracted research baseline**, not the final paper method.

## 1. Source anchor

The baseline was extracted from the pyGWRx implementation:

- repository: `hujinghaoabcd/pyGWRx`
- source path: `src/pygwrx/models/lg_gwr.py`
- source snapshot observed during extraction: commit `ee26988a0c5b7ed15edf2d6065f538ed0d4d5429`
- source model: `pygwrx.models.LGGWR`

The standalone repository intentionally removes package-wide software helpers while preserving the research-critical mathematical behaviour.

## 2. Joint latent geometry

For location/context input

`u_i = [s_i, a_i]`,

the current joint baseline learns a linear map

`z_i = A u_i`.

Pairwise latent distance is

`d_ij = ||z_i - z_j||_2`,

and local weight is

`w_ij = K(d_ij / h)`.

The local regression at location `i` is ordinary locally weighted least squares using the learned latent-distance weights.

## 3. Separable geometry

The separable baseline retains a geographic channel and learns only an attribute channel:

`zeta_i = B a_i`.

Weights are multiplicative:

`w_ij = K(d_geo,ij / h_g) * K(||zeta_i-zeta_j|| / h_a)`.

When `h_a = infinity`, the attribute kernel is exactly one, so the model reduces to geographic GWR at the same `h_g`.

This reduction is treated as a mandatory regression test.

## 4. Geometry-learning objective

The current baseline trains `A` or `B` using leave-one-out local-regression prediction error.

For joint geometry:

`L(A) = mean_i (y_i - yhat_i^(-i)(A))^2 + lambda ||A||_F^2`.

For separable geometry the same structure is used with `B`.

The implementation uses an analytical gradient through the local weighted least-squares solution. Analytical gradients are tested against central finite differences.

## 5. Scale identification

The baseline exposes three policies:

- `frobenius`: keep the transformation Frobenius norm fixed;
- `orthogonal`: project onto an orthogonal-row geometry where feasible;
- `none`: leave scale unconstrained.

Ordinary L2 regularisation is disallowed when the norm is already fixed, because it would be constant under the constraint.

This is an implementation-level identification guard, not a claim that the paper's final identification strategy is settled.

## 6. Geometry standardisation

Coordinates are centred and divided by one shared coordinate scale so geographic shape is preserved. Context attributes are z-standardised column-wise.

The baseline must be invariant to coordinate translation/common rescaling and to affine unit changes in contextual attributes when standardisation is enabled.

## 7. Optimisation

Current baseline optimisation:

1. initialise `A` or `B`;
2. resolve an initial distance bandwidth;
3. compute LOO local regressions;
4. compute analytical gradient;
5. Adam update;
6. gradient clipping;
7. scale projection;
8. early stopping by tolerance/patience;
9. optionally reselect bandwidth by Gaussian GWR AICc;
10. refit local coefficients using the final geometry and bandwidth.

This optimisation sequence is **not frozen as the final LG-GWR paper algorithm**.

## 8. Research baseline invariants

The extracted repository currently protects the following invariants:

- joint analytical gradient agrees with finite difference for Gaussian, bisquare and exponential kernels;
- separable analytical gradient agrees with finite difference for Gaussian and bisquare kernels;
- training does not lose the best encountered LOO state;
- deterministic restarts are reproducible;
- geometry standardisation gives unit-invariant fitted values;
- prediction reproduces training-location fitted values;
- separable `h_a = infinity` reduces to geographic GWR;
- a failed refit clears fitted state;
- the standard GWR baseline produces finite local fits.

## 9. Deliberately not frozen

The following remain open research questions and may be replaced:

- joint vs geography-preserving/separable formulation;
- linear map vs another metric parameterisation;
- latent dimension;
- Frobenius/orthogonal identification strategy;
- LOO-MSE objective;
- AICc bandwidth alternation;
- Adam optimisation;
- kernel family;
- context-variable treatment;
- regularisation and structural sparsity.

See `RESEARCH_QUESTIONS.md` before changing any of these components.
