import numpy as np
from scipy.spatial.distance import cdist

from latentgeometry_gwr import LGGWR


def random_orthogonal(rng, dim):
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q


def test_joint_weights_depend_on_normalized_psd_metric_not_factor_scale_or_rotation():
    rng = np.random.default_rng(20260902)
    n, d, r = 31, 5, 2
    u = rng.normal(size=(n, d))
    A = rng.normal(size=(r, d))
    h = 1.7
    c = 3.4
    Q = random_orthogonal(rng, r)

    model = LGGWR(kernel="gaussian", select_bandwidth=False, bandwidth=h)

    z = u @ A.T
    weights = model._kernel_weights(cdist(z, z), h)

    # Global scale symmetry: (A, h) -> (c A, c h).
    z_scaled = u @ (c * A).T
    weights_scaled = model._kernel_weights(cdist(z_scaled, z_scaled), c * h)
    np.testing.assert_allclose(weights, weights_scaled, atol=1e-13, rtol=1e-13)

    # Left-orthogonal symmetry: A -> Q A.
    z_rotated = u @ (Q @ A).T
    weights_rotated = model._kernel_weights(cdist(z_rotated, z_rotated), h)
    np.testing.assert_allclose(weights, weights_rotated, atol=1e-13, rtol=1e-13)

    # The kernel argument is determined by H = A^T A / h^2.
    H = A.T @ A / h**2
    H_scaled = (c * A).T @ (c * A) / (c * h) ** 2
    H_rotated = (Q @ A).T @ (Q @ A) / h**2
    np.testing.assert_allclose(H, H_scaled, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(H, H_rotated, atol=1e-13, rtol=1e-13)


def test_shape_bandwidth_canonicalization_removes_scale_and_rotation_redundancy():
    rng = np.random.default_rng(9)
    d, r = 6, 3
    A = rng.normal(size=(r, d))
    h = 2.1
    c = 5.0
    Q = random_orthogonal(rng, r)

    def canonical(A_value, h_value):
        M = A_value.T @ A_value
        trace = float(np.trace(M))
        C = M / trace
        scale = np.sqrt(trace)
        b = h_value / scale
        return C, b

    C, b = canonical(A, h)
    C_scaled, b_scaled = canonical(c * A, c * h)
    C_rotated, b_rotated = canonical(Q @ A, h)

    np.testing.assert_allclose(C, C_scaled, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(C, C_rotated, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(b, b_scaled, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(b, b_rotated, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(np.trace(C), 1.0, atol=1e-13, rtol=0.0)


def test_same_weights_imply_same_local_fit_under_scaled_joint_parameterization():
    rng = np.random.default_rng(22)
    n, d, r = 28, 4, 2
    u = rng.normal(size=(n, d))
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = rng.normal(size=n)
    A = rng.normal(size=(r, d))
    h = 1.3
    c = 2.75

    model = LGGWR(kernel="gaussian", select_bandwidth=False, bandwidth=h)
    beta, hat = model._local_fit_with_hat(X, y, u @ A.T, h)
    beta_scaled, hat_scaled = model._local_fit_with_hat(X, y, u @ (c * A).T, c * h)

    np.testing.assert_allclose(beta, beta_scaled, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(hat, hat_scaled, atol=1e-12, rtol=1e-12)


def test_separable_attribute_channel_has_same_B_ha_scale_symmetry():
    rng = np.random.default_rng(31)
    n, q, r = 25, 3, 2
    coords = rng.normal(size=(n, 2))
    attrs = rng.normal(size=(n, q))
    B = rng.normal(size=(r, q))
    hg, ha = 1.8, 1.1
    c = 4.2

    model = LGGWR(kernel="gaussian", geometry="separable", select_bandwidth=False)
    Dg = cdist(coords, coords)
    Kg = model._kernel_weights(Dg, hg)
    z = attrs @ B.T
    W = Kg * model._kernel_weights(cdist(z, z), ha)
    z_scaled = attrs @ (c * B).T
    W_scaled = Kg * model._kernel_weights(cdist(z_scaled, z_scaled), c * ha)

    np.testing.assert_allclose(W, W_scaled, atol=1e-13, rtol=1e-13)
