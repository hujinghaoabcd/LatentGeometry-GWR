import numpy as np
import pytest
from scipy.spatial.distance import cdist

from latentgeometry_gwr import LGGWR


@pytest.mark.parametrize("kernel", ["gaussian", "bisquare"])
def test_separable_gradient_matches_finite_difference(kernel):
    rng = np.random.default_rng(0)
    n, p, q = 22, 2, 2
    coords = rng.random((n, 2)) * 5.0
    attrs = rng.random((n, q))
    X = rng.random((n, p))
    y = rng.random(n)

    model = LGGWR(
        latent_dim=2,
        geometry="separable",
        kernel=kernel,
        lambda_reg=0.01,
        scale_constraint="none",
        fit_intercept=False,
        random_state=1,
    )
    model.B_ = model._initialize_B(q, np.random.default_rng(1), mode="random")
    dg = cdist(coords, coords)
    z0 = attrs @ model.B_.T
    h_g = 2.0 * float(dg.max())
    h_a = 2.0 * float(cdist(z0, z0).max())
    Kg = model._kernel_weights(dg, h_g)

    def loss_at(B):
        model.B_ = B
        cache = model._forward_loo_sep(X, y, Kg, attrs @ B.T, h_a)
        return float(np.mean((y - cache["yhat"]) ** 2) + 0.01 * np.sum(B**2))

    B0 = model.B_.copy()
    cache0 = model._forward_loo_sep(X, y, Kg, attrs @ B0.T, h_a)
    analytical = model._compute_gradient_sep(X, y, attrs, attrs @ B0.T, h_a, cache0)

    eps = 1e-6
    numerical = np.zeros_like(B0)
    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            plus = B0.copy()
            minus = B0.copy()
            plus[i, j] += eps
            minus[i, j] -= eps
            numerical[i, j] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    assert np.allclose(analytical, numerical, atol=1e-4, rtol=1e-3)
