import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from latentgeometry_gwr import LGGWR, LGGWRPredictionResult


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    n = 70
    coords = rng.random((n, 2)) * 10
    attrs = rng.random((n, 2))
    X = rng.random((n, 2))
    beta0 = 1 + 2 * attrs[:, 0]
    y = 0.5 + beta0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.05, n)
    return X, y, coords, attrs


@pytest.mark.parametrize("kernel", ["gaussian", "bisquare", "exponential"])
def test_joint_gradient(kernel):
    rng = np.random.default_rng(0)
    n = 25
    X = rng.random((n, 2))
    y = rng.random(n)
    coords = rng.random((n, 2)) * 5
    attrs = rng.random((n, 2))
    u = np.hstack([coords, attrs])
    model = LGGWR(
        kernel=kernel,
        lambda_reg=0.01,
        scale_constraint="none",
        fit_intercept=False,
    )
    model.A_ = model._initialize_A(u.shape[1], np.random.default_rng(1), mode="random")
    h = float(np.median(np.linalg.norm(u @ model.A_.T, axis=1))) + 1
    A0 = model.A_.copy()
    cache = model._forward_loo(X, y, u @ A0.T, h)
    analytical = model._compute_gradient(X, y, u, u @ A0.T, h, cache)
    numerical = np.zeros_like(A0)
    eps = 1e-6

    def loss(A):
        model.A_ = A
        return model._compute_loss(y, model._forward_loo(X, y, u @ A.T, h)["yhat"])

    for i in range(A0.shape[0]):
        for j in range(A0.shape[1]):
            plus = A0.copy()
            minus = A0.copy()
            plus[i, j] += eps
            minus[i, j] -= eps
            numerical[i, j] = (loss(plus) - loss(minus)) / (2 * eps)
    assert np.allclose(analytical, numerical, atol=1e-4, rtol=1e-3)


def test_training_and_unit_invariance(data):
    X, y, coords, attrs = data
    model = LGGWR(max_iter=35, random_state=0).fit(X, y, coords, attrs)
    assert model.best_loss_ <= model.loss_history_[0]
    assert model.diagnostics_["r2"] > 0.9

    transformed = LGGWR(max_iter=35, random_state=0).fit(
        X,
        y,
        coords * 1000 + np.array([2e6, -4e6]),
        attrs * np.array([100, 0.01]) + np.array([9, -7]),
    )
    assert np.allclose(model.fitted_values_, transformed.fitted_values_, atol=1e-7)


def test_restarts_reproducible(data):
    X, y, coords, attrs = data
    kwargs = dict(max_iter=15, n_restarts=2, random_state=42)
    first = LGGWR(**kwargs).fit(X, y, coords, attrs)
    second = LGGWR(**kwargs).fit(X, y, coords, attrs)
    assert np.allclose(first.A_, second.A_)
    assert first.restart_scores_ == second.restart_scores_


def test_dataframe_prediction(data):
    X, y, coords, attrs = data
    X = pd.DataFrame(X, columns=["income", "housing"])
    attrs = pd.DataFrame(attrs, columns=["context", "noise"])
    model = LGGWR(max_iter=10, random_state=0).fit(X, y, coords, attrs)
    result = model.predict_result(X, coords, attrs)
    assert isinstance(result, LGGWRPredictionResult)
    assert np.allclose(result.predictions, model.fitted_values_)
    assert "coef_income" in model.to_frame()


def test_separable_reduces_to_geo_gwr(data):
    X, y, coords, attrs = data
    model = LGGWR(
        geometry="separable",
        bandwidth_updates=0,
        select_bandwidth=False,
        max_iter=5,
        random_state=0,
    ).fit(X, y, coords, attrs)
    h_g, _ = model.bandwidth_
    geographic_distances = cdist(model.coords_geometry_, model.coords_geometry_)
    zeta = model.attrs_geometry_ @ model.B_.T
    betas, _ = model._local_fit_with_hat_sep(
        model.X_design_,
        model.y_train_,
        geographic_distances,
        zeta,
        h_g,
        np.inf,
    )
    weights = model._kernel_weights(geographic_distances, h_g)
    yhat = np.zeros(len(y))
    for i in range(len(y)):
        beta, _ = model._hat_solution(
            model.X_design_, model.y_train_, weights[i], model.X_design_[i]
        )
        yhat[i] = model.X_design_[i] @ beta
    assert np.allclose(
        np.einsum("ij,ij->i", model.X_design_, betas), yhat, atol=1e-8
    )


def test_failed_refit_clears_state(data):
    X, y, coords, attrs = data
    model = LGGWR(max_iter=3, random_state=0).fit(X, y, coords, attrs)
    with pytest.raises(ValueError):
        model.fit(X[:-1], y, coords, attrs)
    assert not model._is_fitted
    assert model.A_ is None
