import numpy as np

from latentgeometry_gwr import BasicGWR


def test_basic_gwr_smoke():
    rng = np.random.default_rng(1)
    n = 35
    coords = rng.random((n, 2)) * 10
    X = rng.normal(size=(n, 2))
    y = 1 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=n)
    model = BasicGWR(12, kernel="bisquare").fit(X, y, coords)
    assert model.parameters_.shape == (n, 3)
    assert np.all(np.isfinite(model.fitted_values_))
