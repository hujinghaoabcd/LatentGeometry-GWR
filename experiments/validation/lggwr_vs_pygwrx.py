from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from latentgeometry_gwr import LGGWR as StandaloneLGGWR
from pygwrx.models.lg_gwr import LGGWR as PyGWRxLGGWR

SOURCE_COMMIT = "ee26988a0c5b7ed15edf2d6065f538ed0d4d5429"
ATOL = 1.0e-10
RTOL = 1.0e-10


def synthetic_data(seed: int = 20260902, n: int = 54):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-3.0, 5.0, size=(n, 2))
    attrs = rng.normal(size=(n, 2))
    X = rng.normal(size=(n, 2))
    beta0 = 1.2 + 0.8 * attrs[:, 0] - 0.35 * attrs[:, 1]
    beta1 = -0.7 + 0.45 * attrs[:, 1]
    y = 0.4 + beta0 * X[:, 0] + beta1 * X[:, 1] + rng.normal(0.0, 0.08, n)
    return X, y, coords, attrs


def max_abs(a: Any, b: Any) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size == 0 and bb.size == 0:
        return 0.0
    return float(np.max(np.abs(aa - bb)))


def assert_array(name: str, left: Any, right: Any, record: dict[str, Any]) -> None:
    diff = max_abs(left, right)
    record[name] = diff
    np.testing.assert_allclose(left, right, atol=ATOL, rtol=RTOL, err_msg=name)


def assert_scalar(name: str, left: float, right: float, record: dict[str, Any]) -> None:
    diff = abs(float(left) - float(right))
    record[name] = diff
    np.testing.assert_allclose(float(left), float(right), atol=ATOL, rtol=RTOL, err_msg=name)


def compare_fit(case_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    X, y, coords, attrs = synthetic_data()
    source = PyGWRxLGGWR(**kwargs).fit(X, y, coords, attrs)
    standalone = StandaloneLGGWR(**kwargs).fit(X, y, coords, attrs)

    record: dict[str, Any] = {"case": case_name, "kwargs": kwargs.copy()}
    matrix_source = source.A_ if source.geometry == "joint" else source.B_
    matrix_standalone = standalone.A_ if standalone.geometry == "joint" else standalone.B_

    assert_array("matrix_max_abs_diff", standalone=matrix_standalone, right=matrix_source, record=record)

    if isinstance(source.bandwidth_, tuple):
        assert_array("bandwidth_max_abs_diff", standalone.bandwidth_, source.bandwidth_, record)
    else:
        assert_scalar("bandwidth_abs_diff", standalone.bandwidth_, source.bandwidth_, record)

    assert_array("latent_coords_max_abs_diff", standalone.latent_coords_, source.latent_coords_, record)
    assert_array("coefficients_max_abs_diff", standalone.coefficients_, source.coefficients_, record)
    assert_array("coef_max_abs_diff", standalone.coef_, source.coef_, record)
    assert_array("intercept_max_abs_diff", standalone.intercept_, source.intercept_, record)
    assert_array("fitted_values_max_abs_diff", standalone.fitted_values_, source.fitted_values_, record)
    assert_array("residuals_max_abs_diff", standalone.residuals_, source.residuals_, record)
    assert_array("hat_matrix_max_abs_diff", standalone.hat_matrix_, source.hat_matrix_, record)
    assert_array("metric_matrix_max_abs_diff", standalone.metric_matrix_, source.metric_matrix_, record)
    assert_array(
        "metric_contributions_max_abs_diff",
        standalone.metric_contributions_,
        source.metric_contributions_,
        record,
    )
    assert_array("loss_history_max_abs_diff", standalone.loss_history_, source.loss_history_, record)
    assert_scalar("best_loss_abs_diff", standalone.best_loss_, source.best_loss_, record)
    assert_scalar("final_loo_loss_abs_diff", standalone.final_loo_loss_, source.final_loo_loss_, record)

    for key in sorted(set(standalone.diagnostics_) & set(source.diagnostics_)):
        assert_scalar(f"diagnostic_{key}_abs_diff", standalone.diagnostics_[key], source.diagnostics_[key], record)

    assert standalone.n_iter_ == source.n_iter_
    assert standalone.converged_ == source.converged_
    assert standalone.stop_reason_ == source.stop_reason_
    assert len(standalone.bandwidth_history_) == len(source.bandwidth_history_)
    for index, (left, right) in enumerate(zip(standalone.bandwidth_history_, source.bandwidth_history_)):
        if isinstance(right, tuple):
            assert_array(f"bandwidth_history_{index}_max_abs_diff", left, right, record)
        else:
            assert_scalar(f"bandwidth_history_{index}_abs_diff", left, right, record)

    pred_source = source.predict(X, coords, attrs)
    pred_standalone = standalone.predict(X, coords, attrs)
    assert_array("prediction_max_abs_diff", pred_standalone, pred_source, record)

    record["n_iter"] = standalone.n_iter_
    record["converged"] = standalone.converged_
    record["stop_reason"] = standalone.stop_reason_
    record["passes"] = True
    return record


def main() -> None:
    cases = [
        (
            "joint_fixed",
            dict(
                latent_dim=2,
                bandwidth=2.25,
                kernel="gaussian",
                geometry="joint",
                learning_rate=0.03,
                max_iter=14,
                patience=20,
                select_bandwidth=False,
                bandwidth_updates=0,
                initialization="coordinate",
                n_restarts=1,
                random_state=17,
            ),
        ),
        (
            "joint_aicc_reselection",
            dict(
                latent_dim=2,
                bandwidth=None,
                kernel="gaussian",
                geometry="joint",
                learning_rate=0.03,
                max_iter=8,
                patience=20,
                select_bandwidth=True,
                bandwidth_updates=1,
                initialization="coordinate",
                n_restarts=1,
                random_state=17,
            ),
        ),
        (
            "separable_fixed",
            dict(
                latent_dim=2,
                bandwidth=(2.0, 1.4),
                kernel="gaussian",
                geometry="separable",
                learning_rate=0.03,
                max_iter=12,
                patience=20,
                select_bandwidth=False,
                bandwidth_updates=0,
                initialization="random",
                n_restarts=1,
                random_state=17,
            ),
        ),
        (
            "separable_aicc_reselection",
            dict(
                latent_dim=2,
                bandwidth=None,
                kernel="gaussian",
                geometry="separable",
                learning_rate=0.03,
                max_iter=7,
                patience=20,
                select_bandwidth=True,
                bandwidth_updates=1,
                initialization="random",
                n_restarts=1,
                random_state=17,
            ),
        ),
    ]

    summary = {
        "source_repository": "hujinghaoabcd/pyGWRx",
        "source_commit": SOURCE_COMMIT,
        "atol": ATOL,
        "rtol": RTOL,
        "cases": [],
    }
    for name, kwargs in cases:
        print(f"[parity] running {name}", flush=True)
        summary["cases"].append(compare_fit(name, kwargs))

    summary["passes_validation"] = all(case["passes"] for case in summary["cases"])
    out_dir = Path("results/validation/lggwr_vs_pygwrx")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
