"""Anchor the standalone BasicGWR to GeoRegime-GWR and mgwr 2.2.1.

This is a baseline-validation experiment, not an LG-GWR performance experiment.
It uses the canonical Georgia dataset and checks two distinct conventions:

1. research default: exhaustive integer adaptive AICc search;
2. historical compatibility: mgwr 2.2.1 adaptive golden-section search.

The standalone implementation must match the pinned GeoRegime-GWR reference;
its compatibility mode must additionally reproduce mgwr.GWR to machine precision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

from georegime_gwr.gwr import BasicGWR as GeoRegimeBasicGWR
from latentgeometry_gwr import BasicGWR as StandaloneBasicGWR

ATOL = 1.0e-12
RTOL = 1.0e-12
GEOREGIME_COMMIT = "428336399da87eb4ada4f97dfc5cc1993fa4b7e9"


def max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def assert_close(name, a, b, record):
    diff = max_abs(a, b)
    record[name] = diff
    np.testing.assert_allclose(a, b, atol=ATOL, rtol=RTOL, err_msg=name)


def load_georgia(path: Path):
    df = pd.read_csv(path)
    y = df["PctBach"].to_numpy(dtype=float).reshape(-1, 1)
    X = df[["PctFB", "PctBlack", "PctRural"]].to_numpy(dtype=float)
    coords = df[["X", "Y"]].to_numpy(dtype=float)
    Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    yz = (y - y.mean(axis=0)) / y.std(axis=0, ddof=0)
    return Xz, yz, coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--georgia", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/validation/basicgwr_reference_anchor/summary.json"),
    )
    args = parser.parse_args()

    Xz, yz, coords = load_georgia(args.georgia)
    y1 = yz.reshape(-1)

    record = {
        "georegime_reference_commit": GEOREGIME_COMMIT,
        "mgwr_version": "2.2.1",
        "atol": ATOL,
        "rtol": RTOL,
    }

    # Research-default exhaustive adaptive search: standalone must equal the
    # already validated GeoRegime baseline exactly.
    standalone_strict = StandaloneBasicGWR(
        bandwidth="auto", kernel="bisquare", fit_intercept=True
    ).fit(Xz, y1, coords)
    georegime_strict = GeoRegimeBasicGWR(
        bandwidth="auto", kernel="bisquare", fit_intercept=True
    ).fit(Xz, y1, coords)

    strict = {
        "standalone_bandwidth": int(standalone_strict.bandwidth_),
        "georegime_bandwidth": int(georegime_strict.bandwidth_),
    }
    if int(standalone_strict.bandwidth_) != int(georegime_strict.bandwidth_):
        raise AssertionError("strict research-default bandwidth mismatch")
    if int(standalone_strict.bandwidth_) != 116:
        raise AssertionError("canonical Georgia exhaustive optimum is expected to be 116")
    assert_close("parameter_max_abs_difference", standalone_strict.parameters_, georegime_strict.parameters_, strict)
    assert_close("fitted_max_abs_difference", standalone_strict.fitted_values_, georegime_strict.fitted_values_, strict)
    assert_close("residual_max_abs_difference", standalone_strict.residuals_, georegime_strict.residuals_, strict)
    assert_close("hat_max_abs_difference", standalone_strict.hat_matrix_, georegime_strict.hat_matrix_, strict)
    strict["passes"] = True
    record["research_default_exhaustive"] = strict

    # mgwr compatibility search and final GWR fit.
    selector = Sel_BW(coords, yz, Xz, fixed=False, kernel="bisquare")
    mgwr_bw = int(selector.search(criterion="AICc"))
    mgwr_fit = GWR(
        coords,
        yz,
        Xz,
        mgwr_bw,
        fixed=False,
        kernel="bisquare",
        constant=True,
        hat_matrix=True,
        n_jobs=1,
    ).fit()

    standalone_compat = StandaloneBasicGWR(
        bandwidth="auto",
        kernel="bisquare",
        fit_intercept=True,
        adaptive=True,
        search_strategy="mgwr_golden",
    ).fit(Xz, y1, coords)
    georegime_compat = GeoRegimeBasicGWR(
        bandwidth="auto",
        kernel="bisquare",
        fit_intercept=True,
        adaptive=True,
        search_strategy="mgwr_golden",
    ).fit(Xz, y1, coords)

    compat = {
        "mgwr_bandwidth": mgwr_bw,
        "standalone_bandwidth": int(standalone_compat.bandwidth_),
        "georegime_bandwidth": int(georegime_compat.bandwidth_),
    }
    if not (mgwr_bw == int(standalone_compat.bandwidth_) == int(georegime_compat.bandwidth_) == 117):
        raise AssertionError("mgwr-compatible Georgia bandwidth must be 117")

    assert_close("standalone_vs_georegime_parameter", standalone_compat.parameters_, georegime_compat.parameters_, compat)
    assert_close("standalone_vs_georegime_fitted", standalone_compat.fitted_values_, georegime_compat.fitted_values_, compat)
    assert_close("standalone_vs_georegime_hat", standalone_compat.hat_matrix_, georegime_compat.hat_matrix_, compat)

    ref_params = np.asarray(mgwr_fit.params)
    ref_fitted = np.asarray(mgwr_fit.predy).reshape(-1)
    ref_resid = np.asarray(mgwr_fit.resid_response).reshape(-1)
    ref_hat = np.asarray(mgwr_fit.S)
    assert_close("standalone_vs_mgwr_parameter", standalone_compat.parameters_, ref_params, compat)
    assert_close("standalone_vs_mgwr_fitted", standalone_compat.fitted_values_, ref_fitted, compat)
    assert_close("standalone_vs_mgwr_residual", standalone_compat.residuals_, ref_resid, compat)
    assert_close("standalone_vs_mgwr_hat", standalone_compat.hat_matrix_, ref_hat, compat)

    compat["standalone_aicc"] = float(standalone_compat.diagnostics_["aicc"])
    compat["mgwr_aicc"] = float(mgwr_fit.aicc)
    compat["aicc_abs_difference"] = abs(compat["standalone_aicc"] - compat["mgwr_aicc"])
    if compat["aicc_abs_difference"] > ATOL:
        raise AssertionError("AICc mismatch against mgwr")
    compat["passes"] = True
    record["mgwr_compatibility"] = compat

    # Fixed-distance path is not claimed externally validated by GeoRegime, but
    # the standalone copy must still preserve the reference implementation.
    standalone_fixed = StandaloneBasicGWR(
        bandwidth="auto", kernel="bisquare", fit_intercept=True, adaptive=False
    ).fit(Xz, y1, coords)
    georegime_fixed = GeoRegimeBasicGWR(
        bandwidth="auto", kernel="bisquare", fit_intercept=True, adaptive=False
    ).fit(Xz, y1, coords)
    fixed = {
        "standalone_bandwidth": float(standalone_fixed.bandwidth_),
        "georegime_bandwidth": float(georegime_fixed.bandwidth_),
    }
    if abs(float(standalone_fixed.bandwidth_) - float(georegime_fixed.bandwidth_)) > ATOL:
        raise AssertionError("fixed golden-section bandwidth drift")
    assert_close("parameter_max_abs_difference", standalone_fixed.parameters_, georegime_fixed.parameters_, fixed)
    assert_close("fitted_max_abs_difference", standalone_fixed.fitted_values_, georegime_fixed.fitted_values_, fixed)
    assert_close("hat_max_abs_difference", standalone_fixed.hat_matrix_, georegime_fixed.hat_matrix_, fixed)
    fixed["passes"] = True
    record["fixed_reference_parity_only"] = fixed

    record["passes_validation"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
