"""Bandwidth selection for the trusted standard-GWR research baseline.

This module is intentionally aligned with the already validated implementation in
hujinghaoabcd/GeoRegime-GWR. Three policies are explicit:

1. PyGWRxAdaptiveAICcSelector: exhaustive integer adaptive AICc search.
2. FixedGoldenAICcSelector: continuous fixed-distance golden-section search.
3. MGWRCompatibleAICcSelector: mgwr==2.2.1-compatible adaptive search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class BandwidthSearchResult:
    bandwidth: int | float
    score: float
    search_range: tuple[int | float, int | float]
    search_trace: tuple[tuple[int | float, float], ...]
    strategy: str


def _distance_bandwidth_pygwrx(distances: np.ndarray, k: int) -> float:
    if k < 1 or k > distances.size:
        raise ValueError(f"k must satisfy 1 <= k <= {distances.size}; got {k}")
    bandwidth = float(np.partition(distances, k - 1)[k - 1])
    if bandwidth <= 0.0:
        positive = distances[distances > 0.0]
        if positive.size == 0:
            raise np.linalg.LinAlgError("all distances are zero")
        bandwidth = float(np.min(positive))
    return float(np.nextafter(bandwidth, np.inf))


def _distance_bandwidth_mgwr(distances: np.ndarray, k: int) -> float:
    if k < 1 or k > distances.size:
        raise ValueError(f"k must satisfy 1 <= k <= {distances.size}; got {k}")
    bandwidth = float(np.partition(distances, k - 1)[k - 1])
    if bandwidth <= 0.0:
        positive = distances[distances > 0.0]
        if positive.size == 0:
            raise np.linalg.LinAlgError("all distances are zero")
        bandwidth = float(np.min(positive))
    return bandwidth * 1.0000001


def _kernel_from_distance_bandwidth(distances, distance_bandwidth, kernel):
    ratio = distances / float(distance_bandwidth)
    if kernel == "bisquare":
        return np.where(ratio < 1.0, (1.0 - ratio**2) ** 2, 0.0)
    if kernel == "gaussian":
        return np.exp(-0.5 * ratio**2)
    if kernel == "exponential":
        return np.exp(-ratio)
    raise ValueError("kernel must be bisquare, gaussian, or exponential")


def _fit_local_unpenalized(X, y, weights, target_row):
    p = X.shape[1]
    positive = weights > 0.0
    if np.count_nonzero(positive) < p:
        raise np.linalg.LinAlgError("too few positive-weight observations")
    Xw_rank = X[positive] * np.sqrt(weights[positive])[:, None]
    if np.linalg.matrix_rank(Xw_rank) < p:
        raise np.linalg.LinAlgError("rank-deficient weighted design")
    Xtw = X.T * weights
    normal = Xtw @ X
    inverse_normal = np.linalg.inv(normal)
    beta = inverse_normal @ Xtw @ y
    hat_row = target_row @ inverse_normal @ Xtw
    return beta, hat_row


def _aicc(y, fitted, trace_s):
    n = y.size
    residuals = y - fitted
    rss = max(float(residuals @ residuals), np.finfo(float).tiny)
    denominator = n - 2.0 - float(trace_s)
    if denominator <= 0.0:
        return np.inf
    return float(
        n * np.log(rss / n)
        + n * np.log(2.0 * np.pi)
        + n * (n + float(trace_s)) / denominator
    )


def _score_candidate(X, y, distances, bandwidth, kernel, *, policy):
    n = y.size
    fitted = np.zeros(n, dtype=float)
    trace_s = 0.0
    for i in range(n):
        if policy == "pygwrx_adaptive":
            bw = _distance_bandwidth_pygwrx(distances[i], int(bandwidth))
        elif policy == "mgwr_adaptive":
            bw = _distance_bandwidth_mgwr(distances[i], int(bandwidth))
        elif policy == "fixed":
            bw = float(bandwidth)
            if bw <= 0.0:
                return np.inf
        else:
            raise ValueError("unknown bandwidth policy")
        weights = _kernel_from_distance_bandwidth(distances[i], bw, kernel)
        beta, hat_row = _fit_local_unpenalized(X, y, weights, X[i])
        fitted[i] = X[i] @ beta
        trace_s += float(hat_row[i])
    return _aicc(y, fitted, trace_s)


def _validate_inputs(X, y, coords):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    coords = np.asarray(coords, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2)")
    if X.shape[0] != y.size or coords.shape[0] != y.size:
        raise ValueError("X, y, and coords must have the same rows")
    return X, y, coords


class PyGWRxAdaptiveAICcSelector:
    def __init__(self, kernel="bisquare"):
        if kernel not in {"bisquare", "gaussian", "exponential"}:
            raise ValueError("kernel must be bisquare, gaussian, or exponential")
        self.kernel = kernel

    def select(self, X, y, coords):
        X, y, coords = _validate_inputs(X, y, coords)
        n, p = X.shape
        lower = max(p + 1, 2, int(np.ceil(0.05 * n)))
        upper = n
        distances = cdist(coords, coords)
        trace = []
        best_k = None
        best_score = np.inf
        for k in range(lower, upper + 1):
            try:
                score = _score_candidate(X, y, distances, k, self.kernel, policy="pygwrx_adaptive")
            except np.linalg.LinAlgError:
                score = np.inf
            trace.append((k, float(score)))
            if np.isfinite(score) and score < best_score:
                best_k, best_score = k, float(score)
        if best_k is None:
            raise RuntimeError("bandwidth selection failed for every candidate")
        self.result_ = BandwidthSearchResult(
            int(best_k), float(best_score), (int(lower), int(upper)), tuple(trace),
            "pygwrx_exhaustive_integer_aicc",
        )
        return self.result_


class FixedGoldenAICcSelector:
    RESPHI = (np.sqrt(5.0) - 1.0) / 2.0

    def __init__(self, kernel="bisquare", tol=1.0e-4, max_iter=100):
        if kernel not in {"bisquare", "gaussian", "exponential"}:
            raise ValueError("kernel must be bisquare, gaussian, or exponential")
        self.kernel = kernel
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def select(self, X, y, coords):
        X, y, coords = _validate_inputs(X, y, coords)
        distances = cdist(coords, coords)
        positive = distances[distances > 0.0]
        if positive.size == 0:
            raise ValueError("fixed bandwidth search requires distinct coordinates")
        lower = 0.5 * float(np.min(positive))
        upper = 2.0 * float(np.max(distances))
        cache = {}

        def objective(value):
            key = float(value)
            if key not in cache:
                try:
                    score = _score_candidate(X, y, distances, key, self.kernel, policy="fixed")
                except np.linalg.LinAlgError:
                    score = np.inf
                cache[key] = float(score)
            return cache[key]

        a, b = lower, upper
        objective(a); objective(b)
        c = b - self.RESPHI * (b - a)
        d = a + self.RESPHI * (b - a)
        fc, fd = objective(c), objective(d)
        iterations = 0
        while iterations < self.max_iter:
            midpoint = 0.5 * (a + b)
            if (b - a) <= self.tol * (1.0 + abs(midpoint)):
                break
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b - self.RESPHI * (b - a)
                fc = objective(c)
            else:
                a, c, fc = c, d, fd
                d = a + self.RESPHI * (b - a)
                fd = objective(d)
            iterations += 1
        finite = [(bw, score) for bw, score in cache.items() if np.isfinite(score)]
        if not finite:
            raise RuntimeError("fixed golden-section bandwidth search failed")
        best_bw, best_score = min(finite, key=lambda item: (item[1], item[0]))
        self.result_ = BandwidthSearchResult(
            float(best_bw), float(best_score), (float(lower), float(upper)),
            tuple(sorted(cache.items(), key=lambda item: item[0])),
            "pygwrx_fixed_golden_aicc",
        )
        return self.result_


class MGWRCompatibleAICcSelector:
    def __init__(self, kernel="bisquare", tol=1.0e-6, max_iter=200, bw_min=None, bw_max=None):
        if kernel not in {"bisquare", "gaussian", "exponential"}:
            raise ValueError("kernel must be bisquare, gaussian, or exponential")
        self.kernel = kernel
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.bw_min = bw_min
        self.bw_max = bw_max

    def select(self, X, y, coords):
        X, y, coords = _validate_inputs(X, y, coords)
        n, p = X.shape
        a = float(40 + 2 * p if self.bw_min is None else self.bw_min)
        c = float(n if self.bw_max is None else self.bw_max)
        if a > c:
            raise ValueError("sample size/search bounds are invalid for mgwr-compatible search")
        distances = cdist(coords, coords)
        cache = {}
        evaluation_order = []

        def objective(value):
            k = int(np.round(value))
            if k not in cache:
                try:
                    score = _score_candidate(X, y, distances, k, self.kernel, policy="mgwr_adaptive")
                except np.linalg.LinAlgError:
                    score = np.inf
                cache[k] = float(score)
                evaluation_order.append((k, float(score)))
            return cache[k]

        delta = 0.38197
        b = a + delta * abs(c - a)
        d = c - delta * abs(c - a)
        opt_val = None
        opt_score = np.inf
        diff = 1.0e9
        iters = 0
        while abs(diff) > self.tol and iters < self.max_iter:
            iters += 1
            b = float(np.round(b)); d = float(np.round(d))
            score_b, score_d = objective(b), objective(d)
            if score_b <= score_d:
                opt_val, opt_score = b, score_b
                c, d = d, b
                b = a + delta * abs(c - a)
            else:
                opt_val, opt_score = d, score_d
                a, b = b, d
                d = c - delta * abs(c - a)
            diff = score_b - score_d
        if opt_val is None:
            raise RuntimeError("mgwr-compatible golden-section bandwidth search failed")
        best_k = int(np.round(opt_val))
        self.result_ = BandwidthSearchResult(
            best_k, float(opt_score),
            (int(40 + 2 * p if self.bw_min is None else self.bw_min), int(n if self.bw_max is None else self.bw_max)),
            tuple(evaluation_order), "mgwr_2_2_1_discrete_golden_aicc",
        )
        return self.result_
