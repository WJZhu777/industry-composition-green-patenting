from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from . import sample

SEED = 20261028

REPS = 9999

def plan():
    return {
        "tests": ["Full accounting: count, log1p count, entry",
                  "Industry FE plus OI/OD: count, log1p count, entry",
                  "Full accounting: log(count) among positive counts",
                  "Own history plus patent GTOS: count on the matched R&D and patent-history sample",
                  "Guide-only complete profiles: three outcomes for each of two roles"],
        "null": "Agricultural coefficient equals zero; two-sided, pointwise tests.",
        "method": "WCR11 (classic restricted wild cluster bootstrap-t); unadjusted restricted residuals; CV1 studentization in the original fit and every draw.",
        "cluster": sample.INDUSTRY,
        "weights": "Rademacher primary, Webb six-point sensitivity for every test. Enumerate all Rademacher signs when G <= 13; otherwise Monte Carlo.",
        "monte_carlo_draws": REPS, "seed": SEED,
        "p_value": "Fraction with abs(t*) >= abs(t), allowing 1e-10 numerical tie tolerance. No failed draw may be discarded.",
        "fixed": "Original samples, outcomes, regressors, industry codes, GTOS and guide profiles. No data construction or model selection.",
        "not_tested": "No test of coefficient changes, joint GTOS blocks, Gelbach/Oaxaca components or nonlinear count models; no confidence-interval inversion or multiplicity correction.",
        "boundary": "Conditional sensitivity under independent industry clusters, not exact randomization inference or a guarantee of finite-sample validity. Does not fix selection, measurement overlap or cross-industry dependence.",
        "reference": "MacKinnon, Nielsen and Webb (2023), Journal of Econometrics 232:272-299, Section 5.2; doi:10.1016/j.jeconom.2022.04.001",
    }

def prepare(x, y, groups, target):
    n, k = x.shape
    if n <= k or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid finite-sample design.")
    if np.linalg.matrix_rank(x) != k:
        raise ValueError("Rank-deficient design; do not silently remove regressors.")
    codes, levels = pd.factorize(groups, sort=True)
    g = len(levels)
    if g < 2 or (codes < 0).any():
        raise ValueError("Missing clusters or fewer than two clusters.")
    q, r = np.linalg.qr(x, mode="reduced")
    a = np.eye(k)[target]
    contrast = solve_triangular(r.T, a, lower=True)
    theta = q.T @ y
    coefficient = float(contrast @ theta)
    theta0 = theta - contrast * (coefficient / (contrast @ contrast))
    residual = y - q @ theta
    residual0 = y - q @ theta0
    score0 = np.zeros((g, k))
    score = np.zeros((g, k))
    np.add.at(score0, codes, q * residual0[:, None])
    np.add.at(score, codes, q * residual[:, None])
    block_contrast = np.array([q[codes == c].T @ (q[codes == c] @ contrast) for c in range(g)])
    correction = g / (g - 1) * (n - 1) / (n - k)
    se = float(np.sqrt(correction * np.sum((score @ contrast) ** 2)))
    if se <= 0 or not np.isfinite(se):
        raise ValueError("Zero or invalid observed cluster variance.")
    leverage = block_contrast @ contrast / (contrast @ contrast)
    np.testing.assert_allclose(leverage.sum(), 1., rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(contrast @ theta0, 0., atol=1e-10)
    return {"coefficient": coefficient, "industry_se": se, "t": coefficient / se,
            "codes": codes, "levels": levels, "q": q, "theta0": theta0, "residual0": residual0,
            "h": score0 @ contrast, "cross": block_contrast @ score0.T,
            "correction": correction, "partial_leverage": leverage}

def bootstrap_t(fit, weights):
    h, cross = fit["h"], fit["cross"]
    if weights.ndim != 2 or weights.shape[0] != len(h) or not np.isfinite(weights).all():
        raise ValueError("Invalid cluster weights.")
    # Each column is a full-industry perturbation of the restricted residuals.
    # This score calculation equals refitting OLS and CV1 for every draw.
    scores = h[:, None] * weights - cross @ weights
    variances = fit["correction"] * np.sum(scores ** 2, axis=0)
    if np.any(variances <= 0) or not np.isfinite(variances).all():
        raise ValueError("Invalid bootstrap variance; no draw has been discarded.")
    values = (h @ weights) / np.sqrt(variances)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite bootstrap t statistic.")
    return values

def draw_weights(g, distribution, stream):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, stream, g]))
    if distribution == "Rademacher" and g <= 13:
        bits = (np.arange(2 ** g, dtype=np.uint64)[None, :] >> np.arange(g, dtype=np.uint64)[:, None]) & 1
        return 2. * bits.astype(float) - 1., True
    support = np.array([-1., 1.]) if distribution == "Rademacher" else np.array(
        [-np.sqrt(1.5), -1., -np.sqrt(.5), np.sqrt(.5), 1., np.sqrt(1.5)])
    return rng.choice(support, size=(g, REPS)), False

def direct_check(x, y, fit, target, weights):
    errors = []
    fast = bootstrap_t(fit, weights)
    for b in range(weights.shape[1]):
        simulated = fit["q"] @ fit["theta0"] + fit["residual0"] * weights[fit["codes"], b]
        model = sm.OLS(simulated, x).fit()
        se = np.sqrt(cov_cluster(model, fit["codes"], use_correction=True)[target, target])
        direct = model.params[target] / se
        np.testing.assert_allclose(fast[b], direct, rtol=2e-8, atol=2e-9)
        errors.append(abs(float(fast[b] - direct)))
    return max(errors)

def synthetic_checks():
    rng = np.random.default_rng(284)
    groups = np.repeat(np.arange(8), [7, 11, 15, 19, 23, 29, 31, 37])
    x = np.column_stack([np.ones(len(groups)), rng.normal(size=(len(groups), 3)), groups == 0])
    y = x @ np.array([1., .3, -.2, .6, .4]) + rng.normal(size=8)[groups] + rng.normal(size=len(groups))
    fit = prepare(x, y, groups, 1)
    max_error = 0.
    for stream, distribution in enumerate(["Rademacher", "Webb"]):
        weights, _ = draw_weights(8, distribution, stream)
        indices = np.linspace(0, weights.shape[1] - 1, 9, dtype=int)
        max_error = max(max_error, direct_check(x, y, fit, 1, weights[:, indices]))
        np.testing.assert_array_equal(weights, draw_weights(8, distribution, stream)[0])
        order = rng.permutation(len(groups))
        permuted = prepare(x[order], y[order], groups[order], 1)
        np.testing.assert_allclose(bootstrap_t(fit, weights), bootstrap_t(permuted, weights), atol=1e-10)
    try:
        prepare(np.column_stack([x, x[:, 1]]), y, groups, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Rank-deficient design was accepted.")
    return {"direct_refits": 18, "max_t_error": max_error,
            "row_order_and_rng_checks": True, "rank_deficiency_rejected": True}
