from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.special import gammaln
from scipy.special import loggamma
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.stats.sandwich_covariance import cov_cluster
from . import sample
from . import accounting

OPTIMIZERS = {
    "bfgs": {"maxiter": 1000, "gtol": 1e-8},
    "lbfgs": {"maxiter": 1000, "pgtol": 1e-8, "factr": 1e4},
}

MAX_MEAN_SCORE = 1e-6

MAX_INFORMATION_SCALED_SCORE = 1e-3

TERM_FIELDS = {"agri": sample.AGRI, "gtos_intensity": sample.LAGGED[0],
               "gtos_diffusion": sample.LAGGED[1], "frontier": sample.FI}

class StableNB2(NegativeBinomial):
    def _ll_nb2(self, params):
        log_size = -params[-1] if self._transparams else -np.log(params[-1])
        size = np.exp(log_size)
        log_mu_over_size = self.exog @ params[:-1] - log_size
        positive = log_mu_over_size.real > 0
        softplus = np.empty_like(log_mu_over_size)
        softplus[positive] = log_mu_over_size[positive] + np.log1p(np.exp(-log_mu_over_size[positive]))
        softplus[~positive] = np.log1p(np.exp(log_mu_over_size[~positive]))
        y = self.endog
        # Same NB2 likelihood, without log(1-prob) rounding to log(0) for tiny fitted means.
        # Complex-valued loggamma/softplus retain the library's complex-step score checks.
        return loggamma(y + size) - loggamma(size) - loggamma(y + 1) - size * softplus + y * (log_mu_over_size - softplus)

def designs(frame: pd.DataFrame) -> dict:
    full = accounting.build_design(frame)
    blocks = full["blocks"]
    core = np.r_[full["w"], blocks["firm_controls"], blocks["province_fe"]]
    indices = {"S1_year_fe": full["w"], "S2_controls": core,
               "S3_plus_gtos": np.r_[core, blocks["gtos_intensity"], blocks["gtos_diffusion"]],
               "S4_plus_frontier": np.r_[core, blocks["gtos_intensity"], blocks["gtos_diffusion"], blocks["frontier"]]}
    output = {}
    for name, selected in indices.items():
        columns = [full["columns"][i] for i in selected]
        transform = np.eye(len(selected))
        for column, scaling in full["scales"].items():
            if column in columns:
                index = columns.index(column)
                transform[index, index] = 1 / scaling["sd_ddof0"]
                transform[0, index] = -scaling["mean"] / scaling["sd_ddof0"]
        x, raw_x = full["x"][:, selected], full["original_x"][:, selected]
        if not np.allclose(raw_x @ transform, x, rtol=1e-12, atol=1e-12):
            raise AssertionError("The numerical rescaling is not an equivalent design.")
        if np.linalg.matrix_rank(x) != x.shape[1]:
            raise ValueError(f"Non-full-rank baseline design: {name}")
        output[name] = {"columns": columns, "x": x, "raw_x": raw_x, "transform": transform}
    return output

def likelihood_and_scores(y: np.ndarray, x: np.ndarray, beta: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if not alpha > 0:
        raise ValueError("NB2 dispersion must be positive.")
    eta = x @ beta
    mu = np.exp(eta)
    size = 1 / alpha
    log_denom = np.log(size + mu)
    likelihood = gammaln(y + size) - gammaln(size) - gammaln(y + 1) + size * (np.log(size) - log_denom) + y * (eta - log_denom)
    beta_score = x * ((y - mu) / (1 + alpha * mu))[:, None]
    alpha_score = -(digamma(y + size) - digamma(size) + np.log(size) - log_denom - (y - mu) / (size + mu)) / alpha**2
    scores = np.column_stack([beta_score, alpha_score])
    if not np.isfinite(likelihood).all() or not np.isfinite(scores).all():
        raise ArithmeticError("The NB2 likelihood or score is nonfinite; do not alter the data.")
    return likelihood, scores

def fit_diagnostics(result, *, joint: bool) -> dict:
    params = np.asarray(result.params)
    model = result.model
    hessian = np.asarray(model.hessian(params), dtype=float)
    information = -(hessian + hessian.T) / 2
    eigenvalues = np.linalg.eigvalsh(information)
    score = np.asarray(model.score(params))
    if joint:
        # statsmodels returns alpha, not log(alpha); assess the optimization score in log-alpha coordinates too.
        score = score.copy()
        score[-1] *= params[-1]
    diagonal = np.diag(information)
    standardized_information = information / np.sqrt(np.outer(diagonal, diagonal)) if (diagonal > 0).all() else np.full_like(information, np.nan)
    scaled_eigen = np.linalg.eigvalsh(standardized_information) if np.isfinite(standardized_information).all() else np.array([np.nan])
    physical_score = np.asarray(model.score(params))
    score_relative_to_information = float(np.max(np.abs(physical_score) / np.sqrt(diagonal))) if (diagonal > 0).all() else np.inf
    mean_score = float(np.max(np.abs(score)) / result.nobs)
    converged = bool(result.mle_retvals.get("converged", False)) if joint else bool(result.converged)
    finite_likelihood = bool(np.isfinite(result.llf))
    valid = bool(converged and finite_likelihood and np.isfinite(params).all() and np.isfinite(information).all()
                 and eigenvalues[0] > 0 and mean_score <= MAX_MEAN_SCORE
                 and score_relative_to_information <= MAX_INFORMATION_SCALED_SCORE)
    return {"optimizer_reports_convergence": converged, "finite_likelihood": finite_likelihood,
            "finite_parameters": bool(np.isfinite(params).all()),
            "mean_optimization_score_max_abs": mean_score, "information_scaled_score_max_abs": score_relative_to_information,
            "observed_information_min_eigenvalue": float(eigenvalues[0]),
            "observed_information_max_eigenvalue": float(eigenvalues[-1]),
            "diagonal_standardized_information_min_eigenvalue": float(scaled_eigen[0]),
            "diagonal_standardized_information_condition_number": float(scaled_eigen[-1] / scaled_eigen[0]),
            "observed_information_rank": int(np.linalg.matrix_rank(information)),
            "parameter_count": len(params), "numerically_acceptable": valid}

def covariance_and_parameters(result, design: dict, frame: pd.DataFrame, *, joint: bool) -> dict:
    k = design["x"].shape[1]
    params = np.asarray(result.params)
    transformation = np.eye(k + int(joint))
    transformation[:k, :k] = design["transform"]
    raw_params = transformation @ params
    covariances = {}
    for label, column in [("firm", sample.FIRM), ("industry", sample.INDUSTRY)]:
        covariance = cov_cluster(result, frame[column].to_numpy(), use_correction=True)
        covariance = transformation @ covariance @ transformation.T
        if not np.isfinite(covariance).all() or np.any(np.diag(covariance) <= 0):
            raise ArithmeticError(f"Unavailable or invalid {label}-cluster covariance.")
        covariances[label] = covariance
    names = [*design["columns"], *(["alpha"] if joint else [])]
    rows = []
    for index, name in enumerate(names):
        row = {"term": name, "source_field": TERM_FIELDS.get(name, name), "coefficient": float(raw_params[index])}
        for label, covariance in covariances.items():
            se = float(np.sqrt(covariance[index, index]))
            low, high = raw_params[index] - norm.ppf(.975) * se, raw_params[index] + norm.ppf(.975) * se
            row.update({f"{label}_cluster_se": se, f"{label}_cluster_p_normal": float(2 * norm.sf(abs(raw_params[index] / se))),
                        f"{label}_cluster_ci95_low": float(low), f"{label}_cluster_ci95_high": float(high)})
            if name == "agri":
                row.update({f"{label}_cluster_mean_ratio_ci95_low": float(np.exp(low)),
                            f"{label}_cluster_mean_ratio_ci95_high": float(np.exp(high))})
            if name == "alpha":
                row.update({f"{label}_cluster_log_delta_alpha_ci95_low": float(np.exp(np.log(raw_params[index]) - norm.ppf(.975) * se / raw_params[index])),
                            f"{label}_cluster_log_delta_alpha_ci95_high": float(np.exp(np.log(raw_params[index]) + norm.ppf(.975) * se / raw_params[index]))})
        if name == "agri":
            row["conditional_expected_count_ratio"] = float(np.exp(raw_params[index]))
        rows.append(row)
    return {"parameters": rows, "covariance_parameter_order": names,
            "firm_cluster_covariance": covariances["firm"].tolist(), "industry_cluster_covariance": covariances["industry"].tolist()}

def fixed_fit(frame: pd.DataFrame, design: dict, alpha: float) -> tuple[object, dict]:
    model = sm.GLM(frame[sample.COUNT].to_numpy(float), design["x"], family=sm.families.NegativeBinomial(alpha=alpha))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(maxiter=100, tol=1e-8)
    diagnostic = fit_diagnostics(result, joint=False)
    if not diagnostic["numerically_acceptable"]:
        raise RuntimeError(f"Fixed-alpha reproduction failed its numerical checks: {diagnostic}")
    output = {"estimator": "NB2 GLM with fixed unconditional-moment alpha", "alpha": alpha,
              "observations": int(result.nobs), "log_likelihood": float(result.llf),
              "iterations": result.fit_history.get("iteration"), "warnings": [str(item.message) for item in caught],
              "diagnostics": diagnostic, **covariance_and_parameters(result, design, frame, joint=False)}
    return result, output

def joint_fit(frame: pd.DataFrame, design: dict, fixed_result, alpha: float) -> dict:
    y = frame[sample.COUNT].to_numpy(float)
    starts = np.r_[fixed_result.params, alpha]
    candidates, objects = [], {}
    native = NegativeBinomial(y, design["x"], loglike_method="nb2", check_rank=False)
    native._transparams = False
    stable_start = StableNB2(y, design["x"], loglike_method="nb2", check_rank=False)
    stable_start._transparams = False
    with np.errstate(divide="ignore", invalid="ignore"):
        native_ll = native.loglikeobs(starts)
    stable_ll = stable_start.loglikeobs(starts)
    ordinary = np.isfinite(native_ll)
    eta = design["x"] @ fixed_result.params
    likelihood_implementation = {"native_nonfinite_rows_at_fixed_start": int((~ordinary).sum()),
                                 "native_nonfinite_rows_have_zero_count": bool((y[~ordinary] == 0).all()),
                                 "fixed_start_log_mean_minimum": float(eta.min()),
                                 "fixed_start_log_mean_maximum": float(eta.max()),
                                 "max_absolute_likelihood_difference_on_native_finite_rows": float(np.max(np.abs(native_ll[ordinary] - stable_ll[ordinary]))),
                                 "stable_start_likelihood_all_finite": bool(np.isfinite(stable_ll).all())}
    for method, settings in OPTIMIZERS.items():
        print(f"  Joint NB2 {method}; N={len(frame)}, beta columns={design['x'].shape[1]}", flush=True)
        caught = []
        try:
            model = StableNB2(y, design["x"], loglike_method="nb2", check_rank=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(start_params=starts, method=method, disp=False, full_output=True, **settings)
            diagnostic = fit_diagnostics(result, joint=True)
            candidates.append({"optimizer": method, "settings": settings, "diagnostics": diagnostic,
                               "log_likelihood": float(result.llf), "scaled_design_parameters_alpha_last": result.params.tolist(),
                               "optimizer_details": {key: result.mle_retvals[key] for key in ["converged", "warnflag", "iterations", "fcalls", "gcalls"] if key in result.mle_retvals},
                               "warnings": [str(item.message) for item in caught]})
            if diagnostic["numerically_acceptable"]:
                objects[method] = result
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            candidates.append({"optimizer": method, "settings": settings, "error_type": type(error).__name__,
                               "error": str(error), "warnings": [str(item.message) for item in caught],
                               "diagnostics": {"numerically_acceptable": False}})
    agreement = {"both_algorithms_numerically_acceptable": len(objects) == 2}
    if len(objects) == 2:
        first, second = objects["bfgs"], objects["lbfgs"]
        difference = np.abs(first.params - second.params)
        agreement.update({"maximum_scaled_parameter_difference": float(difference.max()),
                          "alpha_relative_difference": float(difference[-1] / first.params[-1]),
                          "mean_log_likelihood_difference": float(abs(first.llf - second.llf) / len(frame)),
                          "agricultural_coefficient_difference": float(difference[design["columns"].index("agri")])})
        agreement["passes_locked_tolerances"] = bool(difference[:-1].max() <= 1e-3 and agreement["alpha_relative_difference"] <= 1e-4 and agreement["mean_log_likelihood_difference"] <= 1e-8)
        if not agreement["passes_locked_tolerances"]:
            return {"usable": False, "reason": "Algorithms pass individual checks but disagree; do not choose a favorable result.",
                    "attempts": candidates, "algorithm_agreement": agreement, "likelihood_implementation": likelihood_implementation}
    if not objects:
        return {"usable": False, "reason": "No joint fit passed convergence, score and observed-information checks.",
                "attempts": candidates, "algorithm_agreement": agreement, "likelihood_implementation": likelihood_implementation}
    # Fixed preference, not selection by likelihood, sign or significance.
    method = "bfgs" if "bfgs" in objects else "lbfgs"
    result = objects[method]
    if result.llf + 1e-6 < fixed_result.llf:
        return {"usable": False, "reason": "Joint likelihood is below the nested fixed-alpha likelihood.",
                "attempts": candidates, "algorithm_agreement": agreement, "likelihood_implementation": likelihood_implementation}
    independent_ll, independent_scores = likelihood_and_scores(y, design["x"], result.params[:-1], result.params[-1])
    library_scores = result.model.score_obs(result.params)
    if not np.isclose(independent_ll.sum(), result.llf, rtol=1e-10, atol=1e-7):
        raise AssertionError("Independent NB2 likelihood disagrees with statsmodels.")
    if not np.allclose(independent_scores, library_scores, rtol=1e-7, atol=1e-7):
        raise AssertionError("Independent analytic NB2 scores disagree with library complex-step scores.")
    output = {"usable": True, "estimator": "Joint maximum-likelihood NB2 with log link and model-specific alpha",
              "selected_optimizer": method, "selection_rule": "BFGS if numerically acceptable; otherwise L-BFGS. Both valid algorithms must agree. No selection by outcome sign or significance.",
              "alpha": float(result.params[-1]), "observations": int(result.nobs), "log_likelihood": float(result.llf),
              "log_likelihood_improvement_over_fixed": float(result.llf - fixed_result.llf),
              "attempts": candidates, "algorithm_agreement": agreement, "likelihood_implementation": likelihood_implementation,
              "score_validation_max_absolute_error": float(np.max(np.abs(independent_scores - library_scores))),
              "likelihood_validation_absolute_error": float(abs(independent_ll.sum() - result.llf)),
              "scaled_observed_information_alpha_last": (-result.model.hessian(result.params)).tolist(),
              **covariance_and_parameters(result, design, frame, joint=True)}
    return output

def self_test() -> None:
    rng = np.random.default_rng(20261008)
    n = 1200
    frame = pd.DataFrame({sample.FIRM: np.repeat([f"{i:06d}" for i in range(150)], 8),
                          sample.AGRI: np.repeat(np.arange(150) % 2, 8), "year": np.tile(np.arange(2010, 2018), 150),
                          "province": np.repeat(np.array(["A", "B", "C"])[np.arange(150) % 3], 8),
                          sample.INDUSTRY: np.repeat([f"C{i % 15:02d}" for i in range(150)], 8)})
    for variable in [*sample.FINANCIAL, "list_age", *sample.LAGGED]:
        frame[variable] = rng.normal(size=n)
    frame["soe"] = rng.integers(0, 2, size=n)
    frame[sample.FI] = rng.integers(0, 2, size=n)
    mu = np.exp(.4 - .3 * frame[sample.AGRI] + .2 * frame[sample.LAGGED[0]])
    true_alpha = 1.5
    frame[sample.COUNT] = rng.negative_binomial(1 / true_alpha, (1 / true_alpha) / (1 / true_alpha + mu))
    design = designs(frame)["S4_plus_frontier"]
    alpha = max((frame[sample.COUNT].var() - frame[sample.COUNT].mean()) / frame[sample.COUNT].mean()**2, 1e-6)
    fixed_result, fixed = fixed_fit(frame, design, alpha)
    joint = joint_fit(frame, design, fixed_result, alpha)
    assert joint["usable"] and joint["alpha"] > 0
    assert joint["log_likelihood"] >= fixed["log_likelihood"] - 1e-6
    raw_beta = np.array([row["coefficient"] for row in joint["parameters"][:-1]])
    raw_ll, _ = likelihood_and_scores(frame[sample.COUNT].to_numpy(float), design["raw_x"], raw_beta, joint["alpha"])
    assert np.isclose(raw_ll.sum(), joint["log_likelihood"], atol=1e-7)
    assert len(joint["parameters"]) == len(design["columns"]) + 1
    assert all(row["conditional_expected_count_ratio"] == np.exp(row["coefficient"]) for row in joint["parameters"] if row["term"] == "agri")
    tiny_x = np.ones((3, 1))
    tiny_model = StableNB2(np.array([0., 0., 1.]), tiny_x, loglike_method="nb2")
    tiny_model._transparams = False
    tiny_params = np.array([-75., 9.])
    expected_ll, expected_scores = likelihood_and_scores(tiny_model.endog, tiny_x, tiny_params[:-1], tiny_params[-1])
    np.testing.assert_allclose(tiny_model.loglikeobs(tiny_params), expected_ll, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tiny_model.score_obs(tiny_params), expected_scores, rtol=1e-7, atol=1e-7)
    assert np.isfinite(tiny_model.loglikeobs(tiny_params)).all()
    print("NB2 synthetic checks passed: joint alpha, design rescaling, tiny-mean likelihood/scores, two algorithms and cluster covariance.", flush=True)
