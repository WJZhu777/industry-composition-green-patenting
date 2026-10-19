from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from .structural_utils import CORE_FIRM_CONTROLS
from .structural_utils import CORE_GTOS_VARS
from .structural_utils import PATENT_OUTCOMES
from .structural_utils import clustered_ols

AGRI = "是否涉农"

GTOS_LABELS = {
    "industry_year_nonagri_patent_per_firm_lag1_log1p": "Lagged opportunity intensity",
    "industry_year_has_patent_share_nonagri_lag1": "Opportunity diffusion",
    "frontier_industry_flag_baseline": "Frontier-industry indicator",
}

FOCAL_LABELS = {
    AGRI: "Agricultural firm",
    **GTOS_LABELS,
}

OUTCOME_LABELS = {
    "green_patent_application_count": "Patent count",
    "log1p_green_patent_application_count": "Log(1 + patent count)",
    "has_green_patent_application": "Patent entry indicator",
}

YEAR_ONLY = "C(year)"

BACKGROUND = "C(year) + " + " + ".join(CORE_FIRM_CONTROLS) + " + C(province)"

CONTROL_BASE = AGRI + " + " + BACKGROUND

SPARSE_BASE = AGRI + " + " + YEAR_ONLY

FULL_ACCOUNTING = CONTROL_BASE + " + " + " + ".join(CORE_GTOS_VARS)

def residualize(values: pd.Series | pd.DataFrame, design: pd.DataFrame) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    coefficients = np.linalg.lstsq(design.to_numpy(dtype=float), array, rcond=None)[0]
    return array - design.to_numpy(dtype=float) @ coefficients

def background_design(dataframe: pd.DataFrame, formula: str = BACKGROUND) -> pd.DataFrame:
    return patsy.dmatrix(formula, dataframe, return_type="dataframe")

def correlation_long(dataframe: pd.DataFrame, variables: list[str], label: str) -> pd.DataFrame:
    matrix = dataframe[variables].astype(float).corr()
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(variables):
        for right in variables[left_index + 1 :]:
            rows.append(
                {
                    "Correlation basis": label,
                    "Variable 1": FOCAL_LABELS.get(left, OUTCOME_LABELS.get(left, left)),
                    "Variable 2": FOCAL_LABELS.get(right, OUTCOME_LABELS.get(right, right)),
                    "Pearson correlation": float(matrix.loc[left, right]),
                }
            )
    return pd.DataFrame(rows)

def residualized_dataframe(
    dataframe: pd.DataFrame,
    variables: list[str],
    formula: str = BACKGROUND,
) -> pd.DataFrame:
    design = background_design(dataframe, formula)
    residuals = {
        variable: residualize(dataframe[variable], design)
        for variable in variables
    }
    return pd.DataFrame(residuals, index=dataframe.index)

def vif_from_matrix(dataframe: pd.DataFrame, basis: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable in dataframe.columns:
        predictors = dataframe.drop(columns=[variable]).astype(float)
        predictors = sm.add_constant(predictors, has_constant="add")
        result = sm.OLS(dataframe[variable].astype(float), predictors).fit()
        r_squared = float(result.rsquared)
        vif = np.inf if r_squared >= 1 else 1.0 / (1.0 - r_squared)
        rows.append(
            {
                "Basis": basis,
                "Variable": FOCAL_LABELS.get(variable, variable),
                "Auxiliary R-squared": r_squared,
                "Variance inflation factor": vif,
                "Tolerance": 0.0 if not np.isfinite(vif) else 1.0 / vif,
            }
        )
    return pd.DataFrame(rows)

def condition_index_table(dataframe: pd.DataFrame, basis: str) -> pd.DataFrame:
    standardized = (dataframe - dataframe.mean()) / dataframe.std(ddof=0)
    standardized = standardized.loc[:, standardized.std(ddof=0).gt(0)]
    scaled = standardized.to_numpy(dtype=float)
    scaled = scaled / np.sqrt((scaled**2).sum(axis=0))
    cross_product = scaled.T @ scaled
    eigenvalues, eigenvectors = np.linalg.eigh(cross_product)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    max_eigenvalue = float(eigenvalues.max())

    inverse_eigen = np.divide(
        eigenvectors**2,
        eigenvalues,
        out=np.full_like(eigenvectors, np.nan),
        where=eigenvalues > 1e-12,
    )
    variance_proportions = inverse_eigen / np.nansum(inverse_eigen, axis=1, keepdims=True)

    rows: list[dict[str, object]] = []
    for dimension, eigenvalue in enumerate(eigenvalues, start=1):
        condition_index = (
            np.sqrt(max_eigenvalue / eigenvalue)
            if eigenvalue > 1e-12
            else np.inf
        )
        high_variables = [
            FOCAL_LABELS.get(variable, variable)
            for variable, proportion in zip(
                standardized.columns,
                variance_proportions[:, dimension - 1],
            )
            if np.isfinite(proportion) and proportion >= 0.50
        ]
        row: dict[str, object] = {
            "Basis": basis,
            "Dimension": dimension,
            "Eigenvalue": float(eigenvalue),
            "Condition index": float(condition_index),
            "Variables with variance proportion >= 0.50": "; ".join(high_variables),
        }
        for variable, proportion in zip(
            standardized.columns,
            variance_proportions[:, dimension - 1],
        ):
            row[f"Variance proportion: {FOCAL_LABELS.get(variable, variable)}"] = float(
                proportion
            )
        rows.append(row)
    return pd.DataFrame(rows)

def fit_specification(
    dataframe: pd.DataFrame,
    outcome: str,
    specification: str,
    rhs: str,
    *,
    model_cache: dict | None = None,
) -> dict[str, object]:
    # Reuse fits only within one diagnostic on one unchanged sample.
    models = {} if model_cache is None else model_cache
    cache_key = (outcome, rhs)
    if cache_key not in models:
        models[cache_key] = clustered_ols(f"{outcome} ~ {rhs}", dataframe)
    result = models[cache_key]
    row: dict[str, object] = {
        "Outcome": OUTCOME_LABELS[outcome],
        "Specification": specification,
        "Agricultural coefficient": float(result.params[AGRI]),
        "Agricultural standard error": float(result.bse[AGRI]),
        "Agricultural p-value": float(result.pvalues[AGRI]),
        "Observations": int(result.nobs),
        "R-squared": float(result.rsquared),
    }
    for variable in CORE_GTOS_VARS:
        row[f"{GTOS_LABELS[variable]} coefficient"] = (
            float(result.params[variable]) if variable in result.params else np.nan
        )
        row[f"{GTOS_LABELS[variable]} standard error"] = (
            float(result.bse[variable]) if variable in result.params else np.nan
        )
        row[f"{GTOS_LABELS[variable]} p-value"] = (
            float(result.pvalues[variable]) if variable in result.params else np.nan
        )
    return row

def build_nested_paths(dataframe: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        ("Sparse restricted: agricultural firm + year effects", SPARSE_BASE),
        ("Control-adjusted: + firm controls and province effects", CONTROL_BASE),
        (
            "Control-adjusted + lagged opportunity intensity",
            CONTROL_BASE + " + " + CORE_GTOS_VARS[0],
        ),
        (
            "Control-adjusted + opportunity diffusion",
            CONTROL_BASE + " + " + CORE_GTOS_VARS[1],
        ),
        (
            "Control-adjusted + frontier-industry indicator",
            CONTROL_BASE + " + " + CORE_GTOS_VARS[2],
        ),
        (
            "Control-adjusted + intensity + diffusion",
            CONTROL_BASE + " + " + " + ".join(CORE_GTOS_VARS[:2]),
        ),
        (
            "Control-adjusted + intensity + frontier",
            CONTROL_BASE + " + " + CORE_GTOS_VARS[0] + " + " + CORE_GTOS_VARS[2],
        ),
        (
            "Control-adjusted + diffusion + frontier",
            CONTROL_BASE + " + " + CORE_GTOS_VARS[1] + " + " + CORE_GTOS_VARS[2],
        ),
        ("Full accounting: controls + all GTOS variables", FULL_ACCOUNTING),
    ]
    rows = [
        fit_specification(dataframe, outcome, label, rhs)
        for outcome in PATENT_OUTCOMES
        for label, rhs in specifications
    ]
    return pd.DataFrame(rows)

def build_order_sensitivity(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    models = {}
    for outcome in PATENT_OUTCOMES:
        for order_id, order in enumerate(itertools.permutations(CORE_GTOS_VARS), start=1):
            included: list[str] = []
            base = fit_specification(
                dataframe,
                outcome,
                "Step 0: control-adjusted baseline",
                CONTROL_BASE,
                model_cache=models,
            )
            rows.append(
                {
                    "Outcome": OUTCOME_LABELS[outcome],
                    "Order": order_id,
                    "Order labels": " -> ".join(GTOS_LABELS[item] for item in order),
                    "Step": 0,
                    "Variable added": "None",
                    "Agricultural coefficient": base["Agricultural coefficient"],
                    "Agricultural standard error": base[
                        "Agricultural standard error"
                    ],
                }
            )
            for step, variable in enumerate(order, start=1):
                included.append(variable)
                result = fit_specification(
                    dataframe,
                    outcome,
                    f"Step {step}",
                    CONTROL_BASE + " + " + " + ".join(included),
                    model_cache=models,
                )
                rows.append(
                    {
                        "Outcome": OUTCOME_LABELS[outcome],
                        "Order": order_id,
                        "Order labels": " -> ".join(
                            GTOS_LABELS[item] for item in order
                        ),
                        "Step": step,
                        "Variable added": GTOS_LABELS[variable],
                        "Agricultural coefficient": result[
                            "Agricultural coefficient"
                        ],
                        "Agricultural standard error": result[
                            "Agricultural standard error"
                        ],
                    }
                )
    return pd.DataFrame(rows)

def build_leave_one_out(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome in PATENT_OUTCOMES:
        specs = [("All GTOS variables", CORE_GTOS_VARS)]
        specs.extend(
            (
                f"Omit {GTOS_LABELS[omitted]}",
                [variable for variable in CORE_GTOS_VARS if variable != omitted],
            )
            for omitted in CORE_GTOS_VARS
        )
        for label, variables in specs:
            result = fit_specification(
                dataframe,
                outcome,
                label,
                CONTROL_BASE + " + " + " + ".join(variables),
            )
            rows.append(result)
    return pd.DataFrame(rows)

def build_fwl_diagnostics(dataframe: pd.DataFrame) -> pd.DataFrame:
    model_covariates = [
        ("Sparse restricted", YEAR_ONLY),
        ("Control-adjusted", BACKGROUND),
        ("Full accounting", BACKGROUND + " + " + " + ".join(CORE_GTOS_VARS)),
    ]
    rows: list[dict[str, object]] = []
    for outcome in PATENT_OUTCOMES:
        for model_label, covariates in model_covariates:
            design = background_design(dataframe, covariates)
            outcome_residual = residualize(dataframe[outcome], design)
            agri_residual = residualize(dataframe[AGRI], design)
            coefficient = float(
                np.dot(agri_residual, outcome_residual)
                / np.dot(agri_residual, agri_residual)
            )
            rhs = f"{AGRI} + {covariates}"
            direct = clustered_ols(f"{outcome} ~ {rhs}", dataframe)
            rows.append(
                {
                    "Outcome": OUTCOME_LABELS[outcome],
                    "Model": model_label,
                    "FWL agricultural coefficient": coefficient,
                    "Direct agricultural coefficient": float(direct.params[AGRI]),
                    "Clustered standard error": float(direct.bse[AGRI]),
                    "Residualized agricultural standard deviation": float(
                        np.std(agri_residual, ddof=1)
                    ),
                    "Residualized outcome standard deviation": float(
                        np.std(outcome_residual, ddof=1)
                    ),
                    "Residual correlation": float(
                        np.corrcoef(agri_residual, outcome_residual)[0, 1]
                    ),
                    "Coefficient verification difference": coefficient
                    - float(direct.params[AGRI]),
                }
            )
    return pd.DataFrame(rows)

def build_conditional_exposure_associations(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable in CORE_GTOS_VARS:
        result = clustered_ols(f"{variable} ~ {CONTROL_BASE}", dataframe)
        rows.append(
            {
                "GTOS variable": GTOS_LABELS[variable],
                "Conditional agricultural coefficient": float(result.params[AGRI]),
                "Clustered standard error": float(result.bse[AGRI]),
                "P-value": float(result.pvalues[AGRI]),
                "Observations": int(result.nobs),
                "R-squared": float(result.rsquared),
            }
        )
    return pd.DataFrame(rows)

def build_gtos_block_joint_tests(dataframe: pd.DataFrame) -> pd.DataFrame:
    restriction = ", ".join(f"{variable} = 0" for variable in CORE_GTOS_VARS)
    rows: list[dict[str, object]] = []
    for outcome in PATENT_OUTCOMES:
        result = clustered_ols(f"{outcome} ~ {FULL_ACCOUNTING}", dataframe)
        test = result.wald_test(restriction, scalar=True)
        base = clustered_ols(f"{outcome} ~ {CONTROL_BASE}", dataframe)
        rows.append(
            {
                "Outcome": OUTCOME_LABELS[outcome],
                "Joint Wald statistic": float(np.asarray(test.statistic).squeeze()),
                "Joint p-value": float(np.asarray(test.pvalue).squeeze()),
                "Control-adjusted R-squared": float(base.rsquared),
                "Full-accounting R-squared": float(result.rsquared),
                "Incremental R-squared": float(result.rsquared - base.rsquared),
                "Observations": int(result.nobs),
            }
        )
    return pd.DataFrame(rows)
