from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from .structural_utils import CORE_FIRM_CONTROLS
from .structural_utils import PATENT_OUTCOMES
from .structural_utils import clustered_ols
from .structural_utils import common_sample

BASE_WITH_CONTROLS = "是否涉农 + C(year) + " + " + ".join(CORE_FIRM_CONTROLS) + " + C(province)"

GTOS_LAG1_TERMS = "industry_year_nonagri_patent_per_firm_lag1_log1p + industry_year_has_patent_share_nonagri_lag1"

GTOS_LAG1_FRONTIER = GTOS_LAG1_TERMS + " + frontier_industry_flag_baseline"

def post_2015_indicator(panel: pd.DataFrame) -> pd.Series:
    indicator = pd.to_numeric(panel["year"], errors="raise").ge(2015).astype(int)
    if "post_2015" in panel and not panel["post_2015"].eq(indicator).fillna(False).all():
        raise ValueError("Stored post_2015 must equal (year >= 2015) on every row.")
    return indicator

def extract_rows(result, outcome_label: str, model_id: str, terms: list[str]) -> list[dict[str, object]]:
    rows = []
    for term in terms:
        if term not in result.params.index:
            continue
        ci_low, ci_high = result.conf_int().loc[term].tolist()
        coef = pd.to_numeric(pd.Series([result.params[term]]), errors="coerce").iloc[0]
        se = pd.to_numeric(pd.Series([result.bse[term]]), errors="coerce").iloc[0]
        pvalue = pd.to_numeric(pd.Series([result.pvalues[term]]), errors="coerce").iloc[0]
        rows.append(
            {
                "outcome": outcome_label,
                "model_id": model_id,
                "term": term,
                "coef": float(coef) if pd.notna(coef) else np.nan,
                "se": float(se) if pd.notna(se) else np.nan,
                "pvalue": float(pvalue) if pd.notna(pvalue) else np.nan,
                "ci_low": float(ci_low) if pd.notna(ci_low) else np.nan,
                "ci_high": float(ci_high) if pd.notna(ci_high) else np.nan,
                "nobs": int(result.nobs),
                "pseudo_r2_or_r2": float(getattr(result, "prsquared", np.nan) if hasattr(result, "prsquared") else getattr(result, "rsquared", np.nan)),
            }
        )
    return rows

def fit_poisson_clustered(formula: str, dataframe: pd.DataFrame):
    model = smf.poisson(formula=formula, data=dataframe)
    return model.fit(disp=False, maxiter=200, cov_type="cluster", cov_kwds={"groups": dataframe["证券代码"]})

def fit_nb_glm(formula: str, dataframe: pd.DataFrame):
    y_name = formula.split("~", 1)[0].strip()
    mean_y = dataframe[y_name].mean()
    var_y = dataframe[y_name].var()
    alpha = max((var_y - mean_y) / (mean_y**2), 1e-6) if mean_y > 0 else 1.0
    model = smf.glm(
        formula=formula,
        data=dataframe,
        family=sm.families.NegativeBinomial(alpha=alpha),
    )
    result = model.fit(cov_type="cluster", cov_kwds={"groups": dataframe["证券代码"]})
    if not result.converged:
        raise RuntimeError("Fixed-dispersion NB2 did not converge.")
    if not (
        np.isfinite(result.params).all()
        and np.isfinite(result.bse).all()
        and np.isfinite(result.pvalues).all()
        and np.isfinite(result.llf)
    ):
        raise ArithmeticError("Fixed-dispersion NB2 returned nonfinite estimates or inference.")
    return result, alpha

def run_ppml(panel: pd.DataFrame) -> pd.DataFrame:
    df = common_sample(panel, ["green_patent_application_count"])
    formulas = {
        "S1_year_fe": "green_patent_application_count ~ 是否涉农 + C(year)",
        "S2_controls": f"green_patent_application_count ~ {BASE_WITH_CONTROLS}",
        "S3_plus_gtos": f"green_patent_application_count ~ {BASE_WITH_CONTROLS} + {GTOS_LAG1_TERMS}",
        "S4_plus_frontier": f"green_patent_application_count ~ {BASE_WITH_CONTROLS} + {GTOS_LAG1_FRONTIER}",
    }
    terms = ["是否涉农", "industry_year_nonagri_patent_per_firm_lag1_log1p",
             "industry_year_has_patent_share_nonagri_lag1", "frontier_industry_flag_baseline"]
    rows = []
    for model_id, formula in formulas.items():
        result = fit_poisson_clustered(formula, df)
        optimizer_converged = bool(result.mle_retvals.get("converged", False))
        finite_fit = all(np.isfinite(np.asarray(values)).all() for values in
                         (result.params, result.bse, result.pvalues, result.conf_int(), result.llf))
        usable = optimizer_converged and finite_fit
        if usable:
            current_rows = extract_rows(result, "patent_count", model_id, terms)
            null_ll = getattr(result, "llnull", np.nan)
            pseudo_r2 = 1 - result.llf / null_ll if np.isfinite(null_ll) and null_ll != 0 else np.nan
        else:
            current_rows = [
                {"outcome": "patent_count", "model_id": model_id, "term": term,
                 "coef": np.nan, "se": np.nan, "pvalue": np.nan,
                 "ci_low": np.nan, "ci_high": np.nan, "nobs": int(result.nobs)}
                for term in terms if term in result.params.index
            ]
            pseudo_r2 = np.nan
        for row in current_rows:
            row["pseudo_r2_or_r2"] = pseudo_r2
            row["estimator"] = "Poisson QMLE"
            row["optimizer_converged"] = optimizer_converged
            row["converged"] = usable
            row["fit_usable"] = usable
            row["term_is_finite"] = usable
            row["numeric_note"] = ("numerically_unstable" if not finite_fit else
                                   "optimizer_not_converged" if not optimizer_converged else "")
        rows.extend(current_rows)
    return pd.DataFrame(rows)

def run_negative_binomial(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = common_sample(panel, ["green_patent_application_count"])
    formulas = {
        "S1_year_fe": "green_patent_application_count ~ 是否涉农 + C(year)",
        "S2_controls": f"green_patent_application_count ~ {BASE_WITH_CONTROLS}",
        "S3_plus_gtos": f"green_patent_application_count ~ {BASE_WITH_CONTROLS} + {GTOS_LAG1_TERMS}",
        "S4_plus_frontier": f"green_patent_application_count ~ {BASE_WITH_CONTROLS} + {GTOS_LAG1_FRONTIER}",
    }
    rows = []
    y_mean = float(df["green_patent_application_count"].mean())
    y_var = float(df["green_patent_application_count"].var())
    alpha_common = max((y_var - y_mean) / (y_mean**2), 1e-6) if y_mean > 0 else 1.0
    meta_rows = [
        {
            "metric": "mean_patent_count",
            "value": y_mean,
        },
        {
            "metric": "variance_patent_count",
            "value": y_var,
        },
        {
            "metric": "variance_to_mean_ratio",
            "value": float(y_var / y_mean) if y_mean > 0 else np.nan,
        },
        {
            "metric": "zero_share",
            "value": float((df["green_patent_application_count"] == 0).mean()),
        },
        {
            "metric": "method_of_moments_alpha",
            "value": alpha_common,
        },
    ]
    for model_id, formula in formulas.items():
        result, alpha = fit_nb_glm(formula, df)
        current_rows = extract_rows(
            result,
            "patent_count",
            model_id,
            ["是否涉农", "industry_year_nonagri_patent_per_firm_lag1_log1p", "industry_year_has_patent_share_nonagri_lag1", "frontier_industry_flag_baseline"],
        )
        for row in current_rows:
            row["estimator"] = "GLM Negative Binomial"
            row["alpha"] = alpha
            row["converged"] = bool(result.converged)
            row["term_is_finite"] = bool(np.isfinite(row["coef"])) and bool(np.isfinite(row["se"]))
            row["numeric_note"] = "numerically_unstable" if not row["term_is_finite"] else ""
        rows.extend(current_rows)
        meta_rows.append(
            {
                "metric": f"{model_id}_observations",
                "value": int(result.nobs),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(meta_rows)

def run_two_part(panel: pd.DataFrame) -> pd.DataFrame:
    df = common_sample(panel.assign(post_2015=post_2015_indicator(panel)), PATENT_OUTCOMES)
    df["log_positive_patent_count"] = np.nan
    positive_mask = df["green_patent_application_count"] > 0
    df.loc[positive_mask, "log_positive_patent_count"] = np.log(df.loc[positive_mask, "green_patent_application_count"])

    model_specs = {
        "base": f"~ {BASE_WITH_CONTROLS}",
        "plus_gtos": f"~ {BASE_WITH_CONTROLS} + {GTOS_LAG1_FRONTIER}",
    }
    rows = []
    for model_id, rhs in model_specs.items():
        stage1 = clustered_ols(f"has_green_patent_application {rhs}", df)
        positive_df = df.loc[df["green_patent_application_count"] > 0].copy()
        stage2 = clustered_ols(f"log_positive_patent_count {rhs}", positive_df)
        for stage_name, result, outcome_name in [
            ("entry_margin", stage1, "has_patent"),
            ("intensive_margin_positive_only", stage2, "log_positive_count_positive_only"),
        ]:
            rows.extend(
                [
                    {
                        "margin": stage_name,
                        "model_id": model_id,
                        "outcome": outcome_name,
                        "term": "是否涉农",
                        "coef": float(result.params["是否涉农"]),
                        "se": float(result.bse["是否涉农"]),
                        "pvalue": float(result.pvalues["是否涉农"]),
                        "nobs": int(result.nobs),
                        "r2": float(result.rsquared),
                    }
                ]
            )

    interaction_rows = []
    for stage_name, formula_outcome, stage_df in [
        ("entry_margin", "has_green_patent_application", df),
        ("intensive_margin_positive_only", "log_positive_patent_count", df.loc[df["green_patent_application_count"] > 0].copy()),
    ]:
        base_inter = clustered_ols(f"{formula_outcome} ~ 是否涉农 + 是否涉农:post_2015 + {BASE_WITH_CONTROLS}", stage_df)
        gtos_inter = clustered_ols(f"{formula_outcome} ~ 是否涉农 + 是否涉农:post_2015 + {BASE_WITH_CONTROLS} + {GTOS_LAG1_FRONTIER}", stage_df)
        for model_id, result in [("interaction_base", base_inter), ("interaction_plus_gtos", gtos_inter)]:
            interaction_rows.append(
                {
                    "margin": stage_name,
                    "model_id": model_id,
                    "term": "是否涉农:post_2015",
                    "coef": float(result.params["是否涉农:post_2015"]),
                    "se": float(result.bse["是否涉农:post_2015"]),
                    "pvalue": float(result.pvalues["是否涉农:post_2015"]),
                    "nobs": int(result.nobs),
                    "r2": float(result.rsquared),
                }
            )

    return pd.concat([pd.DataFrame(rows), pd.DataFrame(interaction_rows)], ignore_index=True)
