from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUTCOME_SPECS = [
    ("green_patent_application_count", "patent_count", "OLS-level"),
    ("log1p_green_patent_application_count", "log1p_patent_count", "OLS-log"),
    ("has_green_patent_application", "has_patent", "LPM"),
    ("is_penalized_announcement_year", "penalized_dummy", "LPM"),
    ("penalty_count_announcement_year", "penalty_count", "OLS-level"),
]

RAW_CONTROLS = ["ln_assets", "roa", "leverage", "ocf_ratio"]

WINSOR_CONTROLS = ["ln_assets_w", "roa_w", "leverage_w", "ocf_ratio_w"]

def fit_clustered_ols(dataframe: pd.DataFrame, formula: str):
    model = smf.ols(formula, data=dataframe)
    return model.fit(cov_type="cluster", cov_kwds={"groups": dataframe["证券代码"]})

def model_formulas(outcome: str) -> dict[str, tuple[str, list[str]]]:
    base_fe = "是否涉农 + C(year) + C(industry_code_std)"
    return {
        "D2_plus_soe": (
            f"{outcome} ~ {base_fe} + ln_assets + roa + leverage + ocf_ratio + soe",
            [outcome, "是否涉农", "证券代码", "year", "industry_code_std", "ln_assets", "roa", "leverage", "ocf_ratio", "soe"],
        ),
        "D3_plus_province": (
            f"{outcome} ~ {base_fe} + ln_assets + roa + leverage + ocf_ratio + C(province)",
            [outcome, "是否涉农", "证券代码", "year", "industry_code_std", "ln_assets", "roa", "leverage", "ocf_ratio", "province"],
        ),
        "D4_full_static": (
            f"{outcome} ~ {base_fe} + ln_assets + roa + leverage + ocf_ratio + soe + C(province) + list_age",
            [outcome, "是否涉农", "证券代码", "year", "industry_code_std", "ln_assets", "roa", "leverage", "ocf_ratio", "soe", "province", "list_age"],
        ),
        "D4_winsorized": (
            f"{outcome} ~ {base_fe} + ln_assets_w + roa_w + leverage_w + ocf_ratio_w + soe + C(province) + list_age",
            [outcome, "是否涉农", "证券代码", "year", "industry_code_std", "ln_assets_w", "roa_w", "leverage_w", "ocf_ratio_w", "soe", "province", "list_age"],
        ),
    }

def run_new_models(panel_v3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    readiness_rows: list[dict[str, object]] = []

    spec_requirements = {
        "D2_plus_soe": RAW_CONTROLS + ["soe"],
        "D3_plus_province": RAW_CONTROLS + ["province"],
        "D4_full_static": RAW_CONTROLS + ["soe", "province", "list_age"],
        "D4_winsorized": WINSOR_CONTROLS + ["soe", "province", "list_age"],
    }
    for model_name, required_controls in spec_requirements.items():
        mask = panel_v3[required_controls].notna().all(axis=1)
        readiness_rows.append(
            {
                "model_id": model_name,
                "rows_retained": int(mask.sum()),
                "share_retained": mask.mean(),
                "agri_share_retained": mask.loc[panel_v3["是否涉农"] == 1].mean(),
                "nonagri_share_retained": mask.loc[panel_v3["是否涉农"] == 0].mean(),
            }
        )

    for outcome, outcome_label, estimator in OUTCOME_SPECS:
        for model_id, (formula, required_cols) in model_formulas(outcome).items():
            work_df = panel_v3.dropna(subset=required_cols).copy()
            if outcome in {"is_penalized_announcement_year", "penalty_count_announcement_year"}:
                # The penalty export has complete calendar years only in 2009-2023.
                work_df = work_df.loc[work_df["year"].between(2009, 2023)].copy()
            result = fit_clustered_ols(work_df, formula)
            ci_low, ci_high = result.conf_int().loc["是否涉农"].tolist()
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": outcome_label,
                    "estimator": estimator,
                    "model_id": model_id,
                    "formula": formula,
                    "coef_agri": float(result.params["是否涉农"]),
                    "se_agri": float(result.bse["是否涉农"]),
                    "pvalue_agri": float(result.pvalues["是否涉农"]),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "nobs": int(result.nobs),
                    "r_squared": float(getattr(result, "rsquared", np.nan)),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(readiness_rows)
