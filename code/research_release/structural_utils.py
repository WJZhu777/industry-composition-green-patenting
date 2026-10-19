from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA

OUTCOME_MAP = {
    "green_patent_application_count": "patent_count",
    "log1p_green_patent_application_count": "log1p_patent_count",
    "has_green_patent_application": "has_patent",
    "is_penalized_announcement_year": "penalized_dummy",
}

PATENT_OUTCOMES = [
    "green_patent_application_count",
    "log1p_green_patent_application_count",
    "has_green_patent_application",
]

CORE_FIRM_CONTROLS = ["ln_assets", "roa", "leverage", "ocf_ratio", "soe", "list_age"]

CORE_GTOS_VARS = [
    "industry_year_nonagri_patent_per_firm_lag1_log1p",
    "industry_year_has_patent_share_nonagri_lag1",
    "frontier_industry_flag_baseline",
]

CORE_SAMPLE_VARS = [
    "是否涉农",
    "year",
    "province",
    "industry_code_std",
    "industry_name_std",
    "证券代码",
    *CORE_FIRM_CONTROLS,
    *PATENT_OUTCOMES,
    "is_penalized_announcement_year",
    *CORE_GTOS_VARS,
]

BASE_RESTRICTED = "是否涉农 + C(year)"

BASE_FULL = "是否涉农 + C(year) + " + " + ".join(CORE_FIRM_CONTROLS) + " + C(province)"

GTOS_FULL = BASE_FULL + " + industry_year_nonagri_patent_per_firm_lag1_log1p + industry_year_has_patent_share_nonagri_lag1 + frontier_industry_flag_baseline"

def build_enhanced_panel(panel: pd.DataFrame) -> pd.DataFrame:
    dataframe = panel.copy()
    dataframe = dataframe.sort_values(["证券代码", "year"]).reset_index(drop=True)
    dataframe["lag_patent_count"] = dataframe.groupby("证券代码")["green_patent_application_count"].shift(1)
    dataframe["lag_has_patent"] = dataframe.groupby("证券代码")["has_green_patent_application"].shift(1)
    dataframe["lag_industry_code_std"] = dataframe.groupby("证券代码")["industry_code_std"].shift(1)

    industry_year = (
        dataframe[
            [
                "industry_code_std",
                "year",
                "industry_year_patent_count_nonagri",
                "industry_year_patent_firms_nonagri",
                "firm_count_nonagri",
                "industry_year_nonagri_patent_per_firm_lag1_log1p",
                "industry_year_has_patent_share_nonagri_lag1",
            ]
        ]
        .drop_duplicates(["industry_code_std", "year"])
        .copy()
    )

    current_structural = (
        dataframe[
            [
                "industry_code_std",
                "year",
                "industry_year_patent_count_nonagri",
                "industry_year_patent_firms_nonagri",
                "firm_count_nonagri",
            ]
        ]
        .drop_duplicates(["industry_code_std", "year"])
        .copy()
    )
    current_structural["current_nonagri_patent_per_firm"] = np.where(
        current_structural["firm_count_nonagri"] > 0,
        current_structural["industry_year_patent_count_nonagri"] / current_structural["firm_count_nonagri"],
        np.nan,
    )
    current_structural["current_nonagri_patent_per_firm_log1p"] = np.log1p(current_structural["current_nonagri_patent_per_firm"])
    current_structural["current_nonagri_patent_diffusion"] = np.where(
        current_structural["firm_count_nonagri"] > 0,
        current_structural["industry_year_patent_firms_nonagri"] / current_structural["firm_count_nonagri"],
        np.nan,
    )

    lag_structural = current_structural.copy()
    lag_structural["year"] = lag_structural["year"] + 1
    lag_structural = lag_structural.rename(
        columns={
            "industry_year_patent_count_nonagri": "nonagri_patent_count_lag",
            "industry_year_patent_firms_nonagri": "nonagri_patent_firms_lag",
            "firm_count_nonagri": "nonagri_firm_count_lag",
            "current_nonagri_patent_per_firm": "gtos_intensity_lag1_raw",
            "current_nonagri_patent_per_firm_log1p": "gtos_intensity_lag1_log1p_raw",
            "current_nonagri_patent_diffusion": "gtos_diffusion_lag1_raw",
        }
    )
    dataframe = dataframe.merge(lag_structural, on=["industry_code_std", "year"], how="left")

    own_exclusion_flag = (
        (dataframe["是否涉农"] == 0)
        & (dataframe["lag_industry_code_std"] == dataframe["industry_code_std"])
        & dataframe["lag_patent_count"].notna()
    )
    own_exclusion_haspat = (
        (dataframe["是否涉农"] == 0)
        & (dataframe["lag_industry_code_std"] == dataframe["industry_code_std"])
        & dataframe["lag_has_patent"].notna()
    )
    dataframe["loo_nonagri_denom"] = dataframe["nonagri_firm_count_lag"] - own_exclusion_flag.astype(float)
    dataframe["loo_nonagri_patent_num"] = dataframe["nonagri_patent_count_lag"] - np.where(
        own_exclusion_flag, dataframe["lag_patent_count"], 0.0
    )
    dataframe["loo_nonagri_patent_firms_num"] = dataframe["nonagri_patent_firms_lag"] - np.where(
        own_exclusion_haspat, dataframe["lag_has_patent"], 0.0
    )
    intensity_loo = np.where(
        dataframe["loo_nonagri_denom"] > 0,
        dataframe["loo_nonagri_patent_num"] / dataframe["loo_nonagri_denom"],
        np.nan,
    )
    diffusion_loo = np.where(
        dataframe["loo_nonagri_denom"] > 0,
        dataframe["loo_nonagri_patent_firms_num"] / dataframe["loo_nonagri_denom"],
        np.nan,
    )
    dataframe["gtos_intensity_lag1_loo"] = np.log1p(intensity_loo)
    dataframe["gtos_diffusion_lag1_loo"] = diffusion_loo

    roll_source = current_structural[["industry_code_std", "year", "current_nonagri_patent_per_firm_log1p", "current_nonagri_patent_diffusion"]].copy()
    roll_source = roll_source.sort_values(["industry_code_std", "year"]).reset_index(drop=True)
    roll_source["gtos_intensity_roll3"] = (
        roll_source.groupby("industry_code_std")["current_nonagri_patent_per_firm_log1p"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    )
    roll_source["gtos_diffusion_roll3"] = (
        roll_source.groupby("industry_code_std")["current_nonagri_patent_diffusion"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    )
    dataframe = dataframe.merge(
        roll_source[["industry_code_std", "year", "gtos_intensity_roll3", "gtos_diffusion_roll3"]],
        on=["industry_code_std", "year"],
        how="left",
    )

    composite_columns = [
        "industry_year_nonagri_patent_per_firm_lag1_log1p",
        "industry_year_has_patent_share_nonagri_lag1",
        "frontier_industry_flag_baseline",
        "gtos_intensity_roll3",
        "gtos_diffusion_roll3",
    ]
    composite_ready = dataframe[composite_columns].copy()
    z_components = pd.DataFrame(index=dataframe.index)
    for column in composite_columns:
        series = pd.to_numeric(composite_ready[column], errors="coerce")
        mean = series.mean()
        std = series.std(ddof=0)
        z_components[column] = (series - mean) / std if std and not np.isnan(std) else np.nan
    dataframe["gtos_index_zmean"] = z_components.mean(axis=1, skipna=False)

    pca_ready = z_components.dropna().copy()
    if not pca_ready.empty:
        pca = PCA(n_components=1)
        component = pca.fit_transform(pca_ready.values)[:, 0]
        if np.corrcoef(component, pca_ready["industry_year_nonagri_patent_per_firm_lag1_log1p"].values)[0, 1] < 0:
            component = -component
        dataframe["gtos_index_pca1"] = np.nan
        dataframe.loc[pca_ready.index, "gtos_index_pca1"] = component
    else:
        dataframe["gtos_index_pca1"] = np.nan

    unique_cells = (
        dataframe[
            [
                "industry_code_std",
                "year",
                "industry_year_nonagri_patent_per_firm_lag1_log1p",
                "frontier_industry_flag_baseline",
                "gtos_index_zmean",
            ]
        ]
        .drop_duplicates(["industry_code_std", "year"])
        .dropna(subset=["gtos_index_zmean"])
        .copy()
    )
    unique_cells["gtos_regime"] = pd.qcut(
        unique_cells["gtos_index_zmean"],
        q=3,
        labels=["low_opportunity", "middle_opportunity", "high_frontier_opportunity"],
        duplicates="drop",
    )
    unique_cells.loc[unique_cells["frontier_industry_flag_baseline"] == 1, "gtos_regime"] = "high_frontier_opportunity"
    dataframe = dataframe.merge(
        unique_cells[["industry_code_std", "year", "gtos_regime"]],
        on=["industry_code_std", "year"],
        how="left",
    )

    return dataframe

def common_sample(panel: pd.DataFrame, outcomes: list[str], extra_vars: list[str] | None = None) -> pd.DataFrame:
    required = CORE_SAMPLE_VARS.copy()
    for outcome in outcomes:
        if outcome not in required:
            required.append(outcome)
    if extra_vars:
        for variable in extra_vars:
            if variable not in required:
                required.append(variable)
    dataframe = panel.dropna(subset=required).copy()
    dataframe["year"] = dataframe["year"].astype(int)
    for column in [
        "是否涉农",
        *CORE_FIRM_CONTROLS,
        *PATENT_OUTCOMES,
        "is_penalized_announcement_year",
        *CORE_GTOS_VARS,
        "frontier_industry_flag_baseline",
        "gtos_intensity_lag1_loo",
        "gtos_diffusion_lag1_loo",
        "gtos_intensity_roll3",
        "gtos_diffusion_roll3",
        "gtos_index_zmean",
        "gtos_index_pca1",
    ]:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe

def clustered_ols(formula: str, dataframe: pd.DataFrame):
    return smf.ols(formula, data=dataframe).fit(
        cov_type="cluster",
        cov_kwds={"groups": dataframe["证券代码"]},
    )

def stars(pvalue: float) -> str:
    if pd.isna(pvalue):
        return ""
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""

def fmt_coef_se(coef: float, se: float, pvalue: float, digits: int = 3) -> str:
    return f"{coef:.{digits}f}{stars(pvalue)} ({se:.{digits}f})"

CORE_SAMPLE_VARS = [v for v in CORE_SAMPLE_VARS if v != "industry_name_std"]
