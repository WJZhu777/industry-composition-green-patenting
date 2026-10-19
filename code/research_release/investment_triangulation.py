from __future__ import annotations

import numpy as np
import pandas as pd

def build_matched_data(
    green: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    green_window = green.loc[green["year"].between(2007, 2024)].copy()
    matched = green_window.merge(
        panel[
            [
                "证券代码",
                "year",
                "是否涉农",
                "industry_code_std",
                "industry_name_std",
            ]
        ],
        on=["证券代码", "year"],
        how="inner",
    )
    matched["green_log_total"] = np.log1p(matched["绿色总投资"].clip(lower=0))
    matched["green_log_capitalized"] = np.log1p(
        matched["资本化绿色投资额"].clip(lower=0)
    )
    return green_window, matched

def build_industry_year_investment(
    panel: pd.DataFrame,
    matched: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    denominator = (
        panel.groupby(["industry_code_std", "year"])
        .agg(
            all_firms=("证券代码", "size"),
            all_nonagricultural_firms=("是否涉农", lambda values: (values == 0).sum()),
        )
        .reset_index()
    )

    nonagricultural = matched.loc[matched["是否涉农"].eq(0)].copy()
    cells = (
        nonagricultural.groupby(["industry_code_std", "year"])
        .agg(
            disclosed_nonagricultural_firms=("证券代码", "nunique"),
            disclosed_green_total_sum=("绿色总投资", "sum"),
            disclosed_green_capitalized_sum=("资本化绿色投资额", "sum"),
            disclosed_green_expensed_sum=("费用化绿色投资额", "sum"),
            mean_log_green_total_among_disclosers=("green_log_total", "mean"),
            mean_log_green_capitalized_among_disclosers=(
                "green_log_capitalized",
                "mean",
            ),
            median_green_total_among_disclosers=("绿色总投资", "median"),
            mean_capitalized_share_among_disclosers=("绿色投资2", "mean"),
            median_capitalized_share_among_disclosers=("绿色投资2", "median"),
        )
        .reset_index()
        .merge(denominator, on=["industry_code_std", "year"], how="left")
    )
    cells["nonagricultural_disclosure_rate"] = (
        cells["disclosed_nonagricultural_firms"]
        / cells["all_nonagricultural_firms"]
    )
    cells["disclosed_green_total_per_all_nonagricultural_firm"] = (
        cells["disclosed_green_total_sum"] / cells["all_nonagricultural_firms"]
    )
    cells["disclosed_green_capitalized_per_all_nonagricultural_firm"] = (
        cells["disclosed_green_capitalized_sum"]
        / cells["all_nonagricultural_firms"]
    )
    cells["log_disclosed_green_total_per_all_nonagricultural_firm"] = np.log1p(
        cells["disclosed_green_total_per_all_nonagricultural_firm"].clip(lower=0)
    )
    cells[
        "log_disclosed_green_capitalized_per_all_nonagricultural_firm"
    ] = np.log1p(
        cells[
            "disclosed_green_capitalized_per_all_nonagricultural_firm"
        ].clip(lower=0)
    )

    lag_columns = [
        "disclosed_nonagricultural_firms",
        "nonagricultural_disclosure_rate",
        "mean_log_green_total_among_disclosers",
        "mean_log_green_capitalized_among_disclosers",
        "mean_capitalized_share_among_disclosers",
        "log_disclosed_green_total_per_all_nonagricultural_firm",
        "log_disclosed_green_capitalized_per_all_nonagricultural_firm",
    ]
    lagged = cells[["industry_code_std", "year", *lag_columns]].copy()
    lagged["year"] = lagged["year"] + 1
    lagged = lagged.rename(columns={column: f"{column}_lag1" for column in lag_columns})
    return cells, lagged

def cell_coverage(
    panel: pd.DataFrame,
    common: pd.DataFrame,
    lagged: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_name, sample in [("Full panel", panel), ("Common estimation sample", common)]:
        merged = sample.merge(
            lagged,
            on=["industry_code_std", "year"],
            how="left",
        )
        for minimum_disclosers in [1, 3, 5, 10]:
            covered = merged["disclosed_nonagricultural_firms_lag1"].ge(
                minimum_disclosers
            )
            agricultural = merged["是否涉农"].eq(1)
            rows.append(
                {
                    "sample": sample_name,
                    "minimum_nonagricultural_disclosers": minimum_disclosers,
                    "observations": len(merged),
                    "covered_observations": int(covered.sum()),
                    "coverage_rate": float(covered.mean()),
                    "agricultural_coverage_rate": float(
                        covered.loc[agricultural].mean()
                    ),
                    "nonagricultural_coverage_rate": float(
                        covered.loc[~agricultural].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)

def gtos_correlations(
    panel: pd.DataFrame,
    lagged: pd.DataFrame,
) -> pd.DataFrame:
    current_gtos = panel[
        [
            "industry_code_std",
            "year",
            "industry_year_nonagri_patent_per_firm_lag1_log1p",
            "industry_year_has_patent_share_nonagri_lag1",
        ]
    ].drop_duplicates(["industry_code_std", "year"])
    cells = current_gtos.merge(
        lagged,
        on=["industry_code_std", "year"],
        how="left",
    )
    cells = cells.loc[
        cells["disclosed_nonagricultural_firms_lag1"].ge(3)
    ].copy()

    candidate_metrics = [
        "mean_log_green_total_among_disclosers_lag1",
        "mean_log_green_capitalized_among_disclosers_lag1",
        "mean_capitalized_share_among_disclosers_lag1",
        "log_disclosed_green_total_per_all_nonagricultural_firm_lag1",
        "log_disclosed_green_capitalized_per_all_nonagricultural_firm_lag1",
        "nonagricultural_disclosure_rate_lag1",
    ]
    patent_metrics = [
        "industry_year_nonagri_patent_per_firm_lag1_log1p",
        "industry_year_has_patent_share_nonagri_lag1",
    ]
    rows: list[dict[str, object]] = []
    for investment_metric in candidate_metrics:
        for patent_metric in patent_metrics:
            data = cells[[investment_metric, patent_metric]].dropna()
            rows.append(
                {
                    "green_investment_metric": investment_metric,
                    "patent_gtos_metric": patent_metric,
                    "industry_year_cells": len(data),
                    "pearson_correlation": float(
                        data[investment_metric].corr(
                            data[patent_metric],
                            method="pearson",
                        )
                    ),
                    "spearman_correlation": float(
                        data[investment_metric].corr(
                            data[patent_metric],
                            method="spearman",
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)

def group_exposure_summary(
    common: pd.DataFrame,
    lagged: pd.DataFrame,
) -> pd.DataFrame:
    merged = common.merge(
        lagged,
        on=["industry_code_std", "year"],
        how="left",
    )
    merged = merged.loc[
        merged["disclosed_nonagricultural_firms_lag1"].ge(3)
    ].copy()
    metrics = [
        "mean_log_green_capitalized_among_disclosers_lag1",
        "log_disclosed_green_capitalized_per_all_nonagricultural_firm_lag1",
        "mean_capitalized_share_among_disclosers_lag1",
        "nonagricultural_disclosure_rate_lag1",
    ]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for agricultural_value, label in [(1, "Agricultural firms"), (0, "Non-agricultural firms")]:
            values = merged.loc[merged["是否涉农"].eq(agricultural_value), metric]
            rows.append(
                {
                    "metric": metric,
                    "group": label,
                    "observations": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "standard_deviation": float(values.std()),
                }
            )
    return pd.DataFrame(rows)
