from __future__ import annotations

import numpy as np
import pandas as pd

RD_NUMERIC_COLUMNS = [
    "RDPerson",
    "RDPersonRatio",
    "RDSpendSum",
    "RDSpendSumRatio",
    "RDExpenses",
    "RDInvest",
    "RDInvestRatio",
    "RDInvestNetprofitRatio",
]

def select_strict_annual(rd: pd.DataFrame) -> pd.DataFrame:
    strict = rd.loc[
        rd["Source"].eq(0)
        & rd["StateTypeCode"].eq(1)
        & rd["year_end_flag"]
        & rd["year"].notna()
        & rd["证券代码"].notna()
    ].copy()
    strict = strict.sort_values(
        ["证券代码", "year", "rd_field_completeness", "EndDate"],
        ascending=[True, True, False, False],
    )
    strict = strict.drop_duplicates(["证券代码", "year"], keep="first")
    strict["selection_rule"] = "Periodic report; consolidated statement; fiscal year-end"
    return strict

def merge_rd(
    panel: pd.DataFrame,
    selected: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    rd_columns = [
        "证券代码",
        "year",
        "Source",
        "StateTypeCode",
        "EndDate",
        *RD_NUMERIC_COLUMNS,
    ]
    renamed = selected[rd_columns].rename(
        columns={column: f"{prefix}_{column}" for column in rd_columns if column not in {"证券代码", "year"}}
    )
    merged = panel.merge(renamed, on=["证券代码", "year"], how="left", validate="one_to_one")
    return merged

def build_industry_rd_environment(
    panel: pd.DataFrame,
    strict_merged: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataframe = strict_merged.copy()
    dataframe["rd_intensity_direct_fraction"] = dataframe["strict_RDSpendSumRatio"] / 100
    dataframe["rd_intensity_calculated"] = (
        dataframe["strict_RDSpendSum"] / dataframe["revenue_used_y"]
    )
    dataframe["rd_person_share_fraction"] = dataframe["strict_RDPersonRatio"] / 100
    dataframe["has_rd_record"] = dataframe["strict_Source"].notna().astype(int)
    dataframe["has_total_rd"] = dataframe["strict_RDSpendSum"].notna().astype(int)
    dataframe["has_positive_rd"] = dataframe["strict_RDSpendSum"].gt(0).astype(int)

    denominator = (
        panel.groupby(["industry_code_std", "year"])
        .agg(
            total_firms=("证券代码", "size"),
            nonagricultural_firms=("是否涉农", lambda values: int(values.eq(0).sum())),
        )
        .reset_index()
    )
    nonagri = dataframe.loc[dataframe["是否涉农"].eq(0)].copy()
    cells = (
        nonagri.groupby(["industry_code_std", "year"])
        .agg(
            observed_nonagricultural_firms=("has_rd_record", "sum"),
            total_rd_observed_firms=("has_total_rd", "sum"),
            positive_rd_observed_firms=("has_positive_rd", "sum"),
            total_rd_sum=("strict_RDSpendSum", "sum"),
            median_total_rd=("strict_RDSpendSum", "median"),
            mean_direct_rd_intensity=("rd_intensity_direct_fraction", "mean"),
            median_direct_rd_intensity=("rd_intensity_direct_fraction", "median"),
            mean_calculated_rd_intensity=("rd_intensity_calculated", "mean"),
            median_calculated_rd_intensity=("rd_intensity_calculated", "median"),
            mean_rd_personnel_share=("rd_person_share_fraction", "mean"),
            median_rd_personnel_share=("rd_person_share_fraction", "median"),
        )
        .reset_index()
        .merge(denominator, on=["industry_code_std", "year"], how="left")
    )
    cells["rd_record_disclosure_rate"] = (
        cells["observed_nonagricultural_firms"] / cells["nonagricultural_firms"]
    )
    cells["total_rd_coverage_rate"] = (
        cells["total_rd_observed_firms"] / cells["nonagricultural_firms"]
    )
    cells["positive_rd_share_all_nonagricultural"] = (
        cells["positive_rd_observed_firms"] / cells["nonagricultural_firms"]
    )
    cells["observed_total_rd_per_all_nonagricultural_firm"] = (
        cells["total_rd_sum"] / cells["nonagricultural_firms"]
    )
    cells["log1p_observed_total_rd_per_all_nonagricultural_firm"] = np.log1p(
        cells["observed_total_rd_per_all_nonagricultural_firm"].clip(lower=0)
    )

    lag_columns = [
        "observed_nonagricultural_firms",
        "rd_record_disclosure_rate",
        "total_rd_coverage_rate",
        "positive_rd_share_all_nonagricultural",
        "mean_direct_rd_intensity",
        "median_direct_rd_intensity",
        "mean_calculated_rd_intensity",
        "median_calculated_rd_intensity",
        "mean_rd_personnel_share",
        "median_rd_personnel_share",
        "log1p_observed_total_rd_per_all_nonagricultural_firm",
    ]
    lagged = cells[["industry_code_std", "year", *lag_columns]].copy()
    lagged["year"] = lagged["year"] + 1
    lagged = lagged.rename(columns={column: f"{column}_lag1" for column in lag_columns})
    return cells, lagged

def industry_environment_coverage(
    panel: pd.DataFrame,
    common: pd.DataFrame,
    lagged: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_name, sample in [
        ("Full panel", panel),
        ("Common estimation sample", common),
    ]:
        merged = sample.merge(lagged, on=["industry_code_std", "year"], how="left")
        for minimum_disclosers in [1, 3, 5, 10]:
            covered = merged["observed_nonagricultural_firms_lag1"].ge(
                minimum_disclosers
            )
            agricultural = merged["是否涉农"].eq(1)
            rows.append(
                {
                    "sample": sample_name,
                    "minimum_lagged_nonagricultural_rd_records": minimum_disclosers,
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
    gtos_columns = [
        "industry_year_nonagri_patent_per_firm_lag1_log1p",
        "industry_year_has_patent_share_nonagri_lag1",
        "frontier_industry_flag_baseline",
    ]
    cell_gtos = (
        panel[
            [
                "industry_code_std",
                "year",
                *gtos_columns,
            ]
        ]
        .drop_duplicates(["industry_code_std", "year"])
        .merge(lagged, on=["industry_code_std", "year"], how="left")
    )
    rd_columns = [
        "mean_direct_rd_intensity_lag1",
        "median_direct_rd_intensity_lag1",
        "mean_calculated_rd_intensity_lag1",
        "mean_rd_personnel_share_lag1",
        "positive_rd_share_all_nonagricultural_lag1",
        "rd_record_disclosure_rate_lag1",
        "log1p_observed_total_rd_per_all_nonagricultural_firm_lag1",
    ]
    rows: list[dict[str, object]] = []
    for minimum_disclosers in [1, 3, 5]:
        supported = cell_gtos.loc[
            cell_gtos["observed_nonagricultural_firms_lag1"].ge(minimum_disclosers)
        ]
        for rd_column in rd_columns:
            for gtos_column in gtos_columns:
                pair = supported[[rd_column, gtos_column]].dropna()
                rows.append(
                    {
                        "minimum_lagged_nonagricultural_rd_records": minimum_disclosers,
                        "rd_environment_variable": rd_column,
                        "gtos_variable": gtos_column,
                        "industry_year_cells": len(pair),
                        "pearson_correlation": float(
                            pair.corr(method="pearson").iloc[0, 1]
                        )
                        if len(pair) > 2
                        else np.nan,
                        "spearman_correlation": float(
                            pair.corr(method="spearman").iloc[0, 1]
                        )
                        if len(pair) > 2
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows), cell_gtos

def group_exposure(
    panel: pd.DataFrame,
    lagged: pd.DataFrame,
) -> pd.DataFrame:
    merged = panel.merge(lagged, on=["industry_code_std", "year"], how="left")
    variables = [
        "mean_direct_rd_intensity_lag1",
        "mean_calculated_rd_intensity_lag1",
        "mean_rd_personnel_share_lag1",
        "positive_rd_share_all_nonagricultural_lag1",
        "rd_record_disclosure_rate_lag1",
        "log1p_observed_total_rd_per_all_nonagricultural_firm_lag1",
    ]
    rows: list[dict[str, object]] = []
    for minimum_disclosers in [1, 3, 5]:
        supported = merged.loc[
            merged["observed_nonagricultural_firms_lag1"].ge(minimum_disclosers)
        ]
        for variable in variables:
            grouped = supported.groupby("是否涉农")[variable]
            agricultural = grouped.get_group(1).dropna() if 1 in grouped.groups else pd.Series(dtype=float)
            nonagricultural = grouped.get_group(0).dropna() if 0 in grouped.groups else pd.Series(dtype=float)
            rows.append(
                {
                    "minimum_lagged_nonagricultural_rd_records": minimum_disclosers,
                    "variable": variable,
                    "agricultural_observations": len(agricultural),
                    "nonagricultural_observations": len(nonagricultural),
                    "agricultural_mean": float(agricultural.mean()) if len(agricultural) else np.nan,
                    "nonagricultural_mean": float(nonagricultural.mean()) if len(nonagricultural) else np.nan,
                    "mean_difference_agricultural_minus_nonagricultural": float(
                        agricultural.mean() - nonagricultural.mean()
                    )
                    if len(agricultural) and len(nonagricultural)
                    else np.nan,
                    "agricultural_median": float(agricultural.median()) if len(agricultural) else np.nan,
                    "nonagricultural_median": float(nonagricultural.median()) if len(nonagricultural) else np.nan,
                }
            )
    return pd.DataFrame(rows)
