from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

AGRI = "是否涉农"

FIRM = "证券代码"

OUTCOME = "green_patent_application_count"

ENTRY = "has_green_patent_application"

CONTROLS = ["ln_assets", "roa", "leverage", "ocf_ratio", "soe", "list_age"]

STORED_GTOS = [
    "industry_year_nonagri_patent_per_firm_lag1_log1p",
    "industry_year_has_patent_share_nonagri_lag1",
    "frontier_industry_flag_baseline",
]

REBUILT_GTOS = ["opportunity_intensity", "opportunity_diffusion", "frontier_exposure"]

OUTCOMES = [
    ("green_patent_application_count", "Patent count"),
    ("log1p_green_patent_application_count", "Log(1 + patent count)"),
    ("has_green_patent_application", "Patent entry indicator"),
]

def build_gtos(data: pd.DataFrame, code: str) -> tuple[pd.DataFrame, float]:
    work = data.copy()
    cells = work[[code, "year"]].drop_duplicates()
    nonagricultural = work.loc[work[AGRI].eq(0)].copy()
    nonagricultural_cells = (
        nonagricultural.groupby([code, "year"], as_index=False)
        .agg(
            nonagricultural_firms=(FIRM, "size"),
            nonagricultural_patents=(OUTCOME, "sum"),
            nonagricultural_patenting_firms=(ENTRY, "sum"),
        )
    )
    industry_year = cells.merge(
        nonagricultural_cells,
        on=[code, "year"],
        how="left",
        validate="1:1",
    )
    for column in [
        "nonagricultural_firms",
        "nonagricultural_patents",
        "nonagricultural_patenting_firms",
    ]:
        industry_year[column] = industry_year[column].fillna(0)
    positive_support = industry_year["nonagricultural_firms"].gt(0)
    industry_year["current_intensity"] = np.where(
        positive_support,
        industry_year["nonagricultural_patents"]
        / industry_year["nonagricultural_firms"],
        np.nan,
    )
    industry_year["current_diffusion"] = np.where(
        positive_support,
        industry_year["nonagricultural_patenting_firms"]
        / industry_year["nonagricultural_firms"],
        np.nan,
    )
    industry_year = industry_year.sort_values([code, "year"]).copy()
    lagged = industry_year[
        [code, "year", "current_intensity", "current_diffusion"]
    ].copy()
    lagged["year"] = lagged["year"] + 1
    lagged = lagged.rename(
        columns={
            "current_intensity": "lagged_intensity",
            "current_diffusion": "lagged_diffusion",
        }
    )
    industry_year = industry_year.merge(
        lagged,
        on=[code, "year"],
        how="left",
        validate="1:1",
    )

    baseline = (
        industry_year.loc[industry_year["year"].between(2007, 2012)]
        .groupby(code, as_index=False)
        .agg(
            baseline_intensity=("current_intensity", "mean"),
            support_years=(
                "current_intensity",
                lambda values: int(values.notna().sum()),
            ),
        )
        .query("support_years >= 3")
        .copy()
    )
    frontier_threshold = float(baseline["baseline_intensity"].quantile(0.75))
    baseline["frontier_exposure"] = baseline["baseline_intensity"].ge(
        frontier_threshold
    ).astype(float)

    rebuilt = (
        work.merge(
            industry_year[
                [code, "year", "lagged_intensity", "lagged_diffusion"]
            ],
            on=[code, "year"],
            how="left",
            validate="m:1",
        )
        .merge(
            baseline[[code, "frontier_exposure"]],
            on=code,
            how="left",
            validate="m:1",
        )
        .rename(columns={"lagged_diffusion": "opportunity_diffusion"})
    )
    rebuilt["opportunity_intensity"] = np.log1p(rebuilt["lagged_intensity"])
    return rebuilt, frontier_threshold

def decomposition(data: pd.DataFrame, code: str) -> dict[str, float | int]:
    work = data[["year", AGRI, code, OUTCOME]].dropna()
    agricultural = work.loc[work[AGRI].eq(1)]
    nonagricultural = work.loc[work[AGRI].eq(0)]
    agricultural_cells = (
        agricultural.groupby(["year", code], as_index=False)
        .agg(
            agricultural_n=(OUTCOME, "size"),
            agricultural_mean=(OUTCOME, "mean"),
        )
    )
    nonagricultural_cells = (
        nonagricultural.groupby(["year", code], as_index=False)
        .agg(nonagricultural_mean=(OUTCOME, "mean"))
    )
    supported = agricultural_cells.merge(
        nonagricultural_cells,
        on=["year", code],
        how="left",
        validate="1:1",
    ).dropna(subset=["nonagricultural_mean"])
    supported_agricultural = agricultural.merge(
        supported[["year", code]],
        on=["year", code],
        how="inner",
        validate="m:1",
    )
    supported["agricultural_weight"] = (
        supported["agricultural_n"] / supported["agricultural_n"].sum()
    )
    agricultural_mean = float(supported_agricultural[OUTCOME].mean())
    nonagricultural_mean = float(nonagricultural[OUTCOME].mean())
    within = float(
        (
            supported["agricultural_weight"]
            * (
                supported["agricultural_mean"]
                - supported["nonagricultural_mean"]
            )
        ).sum()
    )
    between = float(
        (
            supported["agricultural_weight"]
            * supported["nonagricultural_mean"]
        ).sum()
        - nonagricultural_mean
    )
    gap = agricultural_mean - nonagricultural_mean
    if not np.isclose(gap, within + between, atol=1e-10):
        raise AssertionError("The accounting identity does not hold.")
    return {
        "Panel observations": len(work),
        "Supported cells": len(supported),
        "Supported agricultural firm-years": len(supported_agricultural),
        "Agricultural mean": agricultural_mean,
        "Non-agricultural mean": nonagricultural_mean,
        "Aggregate gap": gap,
        "Within-cell component": within,
        "Between-cell component": between,
        "Between-cell share": between / gap,
    }

def accounting_models(data: pd.DataFrame) -> dict[str, float | int]:
    required = [
        OUTCOME,
        AGRI,
        "year",
        FIRM,
        "province",
        *CONTROLS,
        *REBUILT_GTOS,
    ]
    sample = data.dropna(subset=required).copy()
    control_rhs = (
        AGRI
        + " + "
        + " + ".join(CONTROLS)
        + " + C(province) + C(year)"
    )
    full_rhs = control_rhs + " + " + " + ".join(REBUILT_GTOS)
    control_model = smf.ols(
        f"{OUTCOME} ~ {control_rhs}", data=sample
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample[FIRM]},
    )
    full_model = smf.ols(
        f"{OUTCOME} ~ {full_rhs}", data=sample
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample[FIRM]},
    )
    return {
        "Accounting observations": len(sample),
        "Accounting firms": sample[FIRM].nunique(),
        "Control-adjusted agricultural coefficient": float(
            control_model.params[AGRI]
        ),
        "Control-adjusted standard error": float(control_model.bse[AGRI]),
        "Full-accounting agricultural coefficient": float(
            full_model.params[AGRI]
        ),
        "Full-accounting standard error": float(full_model.bse[AGRI]),
        "Coefficient movement": float(
            control_model.params[AGRI] - full_model.params[AGRI]
        ),
    }

def standardized_baseline_models(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work["industry_code_std"] = work["industry_code_std"].fillna("UNKNOWN")
    specifications = [
        (
            "Industry and year effects",
            f"{AGRI} + C(year) + C(industry_code_std)",
            [],
        ),
        (
            "Industry and year effects; financial controls",
            f"{AGRI} + ln_assets + roa + leverage + ocf_ratio"
            " + C(year) + C(industry_code_std)",
            ["ln_assets", "roa", "leverage", "ocf_ratio"],
        ),
        (
            "Industry and year effects; full controls",
            f"{AGRI} + ln_assets + roa + leverage + ocf_ratio"
            " + soe + list_age + C(province) + C(year)"
            " + C(industry_code_std)",
            ["ln_assets", "roa", "leverage", "ocf_ratio", "soe", "list_age", "province"],
        ),
    ]
    rows: list[dict[str, object]] = []
    for outcome, outcome_label in OUTCOMES:
        for specification, rhs, controls in specifications:
            sample = work.dropna(
                subset=[
                    outcome,
                    AGRI,
                    "year",
                    FIRM,
                    "industry_code_std",
                    *controls,
                ]
            ).copy()
            result = smf.ols(f"{outcome} ~ {rhs}", data=sample).fit(
                cov_type="cluster",
                cov_kwds={"groups": sample[FIRM]},
            )
            rows.append(
                {
                    "Outcome": outcome_label,
                    "Specification": specification,
                    "Agricultural coefficient": float(result.params[AGRI]),
                    "Standard error": float(result.bse[AGRI]),
                    "P-value": float(result.pvalues[AGRI]),
                    "Observations": len(sample),
                    "R-squared": float(result.rsquared),
                }
            )
    return pd.DataFrame(rows)

def verify_hybrid_rebuild(panel: pd.DataFrame, rebuilt: pd.DataFrame) -> None:
    merged = panel[
        [
            FIRM,
            "year",
            *STORED_GTOS,
        ]
    ].merge(
        rebuilt[
            [
                FIRM,
                "year",
                *REBUILT_GTOS,
            ]
        ],
        on=[FIRM, "year"],
        how="left",
        validate="1:1",
    )
    pairs = [
        (
            "industry_year_nonagri_patent_per_firm_lag1_log1p",
            "opportunity_intensity",
        ),
        (
            "industry_year_has_patent_share_nonagri_lag1",
            "opportunity_diffusion",
        ),
    ]
    for stored, reconstructed in pairs:
        comparable = merged[stored].notna() & merged[reconstructed].notna()
        max_difference = float(
            (
                merged.loc[comparable, stored]
                - merged.loc[comparable, reconstructed]
            )
            .abs()
            .max()
        )
        if max_difference > 1e-12:
            raise AssertionError(
                f"Hybrid rebuild does not reproduce {stored}: {max_difference}"
            )
