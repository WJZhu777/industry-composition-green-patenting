from __future__ import annotations

import numpy as np
import pandas as pd
from .structural_utils import CORE_FIRM_CONTROLS
from .structural_utils import CORE_GTOS_VARS
from .structural_utils import PATENT_OUTCOMES
from .structural_utils import clustered_ols

AGRI = "是否涉农"

FIRM_ID = "证券代码"

GTOS_VARS = CORE_GTOS_VARS

OUTCOME_LABELS = {
    "green_patent_application_count": "Patent count",
    "log1p_green_patent_application_count": "Log(1 + patent count)",
    "has_green_patent_application": "Patent entry indicator",
}

def standardize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d{1,6})")[0].str.zfill(6)

def stars(pvalue: float) -> str:
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""

def format_estimate(coef: float, se: float, pvalue: float) -> str:
    return f"{coef:.3f}{stars(pvalue)} ({se:.3f})"

def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    low, high = valid.quantile([lower, upper])
    return numeric.clip(lower=low, upper=high)

def formula(
    outcome: str,
    controls: list[str],
    include_gtos: bool,
    include_industry_fe: bool = False,
) -> str:
    terms = [AGRI, "C(year)", *controls, "C(province)"]
    if include_industry_fe:
        terms.append("C(industry_code_std)")
    if include_gtos:
        terms.extend(GTOS_VARS)
    return f"{outcome} ~ " + " + ".join(terms)

def extract_model(
    result,
    outcome: str,
    specification: str,
    model_role: str,
    sample_label: str,
) -> dict[str, object]:
    return {
        "Outcome": OUTCOME_LABELS[outcome],
        "Specification": specification,
        "Model role": model_role,
        "Sample": sample_label,
        "Agricultural coefficient": float(result.params[AGRI]),
        "Standard error": float(result.bse[AGRI]),
        "P-value": float(result.pvalues[AGRI]),
        "Formatted estimate": format_estimate(
            float(result.params[AGRI]),
            float(result.bse[AGRI]),
            float(result.pvalues[AGRI]),
        ),
        "Observations": int(result.nobs),
        "Firms": int(result.model.data.frame[FIRM_ID].nunique()),
        "R-squared": float(result.rsquared),
    }

def run_candidate_control_sensitivity(common: pd.DataFrame) -> pd.DataFrame:
    candidate_specs = [
        {
            "label": "Revenue replaces total assets",
            "mask": common["ln_revenue"].notna(),
            "candidate_controls": [
                "ln_revenue",
                "roa",
                "leverage",
                "ocf_ratio",
                "soe",
                "list_age",
            ],
            "sample": "Common sample with revenue available",
        },
        {
            "label": "Employment scale added",
            "mask": common["ln_staff"].notna(),
            "candidate_controls": [*CORE_FIRM_CONTROLS, "ln_staff"],
            "sample": "Common sample with employment available",
        },
        {
            "label": "Lagged total R&D intensity added",
            "mask": (
                common["year"].between(2012, 2024)
                & common["lagged_total_rd_intensity_w"].notna()
            ),
            "candidate_controls": [
                *CORE_FIRM_CONTROLS,
                "lagged_total_rd_intensity_w",
            ],
            "sample": "2012-2024 common sample with lagged R&D available",
        },
    ]

    rows: list[dict[str, object]] = []
    for candidate in candidate_specs:
        dataframe = common.loc[candidate["mask"]].copy()
        for outcome in PATENT_OUTCOMES:
            models = [
                (
                    "Core controls, without GTOS",
                    CORE_FIRM_CONTROLS,
                    False,
                    "Candidate-specific reference",
                ),
                (
                    f"{candidate['label']}, without GTOS",
                    candidate["candidate_controls"],
                    False,
                    "Candidate-adjusted reference",
                ),
                (
                    "Core controls + GTOS",
                    CORE_FIRM_CONTROLS,
                    True,
                    "Candidate-specific full accounting reference",
                ),
                (
                    f"{candidate['label']} + GTOS",
                    candidate["candidate_controls"],
                    True,
                    "Candidate-adjusted full accounting",
                ),
            ]
            for specification, controls, include_gtos, role in models:
                result = clustered_ols(
                    formula(outcome, controls, include_gtos),
                    dataframe,
                )
                payload = extract_model(
                    result,
                    outcome,
                    specification,
                    role,
                    candidate["sample"],
                )
                payload["Candidate family"] = candidate["label"]
                rows.append(payload)
    results = pd.DataFrame(rows)

    reference = results.loc[
        results["Specification"].isin(
            ["Core controls, without GTOS", "Core controls + GTOS"]
        ),
        [
            "Candidate family",
            "Outcome",
            "Specification",
            "Agricultural coefficient",
        ],
    ].copy()
    reference["Reference type"] = reference["Specification"].map(
        {
            "Core controls, without GTOS": "without GTOS",
            "Core controls + GTOS": "with GTOS",
        }
    )
    reference = reference.rename(
        columns={"Agricultural coefficient": "Reference agricultural coefficient"}
    )[["Candidate family", "Outcome", "Reference type", "Reference agricultural coefficient"]]

    results["Reference type"] = np.where(
        results["Specification"].str.contains("without GTOS"),
        "without GTOS",
        "with GTOS",
    )
    results = results.merge(
        reference,
        on=["Candidate family", "Outcome", "Reference type"],
        how="left",
        validate="many_to_one",
    )
    results["Coefficient change versus same-sample core controls"] = (
        results["Agricultural coefficient"]
        - results["Reference agricultural coefficient"]
    )
    return results

def run_industry_fe_role_sensitivity(common: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome in PATENT_OUTCOMES:
        specifications = [
            (
                "Within-industry controls",
                formula(
                    outcome,
                    CORE_FIRM_CONTROLS,
                    include_gtos=False,
                    include_industry_fe=True,
                ),
                "Within-industry conditional association",
            ),
            (
                "Within-industry controls + time-varying GTOS",
                (
                    f"{outcome} ~ {AGRI} + C(year) + "
                    + " + ".join(CORE_FIRM_CONTROLS)
                    + " + C(province) + C(industry_code_std)"
                    + " + industry_year_nonagri_patent_per_firm_lag1_log1p"
                    + " + industry_year_has_patent_share_nonagri_lag1"
                ),
                (
                    "Within-industry temporal sensitivity; frontier omitted "
                    "because it is time-invariant by industry"
                ),
            ),
        ]
        for specification, model_formula, role in specifications:
            result = clustered_ols(model_formula, common)
            rows.append(
                extract_model(
                    result,
                    outcome,
                    specification,
                    role,
                    "Common estimation sample",
                )
            )
    return pd.DataFrame(rows)
