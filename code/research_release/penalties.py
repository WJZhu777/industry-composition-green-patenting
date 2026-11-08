"""Environmental-penalty comparisons within the source's complete calendar years."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

FIRM = "\u8bc1\u5238\u4ee3\u7801"
AGRI = "\u662f\u5426\u6d89\u519c"
OUTCOME = "is_penalized_announcement_year"
INDUSTRY = "industry_code_std"
FIRST_YEAR, LAST_YEAR = 2009, 2023
FINANCIAL = ["ln_assets", "roa", "leverage", "ocf_ratio"]
CONTROLS = [*FINANCIAL, "soe", "list_age"]
GTOS = [
    "industry_year_nonagri_patent_per_firm_lag1_log1p",
    "industry_year_has_patent_share_nonagri_lag1",
    "frontier_industry_flag_baseline",
]
BASE = f"{AGRI} + C(year) + " + " + ".join(CONTROLS) + " + C(province)"


def covered_sample(frame: pd.DataFrame) -> pd.DataFrame:
    # Pre-2009 zeros are outside source coverage; 2024 ends on April 30.
    return frame.loc[frame["year"].between(FIRST_YEAR, LAST_YEAR)].copy()


def sample_key(frame: pd.DataFrame) -> str:
    keys = frame[[FIRM, "year"]].sort_values([FIRM, "year"])
    return hashlib.sha256(keys.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def fit(frame: pd.DataFrame, rhs: str, terms: list[str]) -> dict:
    model = smf.ols(f"{OUTCOME} ~ {rhs}", data=frame, missing="raise")
    result = model.fit(
        cov_type="cluster", use_t=False,
        cov_kwds={"groups": frame[FIRM], "use_correction": True},
    )
    if not np.isfinite(np.r_[result.params, result.bse, result.pvalues]).all():
        raise ValueError("Nonfinite penalty-model estimates.")
    intervals = result.conf_int()
    return {
        "formula": model.formula,
        "n": int(result.nobs), "firms": int(frame[FIRM].nunique()),
        "agricultural_n": int(frame[AGRI].eq(1).sum()),
        "first_year": int(frame.year.min()), "last_year": int(frame.year.max()),
        "sample_sha256": sample_key(frame), "design_columns": model.exog_names,
        "r_squared": float(result.rsquared),
        "terms": {term: {
            "coefficient": float(result.params[term]), "se": float(result.bse[term]),
            "p_value": float(result.pvalues[term]),
            "ci_low": float(intervals.loc[term, 0]), "ci_high": float(intervals.loc[term, 1]),
        } for term in terms},
    }


def baseline_specs() -> list[tuple[str, str, list[str]]]:
    base = f"{AGRI} + C(year) + C({INDUSTRY})"
    financial = " + ".join(FINANCIAL)
    full = " + ".join(CONTROLS) + " + C(province)"
    winsor = " + ".join(x + "_w" for x in FINANCIAL)
    return [
        ("Industry and year effects", base, []),
        ("Industry and year effects; financial controls", base + " + " + financial, FINANCIAL),
        ("Industry and year effects; financial controls and state ownership",
         base + " + " + financial + " + soe", FINANCIAL + ["soe"]),
        ("Industry and year effects; financial controls and province effects",
         base + " + " + financial + " + C(province)", FINANCIAL + ["province"]),
        ("Industry and year effects; full controls", base + " + " + full, CONTROLS + ["province"]),
        ("Winsorized-control sensitivity", base + " + " + winsor + " + soe + list_age + C(province)",
         [x + "_w" for x in FINANCIAL] + ["soe", "list_age", "province"]),
    ]


def prepare_baseline(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    prepared = panel.copy()
    prepared[INDUSTRY] = prepared[INDUSTRY].fillna("UNKNOWN")
    cutoffs = {}
    # Preserve the original full-panel winsor cutoffs; only penalty years change.
    for variable in FINANCIAL:
        low, high = prepared[variable].dropna().quantile([0.01, 0.99])
        prepared[variable + "_w"] = prepared[variable].clip(lower=low, upper=high)
        cutoffs[variable] = [float(low), float(high)]
    return prepared, cutoffs


def estimate(panel: pd.DataFrame, common: pd.DataFrame) -> dict:
    if panel.duplicated([FIRM, "year"]).any() or common.duplicated([FIRM, "year"]).any():
        raise ValueError("Penalty inputs must be unique by firm and year.")
    if not common.set_index([FIRM, "year"]).index.isin(panel.set_index([FIRM, "year"]).index).all():
        raise ValueError("Common sample is not a subset of the panel.")
    before = (sample_key(panel), sample_key(common))
    prepared, cutoffs = prepare_baseline(panel)
    window = covered_sample(prepared)
    matched = covered_sample(common)
    if not window[OUTCOME].isin([0, 1]).all():
        raise ValueError("Covered penalty observations must be nonmissing binary records.")
    if matched[CONTROLS + GTOS + [OUTCOME, FIRM, AGRI, "province", "year"]].isna().any().any():
        raise ValueError("The fixed accounting sample contains missing penalty-model inputs.")
    panel_a = []
    for label, rhs, controls in baseline_specs():
        selected = window.dropna(subset=[OUTCOME, FIRM, AGRI, "year", INDUSTRY, *controls])
        panel_a.append({"specification": label, **fit(selected, rhs, [AGRI])})
    panel_b, panel_c = [], []
    models = [("Control-adjusted model", BASE), ("Full accounting model", BASE + " + " + " + ".join(GTOS))]
    for period, selected in [
        ("2009-2014", matched.loc[matched.year.le(2014)].copy()),
        ("2015-2023", matched.loc[matched.year.ge(2015)].copy()),
    ]:
        for label, rhs in models:
            panel_b.append({"period": period, "specification": label, **fit(selected, rhs, [AGRI])})
    matched["post_2015"] = matched.year.ge(2015).astype(int)
    interaction = AGRI + ":post_2015"
    for label, rhs in models:
        panel_c.append({"specification": label, **fit(matched, rhs + " + " + interaction, [AGRI, interaction])})
    if before != (sample_key(panel), sample_key(common)):
        raise AssertionError("The penalty comparison changed an input sample.")
    return {
        "definition": "Recorded environmental-penalty announcements, not a measure of all violations.",
        "coverage": {"first_year": FIRST_YEAR, "last_year": LAST_YEAR,
                     "reason": "The export starts in 2009 and ends on 2024-04-30; exclude unobserved early years and incomplete 2024."},
        "inference": "Firm-clustered covariance with finite-sample correction; normal-reference coefficient tests and intervals.",
        "sample": {"panel_window_n": len(window), "accounting_window_n": len(matched),
                   "accounting_window_firms": int(matched[FIRM].nunique()),
                   "original_panel_n": len(panel), "original_common_n": len(common)},
        "winsor_cutoffs_full_panel": cutoffs,
        "panel_a": panel_a, "panel_b": panel_b, "panel_c": panel_c,
    }
