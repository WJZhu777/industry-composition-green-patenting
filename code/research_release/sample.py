from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

FIRM = "\u8bc1\u5238\u4ee3\u7801"

AGRI = "\u662f\u5426\u6d89\u519c"

INDUSTRY = "industry_code_std"

COUNT = "green_patent_application_count"

LOG_COUNT = "log1p_green_patent_application_count"

ENTRY = "has_green_patent_application"

FINANCIAL = ["ln_assets", "roa", "leverage", "ocf_ratio"]

STATIC = ["soe", "province", "list_age", INDUSTRY]

CORE = [FIRM, "year", AGRI, COUNT, LOG_COUNT, ENTRY, "is_penalized_announcement_year"]

LAGGED = [
    "industry_year_nonagri_patent_per_firm_lag1_log1p",
    "industry_year_has_patent_share_nonagri_lag1",
]

FI = "frontier_industry_flag_baseline"

NUMERIC = ["year", AGRI, COUNT, LOG_COUNT, ENTRY, "is_penalized_announcement_year", *FINANCIAL, "soe", "list_age", *LAGGED, FI]

GROUPS = {"All": None, "Agricultural": 1, "Non-agricultural": 0}

STAGES = {
    "Full panel": [],
    "Core identifiers and outcomes": CORE,
    "Financial controls complete": CORE + FINANCIAL,
    "Static controls and fields complete": CORE + FINANCIAL + STATIC,
    "Lagged OI and OD available": CORE + FINANCIAL + STATIC + LAGGED,
    "Common estimation sample": CORE + FINANCIAL + STATIC + LAGGED + [FI],
}

EXPECTED_COUNTS = [55588, 55588, 51696, 49982, 48519, 47773]

METRICS = {
    "Calendar year": ("year", "mean"),
    "Patent count, mean": (COUNT, "mean"),
    "Patent count, median": (COUNT, "median"),
    "Patent entry rate": (ENTRY, "mean"),
    "Log(1 + patent count)": (LOG_COUNT, "mean"),
    "Log total assets": ("ln_assets", "mean"),
    "Return on assets": ("roa", "mean"),
    "Leverage": ("leverage", "mean"),
    "Operating cash flow / assets": ("ocf_ratio", "mean"),
    "State ownership share": ("soe", "mean"),
    "Listing age": ("list_age", "mean"),
}

def validate_panel(frame: pd.DataFrame) -> None:
    required = list(dict.fromkeys(CORE + FINANCIAL + STATIC + LAGGED + [FI]))
    absent = sorted(set(required) - set(frame.columns))
    if absent:
        raise ValueError(f"Missing required columns: {absent}")
    if frame[[FIRM, "year", AGRI]].isna().any().any():
        raise ValueError("Firm, year, or agricultural status is missing.")
    if not frame[FIRM].str.fullmatch(r"\d{6}").all():
        raise ValueError("Firm keys are not six-digit stock codes.")
    if frame.duplicated([FIRM, "year"]).any():
        raise ValueError("Panel is not unique by firm and year.")
    if not frame[AGRI].isin([0, 1]).all():
        raise ValueError("Agricultural status is not binary.")
    if frame.groupby(FIRM)[AGRI].nunique().gt(1).any():
        raise ValueError("The stored CCAD grouping is not time-invariant.")
    numeric = frame[NUMERIC].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("Infinite values are present; no implicit deletion is permitted.")
    if (frame["year"] % 1).ne(0).any():
        raise ValueError("Calendar years are not integers.")
    for variable in ["soe", ENTRY, FI, "is_penalized_announcement_year"]:
        if not frame[variable].dropna().isin([0, 1]).all():
            raise ValueError(f"Nonbinary values in {variable}.")
    available = frame[COUNT].notna()
    counts = frame.loc[available, COUNT]
    if counts.lt(0).any() or (counts % 1).ne(0).any():
        raise ValueError("Patent counts are not nonnegative integers.")
    paired = available & frame[ENTRY].notna()
    if not frame.loc[paired, ENTRY].eq(frame.loc[paired, COUNT].gt(0).astype(int)).all():
        raise ValueError("Patent entry is inconsistent with the stored count.")
    paired = available & frame[LOG_COUNT].notna()
    if not np.allclose(frame.loc[paired, LOG_COUNT], np.log1p(frame.loc[paired, COUNT]), rtol=0, atol=1e-10):
        raise ValueError("Stored log1p outcome is inconsistent with the count.")

def load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={FIRM: str, INDUSTRY: str, "province": str})
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    # Use one category for missing industry codes.
    frame[INDUSTRY] = frame[INDUSTRY].fillna("UNKNOWN")
    validate_panel(frame)
    return frame

def group_mask(frame: pd.DataFrame, value: int | None) -> pd.Series:
    return pd.Series(True, index=frame.index) if value is None else frame[AGRI].eq(value)

def build_masks(frame: pd.DataFrame) -> tuple[dict, dict]:
    masks = {
        name: frame[required].notna().all(axis=1) if required else pd.Series(True, index=frame.index)
        for name, required in STAGES.items()
    }
    previous = masks["Full panel"]
    for name, mask in masks.items():
        if (mask & ~previous).any():
            raise ValueError(f"Non-nested sample stage: {name}")
        previous = mask
    common = masks["Common estimation sample"]
    static = masks["Static controls and fields complete"]
    excluded = {
        "All excluded full-panel observations": ~common,
        "GTOS-missing within static-complete sample": static & ~common,
        "Post-2007 GTOS-missing within static-complete sample": static & ~common & frame["year"].gt(2007),
    }
    if common.any() and not frame.loc[common, INDUSTRY].str.fullmatch(r"[A-Z]\d{2}").all():
        raise ValueError("The common sample contains invalid detailed industry codes.")
    return masks, excluded

def sample_flow(frame: pd.DataFrame, masks: dict) -> pd.DataFrame:
    rows = []
    for group, value in GROUPS.items():
        membership = group_mask(frame, value)
        full_n = int(membership.sum())
        previous_n = full_n
        for stage, mask in masks.items():
            selected = frame.loc[mask & membership]
            n = len(selected)
            rows.append({
                "group": group, "stage": stage, "observations": n,
                "firms": int(selected[FIRM].nunique()),
                "lost_from_previous_stage": previous_n - n,
                "retention_from_previous_stage": n / previous_n if previous_n else None,
                "share_of_group_full_panel": n / full_n if full_n else None,
                "agricultural_share": float(selected[AGRI].mean()) if n else None,
            })
            previous_n = n
    return pd.DataFrame(rows)

def first_failures(frame: pd.DataFrame, masks: dict) -> pd.DataFrame:
    stage_names = list(masks)
    failures = {
        f"First fails: {stage_names[index]}": masks[stage_names[index - 1]] & ~masks[stage_names[index]]
        for index in range(1, len(stage_names))
    }
    failures["Retained in common estimation sample"] = masks[stage_names[-1]]
    if not pd.DataFrame(failures).sum(axis=1).eq(1).all():
        raise ValueError("First-failure groups are not an exhaustive disjoint partition.")
    rows = []
    for group, value in GROUPS.items():
        membership = group_mask(frame, value)
        for reason, mask in failures.items():
            selected = frame.loc[mask & membership]
            rows.append({"group": group, "reason": reason, "observations": len(selected),
                         "firms": int(selected[FIRM].nunique())})
    return pd.DataFrame(rows)

def standardized_difference(retained: pd.Series, excluded: pd.Series) -> float | None:
    left, right = retained.dropna(), excluded.dropna()
    if len(left) < 2 or len(right) < 2:
        return None
    variance = (left.var(ddof=1) + right.var(ddof=1)) / 2
    if not np.isfinite(variance) or variance <= 0:
        return None
    return float((left.mean() - right.mean()) / np.sqrt(variance))

def compare_samples(frame: pd.DataFrame, common: pd.Series, excluded_masks: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows, metric_rows = [], []
    for scope, excluded in excluded_masks.items():
        if (common & excluded).any():
            raise ValueError("Retained and excluded comparison samples overlap.")
        for group, value in GROUPS.items():
            membership = group_mask(frame, value)
            retained_frame = frame.loc[common & membership]
            excluded_frame = frame.loc[excluded & membership]
            sample_rows.append({
                "comparison": scope, "group": group,
                "retained_n": len(retained_frame), "excluded_n": len(excluded_frame),
                "retained_firms": int(retained_frame[FIRM].nunique()),
                "excluded_firms": int(excluded_frame[FIRM].nunique()),
                "firms_in_both": len(set(retained_frame[FIRM]) & set(excluded_frame[FIRM])),
                "retained_agricultural_share": float(retained_frame[AGRI].mean()) if len(retained_frame) else None,
                "excluded_agricultural_share": float(excluded_frame[AGRI].mean()) if len(excluded_frame) else None,
            })
            for label, (variable, statistic) in METRICS.items():
                left, right = retained_frame[variable], excluded_frame[variable]
                left_valid, right_valid = left.dropna(), right.dropna()
                left_value = float(getattr(left_valid, statistic)()) if len(left_valid) else None
                right_value = float(getattr(right_valid, statistic)()) if len(right_valid) else None
                metric_rows.append({
                    "comparison": scope, "group": group, "metric": label, "variable": variable,
                    "statistic": statistic, "retained_value": left_value, "excluded_value": right_value,
                    "retained_minus_excluded": left_value - right_value if left_value is not None and right_value is not None else None,
                    "standardized_mean_difference": standardized_difference(left, right) if statistic == "mean" else None,
                    "retained_valid_n": len(left_valid), "excluded_valid_n": len(right_valid),
                    "retained_missing_n": int(left.isna().sum()), "excluded_missing_n": int(right.isna().sum()),
                })
    return pd.DataFrame(sample_rows), pd.DataFrame(metric_rows)

def categorical_retention(frame: pd.DataFrame, common: pd.Series, excluded_masks: dict, static: pd.Series) -> pd.DataFrame:
    rows = []
    samples = {"full": pd.Series(True, index=frame.index), "common": common, "static_complete": static,
               "all_excluded": excluded_masks["All excluded full-panel observations"],
               "gtos_missing": excluded_masks["GTOS-missing within static-complete sample"],
               "post2007_gtos_missing": excluded_masks["Post-2007 GTOS-missing within static-complete sample"]}
    for group, value in GROUPS.items():
        membership = group_mask(frame, value)
        for variable in ["year", INDUSTRY, "soe"]:
            labels = frame[variable].map(lambda x: "Missing" if pd.isna(x) else str(int(x)) if variable in ["year", "soe"] else str(x))
            counts = {key: labels.loc[mask & membership].value_counts() for key, mask in samples.items()}
            for key, mask in samples.items():
                if int(counts[key].sum()) != int((mask & membership).sum()):
                    raise ValueError(f"Categorical denominators lost observations: {group}/{variable}/{key}")
            for category in sorted(labels.loc[membership].unique()):
                row = {"group": group, "variable": variable, "category": category}
                for key in samples:
                    n, denominator = int(counts[key].get(category, 0)), int(counts[key].sum())
                    row[f"{key}_n"] = n
                    row[f"{key}_share"] = n / denominator if denominator else None
                row["retention_rate_from_full"] = row["common_n"] / row["full_n"] if row["full_n"] else None
                rows.append(row)
    return pd.DataFrame(rows)

def concentration(counts: pd.Series) -> dict:
    positive = counts.loc[counts.gt(0)].sort_values(ascending=False)
    if positive.empty:
        return {"clusters_with_observations": 0, "observations": 0}
    shares = positive / positive.sum()
    hhi = float(np.square(shares).sum())
    return {
        "clusters_with_observations": len(positive), "observations": int(positive.sum()),
        "minimum": int(positive.min()), "median": float(positive.median()), "maximum": int(positive.max()),
        "clusters_below_10_observations": int(positive.lt(10).sum()),
        "clusters_below_30_observations": int(positive.lt(30).sum()),
        "largest_cluster_share": float(shares.iloc[0]),
        "largest_three_share": float(shares.iloc[:3].sum()), "largest_five_share": float(shares.iloc[:5].sum()),
        "size_share_hhi": hhi, "inverse_size_share_hhi": 1 / hhi,
    }

def industry_clusters(frame: pd.DataFrame, common: pd.Series) -> tuple[pd.DataFrame, dict]:
    selected = frame.loc[common]
    for variable in [*LAGGED, FI]:
        if selected.groupby([INDUSTRY, "year"])[variable].nunique().gt(1).any():
            raise ValueError(f"Stored {variable} is not shared within industry-year cells.")
    if selected.groupby(INDUSTRY)[FI].nunique().gt(1).any():
        raise ValueError("The stored FI is not fixed within industries.")
    rows = []
    for code, subset in selected.groupby(INDUSTRY, sort=True):
        agricultural = subset[AGRI].eq(1)
        rows.append({
            "industry": code, "observations": len(subset), "firms": int(subset[FIRM].nunique()),
            "agricultural_observations": int(agricultural.sum()),
            "nonagricultural_observations": int((~agricultural).sum()),
            "agricultural_firms": int(subset.loc[agricultural, FIRM].nunique()),
            "nonagricultural_firms": int(subset.loc[~agricultural, FIRM].nunique()),
            "years": int(subset["year"].nunique()), "FI": int(subset[FI].iloc[0]),
            "share_of_common_sample": len(subset) / len(selected),
        })
    details = pd.DataFrame(rows, columns=["industry", "observations", "firms", "agricultural_observations", "nonagricultural_observations", "agricultural_firms", "nonagricultural_firms", "years", "FI", "share_of_common_sample"])
    details = details.sort_values(["observations", "industry"], ascending=[False, True]).reset_index(drop=True)
    summary = {group: concentration(details[column]) for group, column in [
        ("All", "observations"), ("Agricultural", "agricultural_observations"), ("Non-agricultural", "nonagricultural_observations")
    ]}
    firm_industries = selected.groupby(FIRM)[INDUSTRY].nunique()
    summary["firms_in_multiple_common_sample_industries"] = int(firm_industries.gt(1).sum())
    summary["common_sample_firms"] = int(selected[FIRM].nunique())
    summary["clusters_with_both_groups"] = int((details.agricultural_observations.gt(0) & details.nonagricultural_observations.gt(0)).sum())
    summary["clusters_without_agricultural_observations"] = int(details.agricultural_observations.eq(0).sum())
    summary["clusters_without_nonagricultural_observations"] = int(details.nonagricultural_observations.eq(0).sum())
    return details, summary

def self_test() -> None:
    frame = pd.DataFrame([{key: 0.0 for key in NUMERIC} for _ in range(8)])
    frame[FIRM] = ["000001", "000001", "000002", "000002", "000003", "000004", "000005", "000006"]
    frame["year"] = [2007, 2008, 2007, 2008, 2009, 2009, 2010, 2010]
    frame[AGRI] = [1, 1, 0, 0, 0, 1, 0, 1]
    frame[INDUSTRY], frame["province"] = "C13", "Province"
    frame[COUNT] = [2, 0, 1, 3, 0, 5, 2, 0]
    frame[LOG_COUNT] = np.log1p(frame[COUNT])
    frame[ENTRY] = frame[COUNT].gt(0).astype(int)
    frame.loc[0, LAGGED + [FI]] = np.nan
    frame.loc[2, LAGGED] = np.nan
    frame.loc[4, "roa"] = np.nan
    frame.loc[5, "soe"] = np.nan
    frame.loc[6, LAGGED[0]] = np.nan
    frame.loc[7, FI] = np.nan
    validate_panel(frame)
    masks, excluded = build_masks(frame)
    assert [int(m.sum()) for m in masks.values()] == [8, 8, 7, 6, 3, 2]
    assert int(excluded["Post-2007 GTOS-missing within static-complete sample"].sum()) == 2
    failures = first_failures(frame, masks)
    assert failures.loc[failures.group.eq("All"), "observations"].sum() == 8
    samples, comparisons = compare_samples(frame, masks["Common estimation sample"], excluded)
    assert samples.loc[samples.comparison.str.startswith("All excluded") & samples.group.eq("All"), "firms_in_both"].iloc[0] == 2
    entry = comparisons.loc[comparisons.comparison.str.startswith("All excluded") & comparisons.group.eq("All") & comparisons.metric.eq("Patent entry rate")].iloc[0]
    assert entry.retained_valid_n == 2 and entry.excluded_valid_n == 6
    assert standardized_difference(pd.Series([1.0, 1.0]), pd.Series([1.0, 1.0])) is None
    assert standardized_difference(pd.Series([1.0]), pd.Series([0.0, 2.0])) is None
    categories = categorical_retention(frame, masks["Common estimation sample"], excluded, masks["Static controls and fields complete"])
    missing = categories.loc[categories.group.eq("All") & categories.variable.eq("soe") & categories.category.eq("Missing")].iloc[0]
    assert missing.full_n == 1 and missing.all_excluded_n == 1
    details, cluster_summary = industry_clusters(frame, masks["Common estimation sample"])
    assert len(details) == 1 and cluster_summary["All"]["inverse_size_share_hhi"] == 1
    nonag = frame.loc[frame[AGRI].eq(0)].copy()
    nonag_masks, nonag_excluded = build_masks(nonag)
    empty_samples, empty_metrics = compare_samples(nonag, nonag_masks["Common estimation sample"], nonag_excluded)
    assert empty_samples.loc[empty_samples.group.eq("Agricultural"), "retained_n"].eq(0).all()
    assert empty_metrics.loc[empty_metrics.group.eq("Agricultural"), "retained_value"].isna().all()
    for broken in [pd.concat([frame, frame.iloc[[1]]], ignore_index=True), frame.assign(soe=2), frame.assign(ln_assets=np.inf)]:
        try:
            validate_panel(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("An invalid fixture was accepted.")
    print("Sample-selection synthetic checks passed: nesting, partition, missingness, group scope, overlapping firms, concentration, and invalid inputs.")
