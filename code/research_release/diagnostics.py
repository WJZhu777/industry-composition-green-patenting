from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

OUTCOMES = {
    "Patent count": "green_patent_application_count",
    "Log(1 + patent count)": "log1p_green_patent_application_count",
    "Patent entry indicator": "has_green_patent_application",
}

REGIME_LABELS = {
    "low_opportunity": "Low-opportunity",
    "middle_opportunity": "Middle-opportunity",
    "high_frontier_opportunity": "High/frontier opportunity",
}

def build_core_gtos_regime(common: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    intensity = "industry_year_nonagri_patent_per_firm_lag1_log1p"
    diffusion = "industry_year_has_patent_share_nonagri_lag1"
    frontier = "frontier_industry_flag_baseline"
    cells = (
        common[
            [
                "industry_code_std",
                "year",
                intensity,
                diffusion,
                frontier,
            ]
        ]
        .drop_duplicates(["industry_code_std", "year"])
        .copy()
    )

    for source, target in [(intensity, "z_intensity"), (diffusion, "z_diffusion")]:
        values = cells[source].astype(float)
        std = values.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            raise ValueError(f"Cannot standardize {source}")
        cells[target] = (values - values.mean()) / std

    cells["core_gtos_score"] = cells[["z_intensity", "z_diffusion"]].mean(axis=1)
    cells["gtos_regime_core"] = pd.qcut(
        cells["core_gtos_score"],
        q=3,
        labels=[
            "low_opportunity",
            "middle_opportunity",
            "high_frontier_opportunity",
        ],
        duplicates="raise",
    ).astype(str)
    cells.loc[cells[frontier].eq(1), "gtos_regime_core"] = "high_frontier_opportunity"

    merged = common.merge(
        cells[
            [
                "industry_code_std",
                "year",
                "core_gtos_score",
                "gtos_regime_core",
            ]
        ],
        on=["industry_code_std", "year"],
        how="left",
        validate="many_to_one",
    )
    if merged["gtos_regime_core"].isna().any():
        raise AssertionError("Core GTOS regime is missing on the common estimation sample.")
    return merged, cells

def regime_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for regime in REGIME_LABELS:
        for group_value, group_label in [(1, "Agricultural firms"), (0, "Non-agricultural firms")]:
            group = dataframe.loc[dataframe["是否涉农"].eq(group_value)]
            count = int(group["gtos_regime_core"].eq(regime).sum())
            rows.append(
                {
                    "GTOS regime": REGIME_LABELS[regime],
                    "Group": group_label,
                    "Firm-year observations": count,
                    "Share within group": count / len(group),
                }
            )
    return pd.DataFrame(rows)

def regime_support(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    outcome = "green_patent_application_count"
    for regime in REGIME_LABELS:
        for group_value, group_label in [(0, "Non-agricultural firms"), (1, "Agricultural firms")]:
            subset = dataframe.loc[
                dataframe["gtos_regime_core"].eq(regime) & dataframe["是否涉农"].eq(group_value)
            ]
            rows.append(
                {
                    "GTOS regime": REGIME_LABELS[regime],
                    "Group": group_label,
                    "Firm-year observations": int(len(subset)),
                    "Firms": int(subset["证券代码"].nunique()),
                    "Industries": int(subset["industry_code_std"].nunique()),
                    "Mean patent count": float(subset[outcome].mean()),
                    "Median patent count": float(subset[outcome].median()),
                    "Positive-patent share": float((subset[outcome] > 0).mean()),
                    "Maximum patent count": float(subset[outcome].max()),
                }
            )
    return pd.DataFrame(rows)

def firm_cluster_bootstrap_gap(
    dataframe: pd.DataFrame, regime: str, repetitions: int = 2_000, seed: int = 20260726
) -> dict[str, object]:
    subset = dataframe.loc[dataframe["gtos_regime_core"].eq(regime)].copy()
    outcome = "green_patent_application_count"
    raw_gap = float(
        subset.loc[subset["是否涉农"].eq(1), outcome].mean()
        - subset.loc[subset["是否涉农"].eq(0), outcome].mean()
    )
    median_gap = float(
        subset.loc[subset["是否涉农"].eq(1), outcome].median()
        - subset.loc[subset["是否涉农"].eq(0), outcome].median()
    )

    firm_totals = (
        subset.groupby(["证券代码", "是否涉农"], as_index=False)[outcome]
        .agg(["sum", "count"])
        .reset_index()
    )
    agri = firm_totals.loc[firm_totals["是否涉农"].eq(1), ["sum", "count"]].to_numpy()
    nonagri = firm_totals.loc[firm_totals["是否涉农"].eq(0), ["sum", "count"]].to_numpy()
    if len(agri) == 0 or len(nonagri) == 0:
        raise ValueError(f"Insufficient firm support for {regime}")
    rng = np.random.default_rng(seed)
    boot = np.empty(repetitions)
    for index in range(repetitions):
        agri_draw = agri[rng.integers(0, len(agri), size=len(agri))]
        nonagri_draw = nonagri[rng.integers(0, len(nonagri), size=len(nonagri))]
        boot[index] = agri_draw[:, 0].sum() / agri_draw[:, 1].sum() - (
            nonagri_draw[:, 0].sum() / nonagri_draw[:, 1].sum()
        )
    return {
        "GTOS regime": REGIME_LABELS[regime],
        "Raw mean gap": raw_gap,
        "Median gap": median_gap,
        "Bootstrap repetitions": repetitions,
        "Bootstrap mean gap": float(boot.mean()),
        "Bootstrap standard error": float(boot.std(ddof=1)),
        "95% CI lower": float(np.quantile(boot, 0.025)),
        "95% CI upper": float(np.quantile(boot, 0.975)),
    }

def high_regime_tail_sensitivity(dataframe: pd.DataFrame) -> pd.DataFrame:
    outcome = "green_patent_application_count"
    subset = dataframe.loc[
        dataframe["gtos_regime_core"].eq("high_frontier_opportunity")
    ].copy()
    rows = []
    for quantile in [0.95, 0.975, 0.99]:
        cap = float(subset[outcome].quantile(quantile))
        capped = subset[outcome].clip(upper=cap)
        gap = float(
            capped.loc[subset["是否涉农"].eq(1)].mean()
            - capped.loc[subset["是否涉农"].eq(0)].mean()
        )
        rows.append(
            {
                "Specification": f"Patent count capped at p{quantile * 100:g}",
                "Agricultural firm-years": int(subset["是否涉农"].eq(1).sum()),
                "Non-agricultural firm-years": int(subset["是否涉农"].eq(0).sum()),
                "Adjusted mean gap": gap,
                "Patent-count cap": cap,
            }
        )
    return pd.DataFrame(rows)

def summarize_definition(
    dataframe: pd.DataFrame, mask: pd.Series, label: str
) -> dict[str, object]:
    subset = dataframe.loc[mask].copy()
    outcome = "green_patent_application_count"
    agri = subset.loc[subset["是否涉农"].eq(1), outcome]
    nonagri = subset.loc[subset["是否涉农"].eq(0), outcome]
    p95 = float(subset[outcome].quantile(0.95))
    p99 = float(subset[outcome].quantile(0.99))
    return {
        "High-opportunity definition": label,
        "Agricultural firm-years": int(len(agri)),
        "Non-agricultural firm-years": int(len(nonagri)),
        "Raw mean gap": float(agri.mean() - nonagri.mean()),
        "Median gap": float(agri.median() - nonagri.median()),
        "p95-capped mean gap": float(
            agri.clip(upper=p95).mean() - nonagri.clip(upper=p95).mean()
        ),
        "p99-capped mean gap": float(
            agri.clip(upper=p99).mean() - nonagri.clip(upper=p99).mean()
        ),
    }

def alternative_high_definitions(dataframe: pd.DataFrame) -> pd.DataFrame:
    intensity = "industry_year_nonagri_patent_per_firm_lag1_log1p"
    baseline = "baseline_nonagri_patent_per_firm"
    score = "core_gtos_score"
    rows = []

    for column, label, quantile in [
        (score, "Core GTOS score top tercile", 2 / 3),
        (intensity, "Lagged intensity top tercile", 2 / 3),
        (intensity, "Lagged intensity top quartile", 0.75),
        (baseline, "Baseline intensity top third", 2 / 3),
        (baseline, "Baseline intensity top quartile", 0.75),
        (baseline, "Baseline intensity top quintile", 0.80),
    ]:
        cell_values = (
            dataframe[["industry_code_std", "year", column]]
            .drop_duplicates(["industry_code_std", "year"])[column]
            .dropna()
        )
        threshold = float(cell_values.quantile(quantile))
        rows.append(
            summarize_definition(
                dataframe,
                dataframe[column].ge(threshold),
                label,
            )
        )
    return pd.DataFrame(rows)

def minimum_support_by_industry(dataframe: pd.DataFrame) -> pd.DataFrame:
    high = dataframe.loc[
        dataframe["gtos_regime_core"].eq("high_frontier_opportunity")
    ].copy()
    outcome = "green_patent_application_count"
    agri_by_industry = (
        high.loc[high["是否涉农"].eq(1)]
        .groupby("industry_code_std")
        .size()
    )
    rows = []
    for minimum in [1, 5, 10]:
        industries = agri_by_industry.loc[agri_by_industry.ge(minimum)].index
        subset = high.loc[high["industry_code_std"].isin(industries)]
        agri = subset.loc[subset["是否涉农"].eq(1), outcome]
        nonagri = subset.loc[subset["是否涉农"].eq(0), outcome]
        rows.append(
            {
                "Minimum agricultural firm-years per industry": minimum,
                "Industries retained": int(len(industries)),
                "Agricultural firm-years": int(len(agri)),
                "Non-agricultural firm-years": int(len(nonagri)),
                "Agricultural mean": float(agri.mean()),
                "Non-agricultural mean": float(nonagri.mean()),
                "Mean gap": float(agri.mean() - nonagri.mean()),
            }
        )
    return pd.DataFrame(rows)

def patent_concentration(dataframe: pd.DataFrame) -> pd.DataFrame:
    high = dataframe.loc[
        dataframe["gtos_regime_core"].eq("high_frontier_opportunity")
    ].copy()
    outcome = "green_patent_application_count"
    rows = []
    for group_value, group_label in [(0, "Non-agricultural firms"), (1, "Agricultural firms")]:
        totals = (
            high.loc[high["是否涉农"].eq(group_value)]
            .groupby("证券代码")[outcome]
            .sum()
            .sort_values(ascending=False)
        )
        total_patents = float(totals.sum())
        for top_n in [1, 3, 5, 10]:
            share = float(totals.head(top_n).sum() / total_patents) if total_patents else np.nan
            rows.append(
                {
                    "Group and concentration cutoff": f"{group_label} | top {top_n} firm"
                    + ("s" if top_n > 1 else ""),
                    "Total firms in group": int(len(totals)),
                    "Patent share": share,
                }
            )
    return pd.DataFrame(rows)

def covariance_for_scheme(result, dataframe: pd.DataFrame, scheme: str) -> tuple[np.ndarray, int | None]:
    firm = dataframe["证券代码"].astype("category").cat.codes.to_numpy()
    industry = dataframe["industry_code_std"].astype("category").cat.codes.to_numpy()
    if scheme == "Firm":
        return cov_cluster(result, firm), dataframe["证券代码"].nunique() - 1
    if scheme == "Detailed industry":
        return cov_cluster(result, industry), dataframe["industry_code_std"].nunique() - 1
    if scheme == "Firm and detailed industry":
        covariance, _, _ = cov_cluster_2groups(result, firm, industry)
        return covariance, None
    raise ValueError(scheme)

def robust_term_statistics(
    result, covariance: np.ndarray, term: str, degrees_of_freedom: int | None
) -> tuple[float, float, float]:
    index = list(result.params.index).index(term)
    estimate = float(result.params.iloc[index])
    standard_error = float(np.sqrt(covariance[index, index]))
    statistic = estimate / standard_error
    if degrees_of_freedom is None:
        pvalue = float(2 * stats.norm.sf(abs(statistic)))
    else:
        pvalue = float(2 * stats.t.sf(abs(statistic), df=degrees_of_freedom))
    return estimate, standard_error, pvalue

def joint_wald_pvalue(result, covariance: np.ndarray, terms: list[str]) -> float:
    indices = [list(result.params.index).index(term) for term in terms]
    estimates = result.params.iloc[indices].to_numpy()
    subcov = covariance[np.ix_(indices, indices)]
    statistic = float(estimates.T @ np.linalg.pinv(subcov) @ estimates)
    return float(stats.chi2.sf(statistic, df=len(indices)))

def clustering_sensitivity(dataframe: pd.DataFrame, utils) -> pd.DataFrame:
    rows = []
    schemes = ["Firm", "Detailed industry", "Firm and detailed industry"]
    for outcome_label, outcome in OUTCOMES.items():
        for model_label, rhs in [
            ("Control-adjusted model", utils.BASE_FULL),
            ("Full accounting model", utils.GTOS_FULL),
        ]:
            result = smf.ols(f"{outcome} ~ {rhs}", data=dataframe).fit()
            for scheme in schemes:
                covariance, degrees = covariance_for_scheme(result, dataframe, scheme)
                estimate, standard_error, pvalue = robust_term_statistics(
                    result, covariance, "是否涉农", degrees
                )
                joint_pvalue = np.nan
                if model_label == "Full accounting model":
                    joint_pvalue = joint_wald_pvalue(
                        result,
                        covariance,
                        utils.CORE_GTOS_VARS,
                    )
                rows.append(
                    {
                        "Outcome": outcome_label,
                        "Specification": model_label,
                        "Clustering scheme": scheme,
                        "Agricultural coefficient": estimate,
                        "Standard error": standard_error,
                        "p-value": pvalue,
                        "Joint GTOS-block p-value": joint_pvalue,
                        "Observations": int(result.nobs),
                        "Firm clusters": int(dataframe["证券代码"].nunique()),
                        "Industry clusters": int(dataframe["industry_code_std"].nunique()),
                    }
                )
    return pd.DataFrame(rows)

def post2015_clustering_sensitivity(dataframe: pd.DataFrame, utils) -> pd.DataFrame:
    df = dataframe.copy()
    df["post_2015"] = (df["year"] >= 2015).astype(int)
    controls = " + ".join(utils.CORE_FIRM_CONTROLS)
    base_rhs = f"是否涉农 + 是否涉农:post_2015 + C(year) + {controls} + C(province)"
    full_rhs = base_rhs + " + " + " + ".join(utils.CORE_GTOS_VARS)
    rows = []
    for outcome_label, outcome in OUTCOMES.items():
        for model_label, rhs in [
            ("Control-adjusted model", base_rhs),
            ("Full accounting model", full_rhs),
        ]:
            result = smf.ols(f"{outcome} ~ {rhs}", data=df).fit()
            for scheme in ["Firm", "Detailed industry", "Firm and detailed industry"]:
                covariance, degrees = covariance_for_scheme(result, df, scheme)
                estimate, standard_error, pvalue = robust_term_statistics(
                    result, covariance, "是否涉农:post_2015", degrees
                )
                rows.append(
                    {
                        "Outcome": outcome_label,
                        "Specification": model_label,
                        "Clustering scheme": scheme,
                        "Agricultural firm x Post-2015 coefficient": estimate,
                        "Standard error": standard_error,
                        "p-value": pvalue,
                        "Observations": int(result.nobs),
                    }
                )
    return pd.DataFrame(rows)

def frontier_2013_sensitivity(dataframe: pd.DataFrame, utils) -> pd.DataFrame:
    subset = dataframe.loc[dataframe["year"].ge(2013)].copy()
    rows = []
    for outcome_label, outcome in OUTCOMES.items():
        for model_label, rhs in [
            ("Control-adjusted model", utils.BASE_FULL),
            ("Full accounting model", utils.GTOS_FULL),
        ]:
            result = utils.clustered_ols(f"{outcome} ~ {rhs}", subset)
            rows.append(
                {
                    "Outcome": outcome_label,
                    "Specification": model_label,
                    "Agricultural coefficient": float(result.params["是否涉农"]),
                    "Firm-clustered standard error": float(result.bse["是否涉农"]),
                    "p-value": float(result.pvalues["是否涉农"]),
                    "Observations": int(result.nobs),
                }
            )
    return pd.DataFrame(rows)

def distribution_variance(panel: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    outcome = "green_patent_application_count"
    rows = []
    for sample_label, sample in [
        ("Full panel, 2007-2024", panel),
        ("Common estimation sample", common),
    ]:
        for group_value, group_label in [(None, "All firms"), (1, "Agricultural firms"), (0, "Non-agricultural firms")]:
            subset = sample if group_value is None else sample.loc[sample["是否涉农"].eq(group_value)]
            rows.append(
                {
                    "Sample": sample_label,
                    "Group": group_label,
                    "Variance": float(subset[outcome].var(ddof=1)),
                }
            )
    return pd.DataFrame(rows)
