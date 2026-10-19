from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = {
    "Patent count": "green_patent_application_count",
    "Log(1 + patent count)": "log1p_green_patent_application_count",
    "Patent entry indicator": "has_green_patent_application",
}

AGRI = "是否涉农"

FIRM = "证券代码"

INTENSITY = "industry_year_nonagri_patent_per_firm_lag1_log1p"

DIFFUSION = "industry_year_has_patent_share_nonagri_lag1"

FRONTIER = "frontier_industry_flag_baseline"

def common_sample(panel: pd.DataFrame, utils) -> pd.DataFrame:
    return utils.common_sample(panel, list(OUTCOMES.values()))

def nested_coefficients(
    panel: pd.DataFrame,
    utils,
    construction: str,
) -> pd.DataFrame:
    dataframe = common_sample(panel, utils)
    rows: list[dict[str, object]] = []
    formulas = {
        "Sparse restricted model": utils.BASE_RESTRICTED,
        "Control-adjusted model": utils.BASE_FULL,
        "Full accounting model": utils.GTOS_FULL,
    }
    for outcome_label, outcome in OUTCOMES.items():
        for model_label, rhs in formulas.items():
            result = utils.clustered_ols(f"{outcome} ~ {rhs}", dataframe)
            rows.append(
                {
                    "Construction": construction,
                    "Outcome": outcome_label,
                    "Model": model_label,
                    "Agricultural coefficient": float(result.params[AGRI]),
                    "Standard error": float(result.bse[AGRI]),
                    "p-value": float(result.pvalues[AGRI]),
                    "Observations": int(result.nobs),
                    "R-squared": float(result.rsquared),
                }
            )
    return pd.DataFrame(rows)

def exposure_tables(
    panel: pd.DataFrame,
    industry_year: pd.DataFrame,
    common: pd.DataFrame,
    construction: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    industry = (
        industry_year[
            [
                "industry_code_std",
                "baseline_nonagri_patent_per_firm",
                FRONTIER,
            ]
        ]
        .drop_duplicates("industry_code_std")
        .dropna(subset=["baseline_nonagri_patent_per_firm"])
        .copy()
    )
    industry["Baseline opportunity quartile"] = pd.qcut(
        industry["baseline_nonagri_patent_per_firm"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="raise",
    )
    exposure = panel.merge(
        industry[
            [
                "industry_code_std",
                "Baseline opportunity quartile",
                FRONTIER,
            ]
        ],
        on="industry_code_std",
        how="left",
        suffixes=("", "_corrected"),
        validate="m:1",
    )
    frontier_column = (
        f"{FRONTIER}_corrected"
        if f"{FRONTIER}_corrected" in exposure.columns
        else FRONTIER
    )
    rows: list[dict[str, object]] = []
    for group_value, group_label in [
        (1, "Agricultural firms"),
        (0, "Non-agricultural firms"),
    ]:
        quartile_sample = exposure.loc[
            exposure[AGRI].eq(group_value)
            & exposure["Baseline opportunity quartile"].notna()
        ]
        shares = quartile_sample["Baseline opportunity quartile"].value_counts(
            normalize=True
        )
        for quartile in ["Q1", "Q2", "Q3", "Q4"]:
            rows.append(
                {
                    "Construction": construction,
                    "Exposure category": quartile,
                    "Group": group_label,
                    "Firm-year share": float(shares.get(quartile, 0.0)),
                    "Firm-year observations": int(len(quartile_sample)),
                }
            )
        frontier_sample = exposure.loc[
            exposure[AGRI].eq(group_value) & exposure[frontier_column].notna()
        ]
        rows.append(
            {
                "Construction": construction,
                "Exposure category": "Frontier-industry exposure",
                "Group": group_label,
                "Firm-year share": float(frontier_sample[frontier_column].mean()),
                "Firm-year observations": int(len(frontier_sample)),
            }
        )

    cells = (
        common[
            [
                "industry_code_std",
                "year",
                INTENSITY,
                DIFFUSION,
                FRONTIER,
            ]
        ]
        .drop_duplicates(["industry_code_std", "year"])
        .copy()
    )
    for source, target in [
        (INTENSITY, "z_intensity"),
        (DIFFUSION, "z_diffusion"),
    ]:
        values = cells[source].astype(float)
        cells[target] = (values - values.mean()) / values.std(ddof=0)
    cells["GTOS score"] = cells[["z_intensity", "z_diffusion"]].mean(axis=1)
    cells["GTOS regime"] = pd.qcut(
        cells["GTOS score"],
        q=3,
        labels=[
            "Low opportunity",
            "Middle opportunity",
            "High/frontier opportunity",
        ],
        duplicates="raise",
    ).astype(str)
    cells.loc[cells[FRONTIER].eq(1), "GTOS regime"] = "High/frontier opportunity"
    regime_panel = common.merge(
        cells[["industry_code_std", "year", "GTOS score", "GTOS regime"]],
        on=["industry_code_std", "year"],
        how="left",
        validate="m:1",
    )
    regime_rows: list[dict[str, object]] = []
    for group_value, group_label in [
        (1, "Agricultural firms"),
        (0, "Non-agricultural firms"),
    ]:
        subset = regime_panel.loc[regime_panel[AGRI].eq(group_value)]
        shares = subset["GTOS regime"].value_counts(normalize=True)
        for regime in [
            "Low opportunity",
            "Middle opportunity",
            "High/frontier opportunity",
        ]:
            outcome = subset.loc[
                subset["GTOS regime"].eq(regime),
                "green_patent_application_count",
            ]
            regime_rows.append(
                {
                    "Construction": construction,
                    "GTOS regime": regime,
                    "Group": group_label,
                    "Firm-year observations": int(len(outcome)),
                    "Firm-year share": float(shares.get(regime, 0.0)),
                    "Mean patent count": float(outcome.mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(regime_rows), regime_panel

def decomposition_tables(
    common: pd.DataFrame,
    decomposition,
    construction: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gelbach_summaries = []
    gelbach_details = []
    oaxaca_details = []
    for outcome_label, outcome in OUTCOMES.items():
        summary, detail = decomposition.gelbach_for_outcome(common, outcome)
        summary.insert(0, "Construction", construction)
        detail.insert(0, "Construction", construction)
        gelbach_summaries.append(summary)
        gelbach_details.append(detail)
        oaxaca = decomposition.pooled_oaxaca_for_outcome(common, outcome)
        oaxaca.insert(0, "Construction", construction)
        oaxaca_details.append(oaxaca)
    return (
        pd.concat(gelbach_summaries, ignore_index=True),
        pd.concat(gelbach_details, ignore_index=True),
        pd.concat(oaxaca_details, ignore_index=True),
    )

def support_summary(panel: pd.DataFrame, support_models, construction: str):
    raw, support_panel = support_models.build_common_support(panel)
    raw.insert(0, "Construction", construction)
    return raw, support_panel

def base_design(dataframe: pd.DataFrame, utils) -> np.ndarray:
    year = pd.get_dummies(dataframe["year"].astype(int), drop_first=True).astype(float)
    province = pd.get_dummies(
        dataframe["province"].astype(str),
        drop_first=True,
    ).astype(float)
    controls = dataframe[utils.CORE_FIRM_CONTROLS].astype(float)
    return np.column_stack(
        [np.ones(len(dataframe)), year.values, controls.values, province.values]
    )

def q_matrix(matrix: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return q

def residualize(q: np.ndarray, values: np.ndarray) -> np.ndarray:
    return values - q @ (q.T @ values)

def partial_out(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    if covariates.shape[1] == 0:
        return values
    coefficients = np.linalg.lstsq(covariates, values, rcond=None)[0]
    return values - covariates @ coefficients

def coefficient(y: np.ndarray, d: np.ndarray) -> float:
    return float((d @ y) / (d @ d))

def reassignment_checks(
    common: pd.DataFrame,
    utils,
    firm_draws: int = 500,
    cell_draws: int = 500,
    placebo_draws: int = 200,
) -> pd.DataFrame:
    dataframe = common.reset_index(drop=True)
    base_x = base_design(dataframe, utils)
    q_base = q_matrix(base_x)
    bundle = dataframe[[INTENSITY, DIFFUSION, FRONTIER]].astype(float).to_numpy()
    bundle_residual = residualize(q_base, bundle)
    d_true = dataframe[AGRI].astype(float).to_numpy()
    d_base_true = residualize(q_base, d_true)
    d_full_true = partial_out(d_base_true, bundle_residual)

    outcome_cache: dict[str, dict[str, np.ndarray | float]] = {}
    for label, column in OUTCOMES.items():
        y = dataframe[column].astype(float).to_numpy()
        y_base = residualize(q_base, y)
        y_full = partial_out(y_base, bundle_residual)
        observed = coefficient(y_base, d_base_true) - coefficient(
            y_full,
            d_full_true,
        )
        outcome_cache[label] = {
            "y": y,
            "y_base": y_base,
            "observed": observed,
        }

    draws: dict[tuple[str, str], list[float]] = {}
    for method in [
        "Firm-level ALF-label permutation",
        "Within-industry-year ALF reassignment",
        "Within-year joint GTOS-bundle placebo",
    ]:
        for label in OUTCOMES:
            draws[(method, label)] = []

    firm_labels = dataframe.groupby(FIRM, sort=False)[AGRI].first().astype(float)
    firm_codes = pd.Categorical(dataframe[FIRM], categories=firm_labels.index).codes
    rng_firm = np.random.default_rng(20260727)
    for _ in range(firm_draws):
        d = rng_firm.permutation(firm_labels.to_numpy())[firm_codes]
        d_base = residualize(q_base, d)
        d_full = partial_out(d_base, bundle_residual)
        for label, cached in outcome_cache.items():
            draws[("Firm-level ALF-label permutation", label)].append(
                coefficient(cached["y_base"], d_base)
                - coefficient(
                    partial_out(cached["y_base"], bundle_residual),
                    d_full,
                )
            )

    cell_indices = dataframe.groupby(
        ["industry_code_std", "year"],
        sort=False,
    ).indices
    rng_cell = np.random.default_rng(20260729)
    d_original = dataframe[AGRI].astype(float).to_numpy()
    for _ in range(cell_draws):
        d = d_original.copy()
        for indices in cell_indices.values():
            indices = np.asarray(indices)
            d[indices] = rng_cell.permutation(d[indices])
        d_base = residualize(q_base, d)
        d_full = partial_out(d_base, bundle_residual)
        for label, cached in outcome_cache.items():
            draws[("Within-industry-year ALF reassignment", label)].append(
                coefficient(cached["y_base"], d_base)
                - coefficient(
                    partial_out(cached["y_base"], bundle_residual),
                    d_full,
                )
            )

    cells = (
        dataframe[["industry_code_std", "year", INTENSITY, DIFFUSION, FRONTIER]]
        .drop_duplicates(["industry_code_std", "year"])
        .reset_index(drop=True)
    )
    cell_key = pd.MultiIndex.from_frame(cells[["industry_code_std", "year"]])
    row_key = pd.MultiIndex.from_frame(dataframe[["industry_code_std", "year"]])
    cell_position = cell_key.get_indexer(row_key)
    if np.any(cell_position < 0):
        raise ValueError("Failed to map firm-years to GTOS cells.")
    original_cell_bundle = cells[[INTENSITY, DIFFUSION, FRONTIER]].to_numpy()
    year_groups = cells.groupby("year", sort=False).indices
    rng_placebo = np.random.default_rng(20260728)
    for _ in range(placebo_draws):
        placebo = original_cell_bundle.copy()
        for indices in year_groups.values():
            indices = np.asarray(indices)
            placebo[indices, :] = original_cell_bundle[
                rng_placebo.permutation(indices),
                :,
            ]
        placebo_residual = residualize(q_base, placebo[cell_position, :])
        d_full = partial_out(d_base_true, placebo_residual)
        for label, cached in outcome_cache.items():
            y_full = partial_out(cached["y_base"], placebo_residual)
            draws[("Within-year joint GTOS-bundle placebo", label)].append(
                coefficient(cached["y_base"], d_base_true)
                - coefficient(y_full, d_full)
            )

    rows = []
    for (method, label), values in draws.items():
        array = np.asarray(values)
        observed = float(outcome_cache[label]["observed"])
        exceedances = int(np.sum(np.abs(array) >= abs(observed)))
        rows.append(
            {
                "Outcome": label,
                "Diagnostic": method,
                "Observed control-adjusted-minus-full difference": observed,
                "Reassignment mean": float(array.mean()),
                "Reassignment standard deviation": float(array.std(ddof=1)),
                "Exceedances": exceedances,
                "Replications": int(len(array)),
                "Plus-one exceedance frequency": (exceedances + 1)
                / (len(array) + 1),
            }
        )
    return pd.DataFrame(rows)

def leave_one_out_check(
    panel: pd.DataFrame,
    utils,
) -> pd.DataFrame:
    enhanced = utils.build_enhanced_panel(panel)
    required = [
        AGRI,
        "year",
        "province",
        FIRM,
        *utils.CORE_FIRM_CONTROLS,
        *OUTCOMES.values(),
        "gtos_intensity_lag1_loo",
        "gtos_diffusion_lag1_loo",
        FRONTIER,
    ]
    dataframe = enhanced.dropna(subset=required).copy()
    rhs = (
        f"{AGRI} + C(year) + "
        + " + ".join(utils.CORE_FIRM_CONTROLS)
        + " + C(province) + gtos_intensity_lag1_loo"
        + " + gtos_diffusion_lag1_loo"
        + f" + {FRONTIER}"
    )
    rows = []
    for label, outcome in OUTCOMES.items():
        result = utils.clustered_ols(f"{outcome} ~ {rhs}", dataframe)
        rows.append(
            {
                "Outcome": label,
                "Agricultural coefficient": float(result.params[AGRI]),
                "Standard error": float(result.bse[AGRI]),
                "p-value": float(result.pvalues[AGRI]),
                "Observations": int(result.nobs),
            }
        )
    return pd.DataFrame(rows)
