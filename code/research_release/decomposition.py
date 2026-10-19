from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from .structural_utils import CORE_FIRM_CONTROLS
from .structural_utils import OUTCOME_MAP

GROUP_LABELS = {
    "firm_controls": "Firm controls",
    "province_fe": "Province fixed effects",
    "gtos_intensity": "Lagged opportunity intensity",
    "gtos_diffusion": "Opportunity diffusion",
    "frontier": "Frontier-industry indicator",
}

def build_design_blocks(dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    year_fe = pd.get_dummies(dataframe["year"].astype(int), prefix="year", drop_first=True).astype(float)
    province_fe = pd.get_dummies(dataframe["province"].astype(str), prefix="prov", drop_first=True).astype(float)
    firm_controls = dataframe[CORE_FIRM_CONTROLS].astype(float).copy()
    gtos_intensity = dataframe[["industry_year_nonagri_patent_per_firm_lag1_log1p"]].astype(float).rename(
        columns={"industry_year_nonagri_patent_per_firm_lag1_log1p": "gtos_intensity"}
    )
    gtos_diffusion = dataframe[["industry_year_has_patent_share_nonagri_lag1"]].astype(float).rename(
        columns={"industry_year_has_patent_share_nonagri_lag1": "gtos_diffusion"}
    )
    frontier = dataframe[["frontier_industry_flag_baseline"]].astype(float).rename(
        columns={"frontier_industry_flag_baseline": "frontier"}
    )
    return {
        "year_fe": year_fe,
        "firm_controls": firm_controls,
        "province_fe": province_fe,
        "gtos_intensity": gtos_intensity,
        "gtos_diffusion": gtos_diffusion,
        "frontier": frontier,
    }

def matrix_with_const(*parts: pd.DataFrame) -> pd.DataFrame:
    pieces = [pd.DataFrame({"const": np.ones(len(parts[0]))}, index=parts[0].index)]
    for part in parts:
        if part is not None and not part.empty:
            pieces.append(part)
    return pd.concat(pieces, axis=1)

def fit_ols_matrix(y: pd.Series, x: pd.DataFrame, groups: pd.Series):
    return sm.OLS(y.astype(float), x.astype(float)).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups},
    )

def gelbach_for_outcome(dataframe: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = build_design_blocks(dataframe)
    y = dataframe[outcome].astype(float)
    d = dataframe["是否涉农"].astype(float).rename("agri")
    groups = dataframe["证券代码"]

    x_restricted = matrix_with_const(d.to_frame(), blocks["year_fe"])
    x_full = matrix_with_const(
        d.to_frame(),
        blocks["year_fe"],
        blocks["firm_controls"],
        blocks["province_fe"],
        blocks["gtos_intensity"],
        blocks["gtos_diffusion"],
        blocks["frontier"],
    )

    restricted = fit_ols_matrix(y, x_restricted, groups)
    full = fit_ols_matrix(y, x_full, groups)

    total_change = float(restricted.params["agri"] - full.params["agri"])

    w = matrix_with_const(d.to_frame(), blocks["year_fe"]).values
    z_all = pd.concat(
        [
            blocks["firm_controls"],
            blocks["province_fe"],
            blocks["gtos_intensity"],
            blocks["gtos_diffusion"],
            blocks["frontier"],
        ],
        axis=1,
    )
    b_aux = np.linalg.lstsq(w, z_all.values, rcond=None)[0]
    agri_aux = b_aux[1, :]
    delta = full.params.loc[z_all.columns].values

    contributions = {}
    offset = 0
    for group_name in ["firm_controls", "province_fe", "gtos_intensity", "gtos_diffusion", "frontier"]:
        columns = list(blocks[group_name].columns)
        width = len(columns)
        contributions[group_name] = float(np.dot(agri_aux[offset : offset + width], delta[offset : offset + width]))
        offset += width

    detail = pd.DataFrame(
        [
            {
                "outcome": OUTCOME_MAP[outcome],
                "group": GROUP_LABELS[group_name],
                "contribution": value,
                "contribution_share_of_change": value / total_change if total_change != 0 else np.nan,
            }
            for group_name, value in contributions.items()
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "outcome": OUTCOME_MAP[outcome],
                "base_agri_coef": float(restricted.params["agri"]),
                "base_agri_se": float(restricted.bse["agri"]),
                "expanded_agri_coef": float(full.params["agri"]),
                "expanded_agri_se": float(full.bse["agri"]),
                "total_coefficient_change": total_change,
                "explained_by_group_sum": detail["contribution"].sum(),
                "residual_rounding_gap": total_change - detail["contribution"].sum(),
                "observations": int(restricted.nobs),
            }
        ]
    )
    return summary, detail

def pooled_oaxaca_for_outcome(dataframe: pd.DataFrame, outcome: str) -> pd.DataFrame:
    blocks = build_design_blocks(dataframe)
    x_ref = pd.concat(
        [
            blocks["firm_controls"],
            blocks["province_fe"],
            blocks["year_fe"],
            blocks["gtos_intensity"],
            blocks["gtos_diffusion"],
            blocks["frontier"],
        ],
        axis=1,
    )
    x_ref = sm.add_constant(x_ref, has_constant="add")
    y = dataframe[outcome].astype(float)
    result = fit_ols_matrix(y, x_ref, dataframe["证券代码"])

    agri_mask = dataframe["是否涉农"] == 1
    nonagri_mask = dataframe["是否涉农"] == 0

    gap = float(y[agri_mask].mean() - y[nonagri_mask].mean())
    explained_rows = []
    explained_total = 0.0
    group_map = {
        "firm_controls": list(blocks["firm_controls"].columns),
        "province_fe": list(blocks["province_fe"].columns),
        "year_fe": list(blocks["year_fe"].columns),
        "gtos_intensity": list(blocks["gtos_intensity"].columns),
        "gtos_diffusion": list(blocks["gtos_diffusion"].columns),
        "frontier": list(blocks["frontier"].columns),
    }
    for group_name, columns in group_map.items():
        diff_means = x_ref.loc[agri_mask, columns].mean().values - x_ref.loc[nonagri_mask, columns].mean().values
        beta = result.params.loc[columns].values
        contribution = float(np.dot(diff_means, beta))
        explained_total += contribution
        explained_rows.append(
            {
                "outcome": OUTCOME_MAP[outcome],
                "component": GROUP_LABELS.get(group_name, group_name),
                "explained_contribution": contribution,
                "share_of_gap": contribution / gap if gap != 0 else np.nan,
            }
        )

    unexplained = gap - explained_total
    explained_rows.extend(
        [
            {
                "outcome": OUTCOME_MAP[outcome],
                "component": "Explained total",
                "explained_contribution": explained_total,
                "share_of_gap": explained_total / gap if gap != 0 else np.nan,
            },
            {
                "outcome": OUTCOME_MAP[outcome],
                "component": "Unexplained remainder",
                "explained_contribution": unexplained,
                "share_of_gap": unexplained / gap if gap != 0 else np.nan,
            },
        ]
    )
    detail = pd.DataFrame(explained_rows)
    detail["total_gap"] = gap
    detail["gtos_related_explained_share"] = np.where(
        detail["component"] == "Explained total",
        detail.loc[detail["component"].isin(["Lagged opportunity intensity", "Opportunity diffusion", "Frontier-industry indicator"]), "explained_contribution"].sum() / explained_total if explained_total != 0 else np.nan,
        np.nan,
    )
    return detail
