from __future__ import annotations

import pandas as pd
from .structural_utils import BASE_FULL
from .structural_utils import GTOS_FULL
from .structural_utils import OUTCOME_MAP
from .structural_utils import PATENT_OUTCOMES
from .structural_utils import clustered_ols
from .structural_utils import common_sample

def build_common_support(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = common_sample(panel, PATENT_OUTCOMES)
    intensity = "industry_year_nonagri_patent_per_firm_lag1_log1p"
    diffusion = "industry_year_has_patent_share_nonagri_lag1"
    agri = df.loc[df["是否涉农"] == 1]
    nonagri = df.loc[df["是否涉农"] == 0]

    intensity_bounds = (max(agri[intensity].min(), nonagri[intensity].min()), min(agri[intensity].max(), nonagri[intensity].max()))
    diffusion_bounds = (max(agri[diffusion].min(), nonagri[diffusion].min()), min(agri[diffusion].max(), nonagri[diffusion].max()))
    support_mask = (
        df[intensity].between(*intensity_bounds)
        & df[diffusion].between(*diffusion_bounds)
    )
    support_df = df.loc[support_mask].copy()

    rows = [
        {
            "sample": "full_common_sample",
            "observations": int(len(df)),
            "agri_share": float(df["是否涉农"].mean()),
            "intensity_lower": intensity_bounds[0],
            "intensity_upper": intensity_bounds[1],
            "diffusion_lower": diffusion_bounds[0],
            "diffusion_upper": diffusion_bounds[1],
        },
        {
            "sample": "common_support_sample",
            "observations": int(len(support_df)),
            "agri_share": float(support_df["是否涉农"].mean()),
            "intensity_lower": intensity_bounds[0],
            "intensity_upper": intensity_bounds[1],
            "diffusion_lower": diffusion_bounds[0],
            "diffusion_upper": diffusion_bounds[1],
        },
    ]
    for outcome in PATENT_OUTCOMES:
        base_full = clustered_ols(f"{outcome} ~ {BASE_FULL}", df)
        gtos_full = clustered_ols(f"{outcome} ~ {GTOS_FULL}", df)
        base_support = clustered_ols(f"{outcome} ~ {BASE_FULL}", support_df)
        gtos_support = clustered_ols(f"{outcome} ~ {GTOS_FULL}", support_df)
        rows.extend(
            [
                {
                    "sample": "full_common_sample",
                    "outcome": OUTCOME_MAP[outcome],
                    "model": "base_controls",
                    "agri_coef": float(base_full.params["是否涉农"]),
                    "agri_se": float(base_full.bse["是否涉农"]),
                    "agri_pvalue": float(base_full.pvalues["是否涉农"]),
                    "observations": int(base_full.nobs),
                },
                {
                    "sample": "full_common_sample",
                    "outcome": OUTCOME_MAP[outcome],
                    "model": "plus_gtos",
                    "agri_coef": float(gtos_full.params["是否涉农"]),
                    "agri_se": float(gtos_full.bse["是否涉农"]),
                    "agri_pvalue": float(gtos_full.pvalues["是否涉农"]),
                    "observations": int(gtos_full.nobs),
                },
                {
                    "sample": "common_support_sample",
                    "outcome": OUTCOME_MAP[outcome],
                    "model": "base_controls",
                    "agri_coef": float(base_support.params["是否涉农"]),
                    "agri_se": float(base_support.bse["是否涉农"]),
                    "agri_pvalue": float(base_support.pvalues["是否涉农"]),
                    "observations": int(base_support.nobs),
                },
                {
                    "sample": "common_support_sample",
                    "outcome": OUTCOME_MAP[outcome],
                    "model": "plus_gtos",
                    "agri_coef": float(gtos_support.params["是否涉农"]),
                    "agri_se": float(gtos_support.bse["是否涉农"]),
                    "agri_pvalue": float(gtos_support.pvalues["是否涉农"]),
                    "observations": int(gtos_support.nobs),
                },
            ]
        )
    return pd.DataFrame(rows), support_df
