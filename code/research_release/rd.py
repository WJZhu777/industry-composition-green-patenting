from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
import patsy
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from . import sample

FIRM, AGRI, INDUSTRY = sample.FIRM, sample.AGRI, sample.INDUSTRY

RD_FIELDS = ["RDPerson", "RDPersonRatio", "RDSpendSum", "RDSpendSumRatio", "RDExpenses", "RDInvest", "RDInvestRatio", "RDInvestNetprofitRatio"]

CONTROLS = [*sample.FINANCIAL, "soe", "list_age"]

BASE = AGRI + " + " + " + ".join(CONTROLS) + " + C(year) + C(province)"

PATENT = [*sample.LAGGED, sample.FI]

RD_MEAN = "lag_rd_intensity_mean_pp"

RD_MEDIAN = "lag_rd_intensity_median_pp"

PERSON = "lag_rd_personnel_share_mean_pp"

OUTCOMES = {"Patent count": sample.COUNT, "Log(1 + patent count)": sample.LOG_COUNT, "Patent entry indicator": sample.ENTRY}

MIN_DISCLOSERS = 3

PERSON_FIRST_YEAR = 2016

def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", force_ascii=False, double_precision=15))

def fingerprint(frame: pd.DataFrame) -> str:
    keys = frame[[FIRM, "year"]].sort_values([FIRM, "year"])
    text = "\n".join(f"{firm}|{int(year)}" for firm, year in keys.itertuples(index=False, name=None))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def strict_annual(raw: pd.DataFrame) -> pd.DataFrame:
    needed = {"Symbol", "EndDate", "Source", "StateTypeCode", "Currency", *RD_FIELDS}
    if not needed.issubset(raw.columns):
        raise ValueError(f"R&D fields absent: {sorted(needed - set(raw.columns))}")
    frame = raw.copy()
    date = pd.to_datetime(frame["EndDate"], errors="raise")
    frame = frame.loc[frame.Source.eq(0) & frame.StateTypeCode.eq(1) & date.dt.month.eq(12) & date.dt.day.eq(31)].copy()
    codes = frame.Symbol.astype("string")
    if not codes.str.fullmatch(r"\d{1,6}").all():
        raise ValueError("Invalid stock code in strict annual R&D records.")
    frame[FIRM] = codes.str.zfill(6)
    frame["year"] = pd.to_datetime(frame.EndDate).dt.year
    if frame.duplicated([FIRM, "year"]).any():
        raise ValueError("Ambiguous strict annual R&D keys; do not choose a record by completeness.")
    for field in RD_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
        if np.isinf(frame[field].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite raw R&D value: {field}")
    return frame

def construct_environment(panel: pd.DataFrame, annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fields = [FIRM, "year", *RD_FIELDS]
    source = panel[[FIRM, "year", AGRI, INDUSTRY]].merge(annual[fields].assign(annual_record=1),
                                                      on=[FIRM, "year"], how="left", validate="one_to_one")
    # The benchmark uses the main panel's non-agricultural firms, not its control-complete subset.
    valid_code = source[INDUSTRY].str.fullmatch(r"[A-Z]\d{2}")
    benchmark = source.loc[source[AGRI].eq(0) & valid_code].copy()
    benchmark["rd_valid_pp"] = benchmark.RDSpendSumRatio.where(benchmark.RDSpendSumRatio.ge(0))
    benchmark["person_valid_pp"] = benchmark.RDPersonRatio.where(benchmark.RDPersonRatio.between(0, 100))
    cells = benchmark.groupby([INDUSTRY, "year"]).agg(
        nonag_firms=(FIRM, "size"), annual_records=("annual_record", "count"),
        rd_disclosers=("rd_valid_pp", "count"), rd_mean_pp=("rd_valid_pp", "mean"),
        rd_median_pp=("rd_valid_pp", "median"), person_disclosers=("person_valid_pp", "count"),
        person_mean_pp=("person_valid_pp", "mean"),
    ).reset_index()
    for variable in ["rd_mean_pp", "rd_median_pp"]:
        cells.loc[cells.rd_disclosers.lt(MIN_DISCLOSERS), variable] = np.nan
    cells.loc[cells.person_disclosers.lt(MIN_DISCLOSERS), "person_mean_pp"] = np.nan
    cells["rd_disclosure_share"] = cells.rd_disclosers / cells.nonag_firms
    cells["person_disclosure_share"] = cells.person_disclosers / cells.nonag_firms
    cells = cells.rename(columns={"year": "source_year"})
    lagged = cells.rename(columns={
        "nonag_firms": "lag_nonag_firms", "annual_records": "lag_annual_records",
        "rd_disclosers": "lag_rd_disclosers", "rd_mean_pp": RD_MEAN, "rd_median_pp": RD_MEDIAN,
        "person_disclosers": "lag_person_disclosers", "person_mean_pp": PERSON,
        "rd_disclosure_share": "lag_rd_disclosure_share", "person_disclosure_share": "lag_person_disclosure_share",
    }).copy()
    lagged["year"] = lagged.source_year + 1
    merged = panel.merge(lagged, on=[INDUSTRY, "year"], how="left", validate="many_to_one")
    known = merged.source_year.notna()
    if not merged.loc[known, "year"].eq(merged.loc[known, "source_year"] + 1).all():
        raise AssertionError("Exposure is not from the preceding calendar year.")
    if not merged.loc[merged[RD_MEAN].notna(), "lag_rd_disclosers"].ge(MIN_DISCLOSERS).all():
        raise AssertionError("An R&D exposure lacks sufficient valid disclosers.")
    matched = source.annual_record.notna()
    ratio_comparison = panel[[FIRM, "year", "revenue_used_y"]].merge(annual[[FIRM, "year", "RDSpendSum", "RDSpendSumRatio"]],
                                                                    on=[FIRM, "year"], validate="one_to_one")
    pair = ratio_comparison.loc[ratio_comparison.revenue_used_y.gt(0) & ratio_comparison.RDSpendSumRatio.ge(0)
                                & ratio_comparison.RDSpendSum.notna()].copy()
    pair["absolute_difference_pp"] = abs(pair.RDSpendSumRatio - 100 * pair.RDSpendSum / pair.revenue_used_y)
    audit = {"matched_annual_panel_records": int(matched.sum()),
             "annual_records_outside_main_panel": len(annual) - int(matched.sum()),
             "invalid_industry_panel_records_excluded_from_benchmark": int((~valid_code).sum()),
             "benchmark_negative_rd_ratios_excluded": int(benchmark.RDSpendSumRatio.lt(0).sum()),
             "benchmark_person_ratios_outside_0_100_excluded": int((benchmark.RDPersonRatio.notna() & ~benchmark.RDPersonRatio.between(0, 100)).sum()),
             "benchmark_positive_rd_ratios_above_100_retained": int(benchmark.RDSpendSumRatio.gt(100).sum()),
             "direct_ratio_vs_separate_positive_revenue_check": {
                 "paired_records": len(pair), "median_absolute_difference_pp": float(pair.absolute_difference_pp.median()),
                 "share_within_0_1_pp": float(pair.absolute_difference_pp.le(.1).mean()),
                 "maximum_absolute_difference_pp": float(pair.absolute_difference_pp.max()),
                 "note": "A diagnostic only; direct ratios are not overwritten using a different-source revenue field."},
             "yearly_panel_matching": records(source.groupby("year").agg(panel_observations=(FIRM, "size"),
                 annual_records=("annual_record", "count"), direct_rd_ratios=("RDSpendSumRatio", "count"),
                 personnel_shares=("RDPersonRatio", "count")).reset_index())}
    return merged, cells, audit

def samples(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    masks, _ = sample.build_masks(merged)
    common = masks["Common estimation sample"]
    primary = common & merged[RD_MEAN].notna()
    personnel = primary & merged[PERSON].notna() & merged.year.ge(PERSON_FIRST_YEAR)
    selected = {"Original common sample": merged.loc[common].copy(),
                "R&D primary matched sample": merged.loc[primary].copy(),
                "Personnel supplement matched sample, 2016-2024": merged.loc[personnel].copy(),
                "Common sample excluded from R&D primary": merged.loc[common & ~primary].copy()}
    if len(selected["Original common sample"]) != 47773 or len(selected["R&D primary matched sample"]) != 42261:
        raise ValueError("Sample counts differ from the audited setup; stop rather than adjust the design.")
    for name, frame in list(selected.items())[1:3]:
        if frame[[*CONTROLS, *PATENT, *OUTCOMES.values(), RD_MEAN, RD_MEDIAN]].isna().any().any():
            raise ValueError(f"Incomplete model fields: {name}")
    return selected

def sample_details(selected: dict[str, pd.DataFrame]) -> dict:
    overview, yearly = [], []
    for name, frame in selected.items():
        for group, value in sample.GROUPS.items():
            part = frame.loc[sample.group_mask(frame, value)]
            overview.append({"sample": name, "group": group, "observations": len(part), "firms": part[FIRM].nunique(),
                             "industries": part[INDUSTRY].nunique(), "first_year": part.year.min(), "last_year": part.year.max(),
                             "patent_count_mean": part[sample.COUNT].mean(), "entry_rate": part[sample.ENTRY].mean(),
                             "ln_assets_mean": part.ln_assets.mean(), "state_ownership_share": part.soe.mean()})
        year = frame.groupby("year").agg(observations=(FIRM, "size"), agricultural_observations=(AGRI, "sum")).reset_index()
        year.insert(0, "sample", name)
        yearly.extend(records(year))
    return {"overview": records(pd.DataFrame(overview)), "yearly": yearly,
            "key_sha256": {name: fingerprint(frame) for name, frame in selected.items()}}

def estimate_design(frame: pd.DataFrame, rhs: str, additions: list[str]) -> tuple[list[dict], dict]:
    design = patsy.dmatrix(rhs, frame, return_type="dataframe", NA_action="raise")
    matrix = design.to_numpy(dtype=float)
    rank = np.linalg.matrix_rank(matrix)
    if rank != matrix.shape[1]:
        raise ValueError("Non-full-rank design: report the problem instead of silently changing controls.")
    if len(design) != len(frame) or not design.index.equals(frame.index):
        raise AssertionError("Design construction changed the fixed sample.")
    rows = []
    for outcome, field in OUTCOMES.items():
        result = sm.OLS(frame[field], design).fit(cov_type="cluster", cov_kwds={"groups": frame[FIRM], "use_correction": True}, use_t=False)
        industry_cov = cov_cluster(result, frame[INDUSTRY], use_correction=True)
        scaled_score = np.linalg.norm(matrix.T @ result.resid.to_numpy(), ord=np.inf) / max(1., np.linalg.norm(matrix) * np.linalg.norm(frame[field]))
        if scaled_score > 1e-9:
            raise AssertionError(f"OLS residual orthogonality failed: {scaled_score}")
        for term in [AGRI, *additions]:
            j = design.columns.get_loc(term)
            estimate = float(result.params[term])
            firm_se, industry_se = float(result.bse[term]), float(np.sqrt(industry_cov[j, j]))
            if not np.isfinite([estimate, firm_se, industry_se]).all() or min(firm_se, industry_se) <= 0:
                raise ValueError("Unavailable or invalid coefficient covariance.")
            rows.append({"outcome": outcome, "term": term, "coefficient": estimate,
                         "firm_cluster_se": firm_se, "firm_cluster_p_normal": float(result.pvalues[term]),
                         "firm_cluster_ci95_low": estimate - norm.ppf(.975) * firm_se,
                         "firm_cluster_ci95_high": estimate + norm.ppf(.975) * firm_se,
                         "industry_cluster_se": industry_se,
                         "industry_cluster_p_normal": float(2 * norm.sf(abs(estimate / industry_se))),
                         "industry_cluster_ci95_low": estimate - norm.ppf(.975) * industry_se,
                         "industry_cluster_ci95_high": estimate + norm.ppf(.975) * industry_se,
                         "observations": int(result.nobs), "firm_clusters": frame[FIRM].nunique(),
                         "industry_clusters": frame[INDUSTRY].nunique(), "r_squared": float(result.rsquared)})
    return rows, {"rhs": rhs, "design_columns": list(design.columns), "columns": len(design.columns), "rank": int(rank)}

def compare_models(selected: dict[str, pd.DataFrame]) -> dict:
    primary = {"Control-adjusted": [], "Patent GTOS": PATENT,
               "Non-patent R&D mean": [RD_MEAN], "Non-patent R&D median": [RD_MEDIAN]}
    supplemental = {"Control-adjusted": [], "Patent GTOS": PATENT,
                    "Non-patent R&D mean plus personnel share": [RD_MEAN, PERSON]}
    all_rows, designs = [], []
    for name, specifications in [("R&D primary matched sample", primary),
                                  ("Personnel supplement matched sample, 2016-2024", supplemental)]:
        frame = selected[name]
        for label, additions in specifications.items():
            rhs = BASE + (" + " + " + ".join(additions) if additions else "")
            print(f"Fitting {name}: {label}; N={len(frame)}", flush=True)
            rows, design = estimate_design(frame, rhs, additions)
            for row in rows:
                row.update(sample=name, model=label, sample_key_sha256=fingerprint(frame))
            all_rows.extend(rows)
            designs.append({"sample": name, "model": label, **design})
    coefficients = pd.DataFrame(all_rows)
    agricultural = coefficients.loc[coefficients.term.eq(AGRI)].copy()
    movement = []
    for (name, outcome), frame in agricultural.groupby(["sample", "outcome"], sort=False):
        control = float(frame.loc[frame.model.eq("Control-adjusted"), "coefficient"].iloc[0])
        for row in frame.loc[~frame.model.eq("Control-adjusted")].to_dict("records"):
            movement.append({"sample": name, "outcome": outcome, "model": row["model"],
                             "control_coefficient": control, "augmented_coefficient": row["coefficient"],
                             "control_minus_augmented": control - row["coefficient"]})
    return {"coefficients": records(coefficients), "agricultural_coefficients": records(agricultural),
            "control_minus_augmented_movements": records(pd.DataFrame(movement)), "designs": designs,
            "inference": "OLS with intercept, six firm controls, year and province indicators; no industry indicators. Cluster covariances use finite-sample correction and normal-reference p-values/intervals, conditional on the constructed exposures. Movement differences are point estimates, not independently tested effects."}

def self_test() -> None:
    raw = pd.DataFrame({"Symbol": ["1", "2", "3", "4", "5", "6"], "EndDate": ["2010-12-31"] * 6,
                        "Source": [0, 0, 0, 0, 1, 0], "StateTypeCode": [1, 1, 1, 1, 1, 2], "Currency": ["CNY"] * 6})
    for field in RD_FIELDS:
        raw[field] = np.nan
    raw["RDSpendSumRatio"] = [0, 0, 0, 999, 1000, 1000]
    raw["RDPersonRatio"] = [1, 2, np.nan, 100, 100, 100]
    annual = strict_annual(raw)
    assert len(annual) == 4
    panel = pd.DataFrame({FIRM: ["000001", "000002", "000003", "000004", "000001", "000001", "000001"],
                          "year": [2010, 2010, 2010, 2010, 2011, 2012, 2013],
                          AGRI: [0, 0, 0, 1, 0, 0, 0], INDUSTRY: ["C01"] * 7})
    # Positive reported ratios above 100 remain valid; negative ratios are ineligible for intensity.
    annual["RDSpendSum"] = 0
    panel["revenue_used_y"] = 1
    merged, cells, _ = construct_environment(panel, annual)
    assert merged.loc[merged.year.eq(2011), RD_MEAN].iloc[0] == 0
    assert merged.loc[merged.year.isin([2012, 2013]), RD_MEAN].isna().all()
    assert cells.loc[cells.source_year.eq(2010), "rd_disclosers"].iloc[0] == 3
    assert cells.loc[cells.source_year.eq(2010), "person_mean_pp"].isna().all()
    annual.loc[annual[FIRM].eq("000001"), "RDSpendSumRatio"] = 150
    _, cells, _ = construct_environment(panel, annual)
    assert cells.loc[cells.source_year.eq(2010), "rd_mean_pp"].iloc[0] == 50
    annual.loc[annual[FIRM].eq("000001"), "RDSpendSumRatio"] = -1
    _, cells, _ = construct_environment(panel, annual)
    assert cells.loc[cells.source_year.eq(2010), "rd_mean_pp"].isna().all()
    try:
        strict_annual(pd.concat([raw, raw.iloc[[0]]]))
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate annual key was not rejected.")
    # A known linear coefficient checks design alignment and clustered-fit execution.
    rng = np.random.default_rng(20261008)
    test = pd.DataFrame({FIRM: np.repeat([f"{i:06d}" for i in range(60)], 3),
                         AGRI: np.repeat(np.arange(60) % 2, 3), INDUSTRY: np.repeat([f"C{i % 12:02d}" for i in range(60)], 3),
                         RD_MEAN: rng.normal(size=180)})
    noise = rng.normal(size=180)
    y = 1 + 2 * test[AGRI] + .5 * test[RD_MEAN] + noise
    for field in OUTCOMES.values():
        test[field] = y
    rows, _ = estimate_design(test, AGRI + " + " + RD_MEAN, [RD_MEAN])
    direct = np.linalg.lstsq(np.column_stack([np.ones(180), test[AGRI], test[RD_MEAN]]), y, rcond=None)[0]
    assert all(np.isclose(row["coefficient"], direct[1 if row["term"] == AGRI else 2]) for row in rows)
    print("R&D synthetic checks passed.")
