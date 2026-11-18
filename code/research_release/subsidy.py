from __future__ import annotations

import numpy as np
import pandas as pd
from . import sample
from . import rd
from . import history as patent_history
from . import environment as industry_environment

VARIABLE = "lag_clear_green_rd_subsidy_firm_share_pp"

MIN_REPORTERS = 3

KEYS = [sample.FIRM, "year"]

def environment(panel: pd.DataFrame, flags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if panel.duplicated(KEYS).any() or flags.duplicated(KEYS).any():
        raise ValueError("Environment construction requires unique firm-year keys.")
    joined = flags.merge(panel[[*KEYS, sample.AGRI, sample.INDUSTRY]], on=KEYS, how="inner", validate="one_to_one")
    valid = joined[sample.INDUSTRY].str.fullmatch(r"[A-Z]\d{2}")
    benchmark = joined.loc[joined[sample.AGRI].eq(0) & valid].copy()
    cells = benchmark.groupby([sample.INDUSTRY, "year"]).agg(
        recorded_reporters=(sample.FIRM, "nunique"),
        clear_title_reporters=("has_clear_title", "sum"),
        uncertain_title_reporters=("has_uncertain_title", "sum"),
    ).reset_index().rename(columns={"year": "source_year"})
    if cells.clear_title_reporters.gt(cells.recorded_reporters).any():
        raise AssertionError("The clear-title numerator exceeds the recorded-firm denominator.")
    cells["unadjusted_share_pp"] = 100 * cells.clear_title_reporters / cells.recorded_reporters
    lagged = cells.copy()
    lagged["year"] = lagged.source_year + 1
    merged = panel.merge(lagged, on=[sample.INDUSTRY, "year"], how="left", validate="many_to_one", sort=False)
    own = benchmark[[*KEYS, sample.INDUSTRY, "has_clear_title"]].rename(columns={"year": "source_year", "has_clear_title": "own_clear_title"})
    own["year"] = own.source_year + 1
    own = own.drop(columns="source_year").assign(own_recorded_reporter=1)
    merged = merged.merge(own, on=[*KEYS, sample.INDUSTRY], how="left", validate="one_to_one", sort=False)
    if len(merged) != len(panel) or not merged[KEYS].equals(panel[KEYS].reset_index(drop=True)):
        raise AssertionError("The measure merge changed panel keys or order.")
    remove_own = merged[sample.AGRI].eq(0) & merged.own_recorded_reporter.eq(1)
    merged["effective_reporters"] = merged.recorded_reporters - remove_own.astype(int)
    merged["effective_clear_title_reporters"] = merged.clear_title_reporters - np.where(remove_own, merged.own_clear_title, 0)
    supported = merged.effective_reporters.ge(MIN_REPORTERS)
    merged[VARIABLE] = (100 * merged.effective_clear_title_reporters / merged.effective_reporters).where(supported)
    known = merged.source_year.notna()
    if not merged.loc[known, "year"].eq(merged.loc[known, "source_year"] + 1).all():
        raise AssertionError("An exposure did not use the exact previous calendar year.")
    if not merged.loc[supported, VARIABLE].between(0, 100).all():
        raise AssertionError("The supported title-share measure is outside 0-100 percentage points.")
    if not merged.loc[supported, "effective_reporters"].ge(MIN_REPORTERS).all():
        raise AssertionError("The denominator rule was not applied after focal removal.")
    construction = {
        "matched_recorded_firm_years": len(joined),
        "unmatched_recorded_firm_years": len(flags) - len(joined),
        "agricultural_recorded_firm_years_excluded_from_benchmark": int(joined[sample.AGRI].eq(1).sum()),
        "invalid_industry_recorded_firm_years": int((~valid).sum()),
        "nonagricultural_benchmark_firm_years": len(benchmark),
        "benchmark_industries": int(benchmark[sample.INDUSTRY].nunique()),
        "benchmark_cells": len(cells),
        "cells_with_at_least_three_recorded_firms_before_focal_removal": int(cells.recorded_reporters.ge(MIN_REPORTERS).sum()),
        "cells_with_no_identified_clear_title": int(cells.clear_title_reporters.eq(0).sum()),
        "current_panel_nonagricultural_focal_removals": int(remove_own.sum()),
        "current_panel_supported_exposures": int(supported.sum()),
        "current_panel_supported_zero_title_shares": int((supported & merged[VARIABLE].eq(0)).sum()),
        "denominator": "Distinct non-agricultural firms with at least one usable consolidated subsidy title in this export and a valid current source-year detailed industry; at least three remain after focal removal.",
        "numerator": "Distinct firms in that denominator with at least one title coded clear_title in the original subsidy classifications, regardless of recorded amount.",
        "zero": "Zero is an observed absence of identified clear-title records among the recorded benchmark firms, not proof of no green R&D or subsidy receipt. Missing firm-year records are not filled.",
    }
    return merged, cells, construction

def audit(merged: pd.DataFrame) -> dict:
    masks, _ = sample.build_masks(merged)
    if [int(mask.sum()) for mask in masks.values()] != sample.EXPECTED_COUNTS:
        raise ValueError("The canonical sample flow has changed.")
    common = masks["Common estimation sample"]
    selected = common & merged[VARIABLE].notna()
    history = patent_history.add_calendar_history(merged)
    with_history = selected & history[patent_history.HISTORY].notna()
    parts = {"Original common sample": common, "Input-matched sample": selected,
             "Common sample without eligible input": common & ~selected,
             "Input and own-history intersection": with_history}
    rows = [row for name, mask in parts.items() for row in industry_environment.sample_overview(history.loc[mask], name)]
    chosen = history.loc[selected]
    return {
        "overview": rows,
        "matched_key_sha256": rd.fingerprint(chosen),
        "history_key_sha256": rd.fingerprint(history.loc[with_history]),
        "matched_sample_keys": rd.records(chosen[[*KEYS, sample.AGRI, sample.INDUSTRY, VARIABLE, "source_year", "effective_reporters", "effective_clear_title_reporters"]]),
        "yearly": rd.records(chosen.groupby("year").agg(observations=(sample.FIRM, "size"), agricultural_observations=(sample.AGRI, "sum"), mean_share_pp=(VARIABLE, "mean")).reset_index()),
        "exposure_by_group": [
            {"group": group, "observations": len(f), "mean_share_pp": float(f[VARIABLE].mean()),
             "zero_share_observations": int(f[VARIABLE].eq(0).sum()), "maximum_share_pp": float(f[VARIABLE].max())}
            for group, value in sample.GROUPS.items() for f in [chosen.loc[sample.group_mask(chosen, value)]]
        ],
        "uncertainty": "Recorded-title classification and selected disclosure denominator; not a census of subsidy recipients or externally identified opportunity.",
    }

def self_test() -> None:
    source_keys = ["000001", "000002", "000003", "000004", "000005"]
    panel = pd.DataFrame({sample.FIRM: source_keys + source_keys + ["000001"],
        "year": [2010] * 5 + [2011] * 5 + [2013], sample.AGRI: [0, 0, 0, 0, 1] * 2 + [0],
        sample.INDUSTRY: ["C13"] * 11})
    flags = pd.DataFrame({sample.FIRM: source_keys, "year": [2010] * 5,
        "has_clear_title": [1, 0, 0, 0, 1], "has_uncertain_title": [0, 1, 0, 0, 0]})
    merged, cells, _ = environment(panel, flags)
    agriculture = merged.loc[merged.year.eq(2011) & merged[sample.AGRI].eq(1)]
    assert agriculture[VARIABLE].iloc[0] == 25
    focal_clear = merged.loc[merged.year.eq(2011) & merged[sample.FIRM].eq("000001")]
    assert focal_clear[VARIABLE].iloc[0] == 0 and focal_clear.effective_reporters.iloc[0] == 3
    focal_other = merged.loc[merged.year.eq(2011) & merged[sample.FIRM].eq("000002")]
    assert np.isclose(focal_other[VARIABLE].iloc[0], 100/3)
    assert merged.loc[merged.year.isin([2010, 2013]), VARIABLE].isna().all()
    assert cells.recorded_reporters.iloc[0] == 4 and cells.clear_title_reporters.iloc[0] == 1
    # A firm without a source record stays outside the denominator but may observe its industry.
    no_record = flags.loc[flags[sample.FIRM].ne("000004")]
    limited, _, _ = environment(panel, no_record)
    assert limited.loc[limited.year.eq(2011) & limited[sample.FIRM].eq("000004"), VARIABLE].iloc[0] == 100/3
    assert limited.loc[limited.year.eq(2011) & limited[sample.FIRM].eq("000001"), VARIABLE].isna().all()
    zero_flags = flags.copy()
    zero_flags["has_clear_title"] = 0
    zero, _, _ = environment(panel, zero_flags)
    assert zero.loc[zero.year.eq(2011), VARIABLE].eq(0).all()
    changed = panel.copy()
    changed.loc[changed.year.eq(2011) & changed[sample.FIRM].eq("000001"), sample.INDUSTRY] = "C14"
    new_industry, _, _ = environment(changed, flags)
    assert new_industry.loc[new_industry.year.eq(2011) & new_industry[sample.FIRM].eq("000001"), VARIABLE].isna().all()
    try:
        environment(panel, pd.concat([flags, flags.iloc[[0]]]))
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate firm-year flags were accepted.")
    print("PASS: unique firms, denominator/numerator, agricultural exclusion, focal removal, zero versus missing, exact calendar, industry changes and minimum reporters.")
