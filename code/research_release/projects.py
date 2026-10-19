from __future__ import annotations

import numpy as np
import pandas as pd
from . import sample
from . import rd
from . import history as patent_history
from . import environment as industry_environment
from . import project_titles

KEYS = [sample.FIRM, "year"]

MIN_REPORTERS = 3

MEASURES = {
    5: {"label": "Development-expenditure green R&D title-record share",
        "variable": "lag_clear_green_rd_development_firm_share_pp", "role": "primary"},
    2: {"label": "Research-expense green R&D title-record share",
        "variable": "lag_clear_green_rd_expense_firm_share_pp", "role": "separate companion"},
}

def firm_flags(records: pd.DataFrame, code: int) -> tuple[pd.DataFrame, dict]:
    scope = records.loc[records.InvestSourceCode.eq(code)].copy()
    usable = scope.loc[scope.normalized_title.ne("")].copy()
    flags = usable.groupby(KEYS, sort=True).agg(
        recorded_title_rows=("source_excel_row", "size"),
        clear_title_rows=("clear_record", "sum"),
        uncertain_title_rows=("uncertain_record", "sum"),
        accounting_adjustment_rows=("accounting_adjustment_title", "sum"),
    ).reset_index()
    flags["has_clear_title"] = flags.clear_title_rows.gt(0).astype(int)
    flags["has_uncertain_title"] = flags.uncertain_title_rows.gt(0).astype(int)
    if flags.duplicated(KEYS).any() or not flags.recorded_title_rows.gt(0).all():
        raise AssertionError("Recorded source-specific firm-year flags are not unique.")
    summary = {
        "source_code": code, "accounting_field": project_titles.SOURCE_FIELDS[code],
        "scope_records": len(scope), "usable_title_records": len(usable),
        "blank_title_records_not_in_denominator": len(scope) - len(usable),
        "clear_records": int(scope.clear_record.sum()),
        "uncertain_records": int(scope.uncertain_record.sum()),
        "recorded_firm_years": len(flags), "recorded_firms": int(flags[sample.FIRM].nunique()),
        "firm_years_with_clear_title": int(flags.has_clear_title.sum()),
        "firm_years_with_uncertain_but_no_clear_title": int(
            (flags.has_uncertain_title.eq(1) & flags.has_clear_title.eq(0)).sum()),
        "clear_accounting_adjustment_records": int(
            (scope.clear_record.eq(1) & scope.accounting_adjustment_title).sum()),
        "first_source_year": int(scope.year.min()), "last_source_year": int(scope.year.max()),
    }
    return flags, summary

def environment(panel: pd.DataFrame, flags: pd.DataFrame, variable: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if panel.duplicated(KEYS).any() or flags.duplicated(KEYS).any():
        raise ValueError("Environment construction requires unique firm-year keys.")
    joined = flags.merge(panel[[*KEYS, sample.AGRI, sample.INDUSTRY]], on=KEYS,
                         how="inner", validate="one_to_one")
    valid = joined[sample.INDUSTRY].str.fullmatch(r"[A-Z]\d{2}")
    benchmark = joined.loc[joined[sample.AGRI].eq(0) & valid].copy()
    cells = benchmark.groupby([sample.INDUSTRY, "year"]).agg(
        recorded_reporters=(sample.FIRM, "nunique"),
        clear_title_reporters=("has_clear_title", "sum"),
        uncertain_title_reporters=("has_uncertain_title", "sum"),
    ).reset_index().rename(columns={"year": "source_year"})
    cells["unadjusted_share_pp"] = 100 * cells.clear_title_reporters / cells.recorded_reporters
    lagged = cells.assign(year=cells.source_year + 1)
    merged = panel.merge(lagged, on=[sample.INDUSTRY, "year"], how="left",
                         validate="many_to_one", sort=False)
    own = benchmark[[*KEYS, sample.INDUSTRY, "has_clear_title"]].rename(
        columns={"has_clear_title": "own_clear_title"})
    own["year"] += 1
    own["own_recorded_reporter"] = 1
    merged = merged.merge(own, on=[*KEYS, sample.INDUSTRY], how="left",
                          validate="one_to_one", sort=False)
    if len(merged) != len(panel) or not merged[KEYS].equals(panel[KEYS].reset_index(drop=True)):
        raise AssertionError("Exposure construction changed the panel keys or their order.")
    remove_own = merged[sample.AGRI].eq(0) & merged.own_recorded_reporter.eq(1)
    merged["effective_reporters"] = merged.recorded_reporters - remove_own.astype(int)
    merged["effective_clear_title_reporters"] = (
        merged.clear_title_reporters - np.where(remove_own, merged.own_clear_title, 0))
    supported = merged.effective_reporters.ge(MIN_REPORTERS)
    merged[variable] = (100 * merged.effective_clear_title_reporters / merged.effective_reporters).where(supported)
    known = merged.source_year.notna()
    if not merged.loc[known, "year"].eq(merged.loc[known, "source_year"] + 1).all():
        raise AssertionError("An exposure was not matched to the exact preceding calendar year.")
    if not merged.loc[supported, variable].between(0, 100).all():
        raise AssertionError("A supported title-record share is outside 0-100 percentage points.")
    if not (merged.effective_clear_title_reporters.dropna().ge(0).all()
            and merged.effective_clear_title_reporters.le(merged.effective_reporters).loc[known].all()):
        raise AssertionError("The numerator is not a subset of the recorded denominator.")
    construction = {
        "matched_recorded_firm_years": len(joined),
        "unmatched_recorded_firm_years": len(flags) - len(joined),
        "agricultural_recorded_firm_years_excluded_from_benchmark": int(joined[sample.AGRI].eq(1).sum()),
        "invalid_industry_recorded_firm_years": int((~valid).sum()),
        "nonagricultural_benchmark_firm_years": len(benchmark),
        "benchmark_industries": int(benchmark[sample.INDUSTRY].nunique()), "benchmark_cells": len(cells),
        "cells_with_at_least_three_recorded_firms_before_focal_removal": int(cells.recorded_reporters.ge(MIN_REPORTERS).sum()),
        "cells_with_no_identified_clear_title": int(cells.clear_title_reporters.eq(0).sum()),
        "current_panel_nonagricultural_focal_removals": int(remove_own.sum()),
        "current_panel_supported_exposures": int(supported.sum()),
        "current_panel_supported_zero_title_shares": int((supported & merged[variable].eq(0)).sum()),
    }
    return merged, cells, construction

def audit(merged: pd.DataFrame, variable: str) -> dict:
    masks, _ = sample.build_masks(merged)
    if [int(mask.sum()) for mask in masks.values()] != sample.EXPECTED_COUNTS:
        raise ValueError("The canonical sample flow changed.")
    common = masks["Common estimation sample"]
    selected = common & merged[variable].notna()
    history = patent_history.add_calendar_history(merged)
    with_history = selected & history[patent_history.HISTORY].notna()
    parts = {"Original common sample": common, "Input-matched sample": selected,
             "Common sample without eligible input": common & ~selected,
             "Input and own-history intersection": with_history}
    chosen = history.loc[selected]
    return {
        "overview": [row for name, mask in parts.items() for row in industry_environment.sample_overview(history.loc[mask], name)],
        "matched_key_sha256": rd.fingerprint(chosen),
        "history_key_sha256": rd.fingerprint(history.loc[with_history]),
        "matched_sample_keys": rd.records(chosen[[*KEYS, sample.AGRI, sample.INDUSTRY, variable,
            "source_year", "effective_reporters", "effective_clear_title_reporters"]]),
        "yearly": rd.records(chosen.groupby("year").agg(observations=(sample.FIRM, "size"),
            agricultural_observations=(sample.AGRI, "sum"), mean_share_pp=(variable, "mean")).reset_index()),
        "exposure_by_group": [{"group": group, "observations": len(f),
            "mean_share_pp": float(f[variable].mean()), "zero_share_observations": int(f[variable].eq(0).sum())}
            for group, value in sample.GROUPS.items() for f in [chosen.loc[sample.group_mask(chosen, value)]]],
    }

def self_test() -> None:
    firms = [f"{i:06d}" for i in range(1, 6)]
    panel = pd.DataFrame({sample.FIRM: firms * 2 + [firms[0]],
        "year": [2010] * 5 + [2011] * 5 + [2013], sample.AGRI: [0, 0, 0, 0, 1] * 2 + [0],
        sample.INDUSTRY: ["C13"] * 11})
    flags = pd.DataFrame({sample.FIRM: firms, "year": [2010] * 5,
        "has_clear_title": [1, 0, 0, 0, 1], "has_uncertain_title": [0, 1, 0, 0, 0]})
    variable = "test_title_share"
    merged, cells, _ = environment(panel, flags, variable)
    assert cells.recorded_reporters.iloc[0] == 4 and cells.clear_title_reporters.iloc[0] == 1
    assert merged.loc[merged.year.eq(2011) & merged[sample.AGRI].eq(1), variable].iloc[0] == 25
    assert merged.loc[merged.year.eq(2011) & merged[sample.FIRM].eq(firms[0]), variable].iloc[0] == 0
    assert merged.loc[merged.year.isin([2010, 2013]), variable].isna().all()
    limited, _, _ = environment(panel, flags.loc[flags[sample.FIRM].ne(firms[3])], variable)
    assert limited.loc[limited.year.eq(2011) & limited[sample.FIRM].eq(firms[0]), variable].isna().all()
    assert np.isclose(limited.loc[limited.year.eq(2011) & limited[sample.FIRM].eq(firms[3]), variable].iloc[0], 100 / 3)
    zero, _, _ = environment(panel, flags.assign(has_clear_title=0), variable)
    assert zero.loc[zero.year.eq(2011), variable].eq(0).all()
    changed = panel.copy()
    changed.loc[changed.year.eq(2011) & changed[sample.FIRM].eq(firms[0]), sample.INDUSTRY] = "C14"
    new_industry, _, _ = environment(changed, flags, variable)
    assert new_industry.loc[new_industry.year.eq(2011) & new_industry[sample.FIRM].eq(firms[0]), variable].isna().all()
    try:
        environment(panel, pd.concat([flags, flags.iloc[[0]]]), variable)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate firm-year flags were not rejected.")
    records = pd.DataFrame({sample.FIRM: [firms[0]] * 3 + [firms[1]], "year": [2010] * 4,
        "InvestSourceCode": [5, 5, 2, 5], "normalized_title": ["a", "a", "b", ""],
        "source_excel_row": [4, 5, 6, 7], "clear_record": [1, 1, 0, 0],
        "uncertain_record": [0, 0, 1, 1], "accounting_adjustment_title": [0, 1, 0, 0],
        "Amount": [np.nan, 0, -1, np.nan]})
    development, _ = firm_flags(records, 5)
    expense, _ = firm_flags(records, 2)
    assert len(development) == 1 and development.has_clear_title.iloc[0] == 1
    assert len(expense) == 1 and expense.has_clear_title.iloc[0] == 0
    no_amount, _ = firm_flags(records.drop(columns="Amount"), 5)
    assert development.equals(no_amount)
    print("Project-record self-tests passed: distinct firms, separate sources, unknown/blank titles, amount independence, agriculture/focal exclusion, exact lag and support.", flush=True)
