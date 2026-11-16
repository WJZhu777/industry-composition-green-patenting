from __future__ import annotations

import pandas as pd
from . import sample

SOURCE = "industry_code_std_source"

PRIMARY = "financial_library"

FALLBACK = "existing_panel_fallback"

CLASS = "sector_a_classification"

CLASSES = ["Sector A", "Other valid sector", "Unknown/invalid"]

def classify(codes: pd.Series) -> pd.Series:
    valid = codes.astype("string").str.fullmatch(r"[A-Z]\d{2}", na=False)
    result = pd.Series("Unknown/invalid", index=codes.index, dtype="string")
    result.loc[valid] = "Other valid sector"
    result.loc[valid & codes.astype("string").str.startswith("A", na=False)] = "Sector A"
    return result

def describe(frame: pd.DataFrame) -> dict:
    return {"observations": len(frame), "firms": int(frame[sample.FIRM].nunique()),
            "detailed_industries": int(frame[sample.INDUSTRY].nunique()),
            "industry_year_cells": len(frame[[sample.INDUSTRY, "year"]].drop_duplicates()),
            "patent_count_sum": float(frame[sample.COUNT].sum()) if len(frame) else None,
            "patent_count_mean": float(frame[sample.COUNT].mean()) if len(frame) else None,
            "patent_count_median": float(frame[sample.COUNT].median()) if len(frame) else None,
            "patent_entry_rate": float(frame[sample.ENTRY].mean()) if len(frame) else None,
            "zero_share": float(frame[sample.COUNT].eq(0).mean()) if len(frame) else None}

def stage_and_year_tables(frame: pd.DataFrame, masks: dict) -> tuple[list, list, list, list]:
    stages, years, cross, sources = [], [], [], []
    for label, mask in masks.items():
        for group in CLASSES:
            selected = frame.loc[mask & frame[CLASS].eq(group)]
            full_n = int(frame[CLASS].eq(group).sum())
            stages.append({"sample": label, "classification": group, **describe(selected),
                           "retention_of_group_full_panel": len(selected) / full_n if full_n else None})
    for label in ["Full panel", "Common estimation sample"]:
        selected = frame.loc[masks[label]]
        for year, annual in selected.groupby("year", sort=True):
            for group in CLASSES:
                rows = annual.loc[annual[CLASS].eq(group)]
                years.append({"sample": label, "year": int(year), "classification": group,
                              **describe(rows), "primary_n": int(rows[SOURCE].eq(PRIMARY).sum()),
                              "fallback_n": int(rows[SOURCE].eq(FALLBACK).sum())})
        for original in [0, 1]:
            for group in CLASSES:
                rows = selected.loc[selected[sample.AGRI].eq(original) & selected[CLASS].eq(group)]
                cross.append({"sample": label, "ccad_agricultural": original,
                              "classification": group, **describe(rows)})
        for source in [PRIMARY, FALLBACK]:
            for group in CLASSES:
                rows = selected.loc[selected[SOURCE].eq(source) & selected[CLASS].eq(group)]
                sources.append({"sample": label, "source": source, "classification": group, **describe(rows)})
    return stages, years, cross, sources

def source_agreement(frame: pd.DataFrame) -> dict:
    primary_valid = frame.primary_classification.ne("Unknown/invalid")
    old_valid = frame.fallback_classification.ne("Unknown/invalid")
    both = frame.loc[primary_valid & old_valid]
    detailed_agreement = both.primary_code.eq(both.industry_code)
    broad_agreement = both.primary_code.str[0].eq(both.industry_code.str[0])
    sector_a_agreement = both.primary_classification.eq(both.fallback_classification)
    confusion = []
    for primary_a in [False, True]:
        for old_a in [False, True]:
            selected = both.loc[both.primary_classification.eq("Sector A").eq(primary_a)
                                & both.fallback_classification.eq("Sector A").eq(old_a)]
            confusion.append({"primary_sector_a": primary_a, "existing_panel_sector_a": old_a,
                              "observations": len(selected), "firms": int(selected[sample.FIRM].nunique())})
    names = []
    sector_a = frame.loc[frame[CLASS].eq("Sector A")]
    for (code, source, name), rows in sector_a.groupby([sample.INDUSTRY, SOURCE, "industry_name_std"], dropna=False):
        names.append({"code": code, "source": source, "stored_name": None if pd.isna(name) else name,
                      "observations": len(rows)})
    return {"both_valid_observations": len(both),
            "detailed_agreement_n": int(detailed_agreement.sum()),
            "broad_sector_agreement_n": int(broad_agreement.sum()),
            "sector_a_membership_agreement_n": int(sector_a_agreement.sum()),
            "sector_a_membership_confusion": confusion, "sector_a_stored_names_by_source": names,
            "boundary": "Source agreement is not external verification of annual business revenue. Stored names are reported as-is, not used to overwrite any code."}

def transitions(frame: pd.DataFrame) -> dict:
    valid = frame.loc[frame[CLASS].ne("Unknown/invalid")].sort_values([sample.FIRM, "year"]).copy()
    shifted = valid.groupby(sample.FIRM)[["year", sample.INDUSTRY, CLASS, SOURCE]].shift()
    consecutive = valid.year.sub(shifted.year).eq(1)
    sector_cross = consecutive & valid[CLASS].ne(shifted[CLASS])
    detailed_cross = consecutive & valid[sample.INDUSTRY].ne(shifted[sample.INDUSTRY])
    source_change = valid[SOURCE].ne(shifted[SOURCE])
    both_primary = valid[SOURCE].eq(PRIMARY) & shifted[SOURCE].eq(PRIMARY)
    summary = []
    for label, scope in [("All consecutive valid-code pairs", consecutive),
                         ("Both endpoints primary", consecutive & both_primary),
                         ("Source changes", consecutive & source_change),
                         ("Both endpoints fallback", consecutive & valid[SOURCE].eq(FALLBACK) & shifted[SOURCE].eq(FALLBACK))]:
        selected = valid.loc[scope & sector_cross]
        summary.append({"scope": label, "consecutive_pairs": int(scope.sum()),
                        "detailed_code_changes": int((scope & detailed_cross).sum()),
                        "sector_a_crossings": len(selected), "crossing_firms": int(selected[sample.FIRM].nunique()),
                        "enter_sector_a": int((scope & sector_cross & valid[CLASS].eq("Sector A")).sum()),
                        "exit_sector_a": int((scope & sector_cross & shifted[CLASS].eq("Sector A")).sum())})
    yearly = []
    for year in sorted(valid.year.unique()):
        annual = sector_cross & valid.year.eq(year)
        yearly.append({"destination_year": int(year), "crossings": int(annual.sum()),
                       "with_source_change": int((annual & source_change).sum()),
                       "both_primary": int((annual & both_primary).sum())})
    groups = valid.groupby(sample.FIRM)[CLASS].agg(set)
    return {"consecutive_transition_summary": summary, "sector_a_crossings_by_year": yearly,
            "firms_ever_sector_a": int(groups.map(lambda s: "Sector A" in s).sum()),
            "firms_with_both_a_and_other_observed_codes": int(groups.map(lambda s: len(s) > 1).sum()),
            "nonconsecutive_valid_pairs_not_treated_as_annual_transitions": int((shifted.year.notna() & ~consecutive).sum()),
            "boundary": "These are changes in stored classifications, not verified business switches. Consecutive calendar years are required; missing/invalid years are not bridged."}

def support_diagnostics(frame: pd.DataFrame) -> dict:
    rows, sector_rows = [], []
    for label, mask in [("Full valid-classification panel", frame[CLASS].ne("Unknown/invalid")),
                        ("Original common sample", frame["is_common"]),
                        ("Primary-source subset of original common sample", frame["is_common"] & frame[SOURCE].eq(PRIMARY))]:
        selected = frame.loc[mask].copy()
        selected["sector_a"] = selected[CLASS].eq("Sector A").astype(int)
        cells = selected.groupby([sample.INDUSTRY, "year"])["sector_a"].agg(["size", "sum", "nunique"])
        industry_variation = selected.groupby(sample.INDUSTRY).sector_a.nunique()
        a_cells = cells.loc[cells["sum"].gt(0)]
        a = selected.loc[selected.sector_a.eq(1)]
        shares = a.groupby(sample.FIRM).size() / len(a) if len(a) else pd.Series(dtype=float)
        primary_years = sorted(selected.loc[selected[SOURCE].eq(PRIMARY), "year"].astype(int).unique().tolist())
        rows.append({"sample": label, "observations": len(selected), "sector_a": describe(a),
                     "years_with_primary_codes": primary_years, "sector_a_years": sorted(a.year.astype(int).unique().tolist()),
                     "a_cells": len(a_cells), "a_cells_with_both_alternative_groups": int(a_cells["nunique"].gt(1).sum()),
                     "all_cells_with_both_alternative_groups": int(cells["nunique"].gt(1).sum()),
                     "industries_with_within_industry_alternative_group_variation": int(industry_variation.gt(1).sum()),
                     "sector_a_indicator_equals_sum_of_a_industry_indicators": bool(selected.sector_a.eq(selected[sample.INDUSTRY].str.startswith("A").astype(int)).all()),
                     "largest_firm_share_of_sector_a_rows": float(shares.max()) if len(shares) else None,
                     "effective_firm_count_from_row_shares": float(1 / shares.pow(2).sum()) if len(shares) else None,
                     "a_cell_size_min": int(a_cells["size"].min()) if len(a_cells) else None,
                     "a_cell_size_median": float(a_cells["size"].median()) if len(a_cells) else None,
                     "a_cell_size_max": int(a_cells["size"].max()) if len(a_cells) else None,
                     "original_ccad_nonagricultural_sector_a_rows": int((selected.sector_a.eq(1) & selected[sample.AGRI].eq(0)).sum()),
                     "alternative_nonagricultural_rows_in_a_industries": int((selected.sector_a.eq(0) & selected[sample.INDUSTRY].str.startswith("A")).sum())})
        for code, cell in a.groupby(sample.INDUSTRY):
            sector_rows.append({"sample": label, "code": code, **describe(cell),
                                "primary_n": int(cell[SOURCE].eq(PRIMARY).sum()),
                                "fallback_n": int(cell[SOURCE].eq(FALLBACK).sum())})
    return {"sample_support": rows, "sector_a_detailed_codes": sector_rows,
            "interpretation": {"industry_fixed_effects": "The alternative sector-A indicator is exactly a function of detailed industry and is absorbed by detailed industry effects; no model was fitted to establish this identity.",
                               "industry_year_decomposition": "There are no industry-year cells with both alternative groups. The original agricultural-reference within-cell comparison cannot be reproduced for this definition; an all-between label would be definitional, not new evidence.",
                               "gtos_benchmark": "Stored GTOS excludes CCAD agricultural firms, not all sector-A firms. Reclassifying alone does not change that benchmark. Excluding all sector-A observations from a same-industry benchmark leaves no alternative non-agricultural observations in A01-A05, so benchmark values are unavailable, not zero.",
                               "source_and_period": "Primary annual financial codes and fallback codes are kept separate. Source changes must not be interpreted as verified annual revenue composition changes.",
                               "decision": "Sector A is used for a descriptive comparison. It does not replace the CCAD definition in the main analysis."}}

def self_test() -> None:
    codes = pd.Series(["A01", "A05", "C13", "J66", "A", "A1", "UNKNOWN", None, " A01", "a01"])
    assert classify(codes).tolist() == ["Sector A", "Sector A", "Other valid sector", "Other valid sector", *["Unknown/invalid"] * 6]
    synthetic = pd.DataFrame({sample.FIRM: ["000001"] * 5 + ["000002"] * 2,
                              "year": [2007, 2008, 2009, 2011, 2012, 2007, 2008],
                              sample.INDUSTRY: ["A01", "A02", "C13", "A01", "C13", "A04", "UNKNOWN"],
                              SOURCE: [PRIMARY, PRIMARY, PRIMARY, PRIMARY, FALLBACK, PRIMARY, FALLBACK]})
    synthetic[CLASS] = classify(synthetic[sample.INDUSTRY])
    checks = transitions(synthetic)
    all_pairs = checks["consecutive_transition_summary"][0]
    assert all_pairs["consecutive_pairs"] == 3 and all_pairs["sector_a_crossings"] == 2
    assert checks["consecutive_transition_summary"][1]["sector_a_crossings"] == 1
    assert checks["consecutive_transition_summary"][2]["sector_a_crossings"] == 1
    assert checks["nonconsecutive_valid_pairs_not_treated_as_annual_transitions"] == 1
    valid = synthetic.loc[synthetic[CLASS].ne("Unknown/invalid")]
    assert valid.groupby([sample.INDUSTRY, "year"])[CLASS].nunique().eq(1).all()
    print("Sector-A synthetic checks passed: unknown codes, strict calendar transitions, source switches and deterministic industry support.", flush=True)
