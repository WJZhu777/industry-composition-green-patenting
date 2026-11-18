from __future__ import annotations

from pathlib import Path
import openpyxl
import pandas as pd
from . import sample
from . import rd

MIN_REPORTERS = 3

def read_export(path: Path) -> pd.DataFrame:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.active
        # Some Excel exports list only column A in the stored dimensions.
        sheet.reset_dimensions()
        rows = sheet.iter_rows(values_only=True)
        fields = next(rows)
        next(rows)
        next(rows)
        frame = pd.DataFrame(list(rows), columns=fields)
    finally:
        workbook.close()
    if frame.columns.duplicated().any() or frame.columns.isna().any():
        raise ValueError(f"Invalid export field names: {path}")
    if frame.Symbol.isna().any():
        raise ValueError("Missing stock-code identifier.")
    codes = frame.Symbol.astype(str)
    if not codes.str.fullmatch(r"\d{1,6}").all():
        raise ValueError("Stock codes must contain one to six digits.")
    frame[sample.FIRM] = codes.str.zfill(6)
    dates = pd.to_datetime(frame.EndDate, errors="raise")
    if not (dates.dt.month.eq(12) & dates.dt.day.eq(31)).all():
        raise ValueError("Non-annual records need a separate selection rule.")
    frame["year"] = dates.dt.year
    return frame

def environment(panel: pd.DataFrame, groups: pd.DataFrame, variable: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source = panel[[sample.FIRM, "year", sample.AGRI, sample.INDUSTRY]].merge(
        groups[[sample.FIRM, "year", "firm_ratio_pp"]], on=[sample.FIRM, "year"], how="left", validate="one_to_one")
    benchmark = source.loc[source[sample.AGRI].eq(0) & source[sample.INDUSTRY].str.fullmatch(r"[A-Z]\d{2}")].copy()
    cells = benchmark.groupby([sample.INDUSTRY, "year"]).agg(
        nonag_firms=(sample.FIRM, "size"), reporters=("firm_ratio_pp", "count"),
        ratio_sum=("firm_ratio_pp", lambda s: s.sum(min_count=1)),
    ).reset_index().rename(columns={"year": "input_source_year"})
    cells["reporter_share"] = cells.reporters / cells.nonag_firms
    lag = cells.copy()
    lag["year"] = lag.input_source_year + 1
    merged = panel.merge(lag, on=[sample.INDUSTRY, "year"], how="left", validate="many_to_one", sort=False)
    # Remove a non-agricultural focal firm's preceding-year contribution only in the same industry.
    focal = benchmark[[sample.FIRM, sample.INDUSTRY, "year", "firm_ratio_pp"]].rename(columns={"firm_ratio_pp": "focal_previous_ratio"})
    focal["year"] += 1
    merged = merged.merge(focal, on=[sample.FIRM, sample.INDUSTRY, "year"], how="left", validate="one_to_one", sort=False)
    remove = merged[sample.AGRI].eq(0) & merged.focal_previous_ratio.notna()
    merged["effective_reporters"] = merged.reporters - remove.astype(int)
    numerator = merged.ratio_sum - merged.focal_previous_ratio.where(remove, 0)
    merged[variable] = (numerator / merged.effective_reporters).where(merged.effective_reporters.ge(MIN_REPORTERS))
    tiny = merged[variable].between(-1e-10, 0, inclusive="left")
    merged.loc[tiny, variable] = 0.
    if merged[variable].dropna().lt(0).any():
        raise ValueError("Negative industry mean from nonnegative amounts.")
    if not merged[[sample.FIRM, "year"]].equals(panel[[sample.FIRM, "year"]].reset_index(drop=True)):
        raise AssertionError("Environment matching changed firm-year keys.")
    available = merged[variable].notna()
    if not merged.loc[available, "year"].eq(merged.loc[available, "input_source_year"] + 1).all():
        raise AssertionError("Not a strict preceding-calendar-year exposure.")
    if not merged.loc[available, "effective_reporters"].ge(MIN_REPORTERS).all():
        raise AssertionError("Insufficient leave-one-out non-agricultural support.")
    audit = {"benchmark_valid_nonag_firm_years": int(benchmark.firm_ratio_pp.notna().sum()),
        "focal_nonag_contributions_removed": int(remove.sum()),
        "removals_losing_minimum_support": int((remove & merged.reporters.ge(MIN_REPORTERS) & merged.effective_reporters.lt(MIN_REPORTERS)).sum()),
        "variable": variable, "industry_year_cells": rd.records(cells)}
    return merged, cells, audit

def sample_overview(frame: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for name, value in sample.GROUPS.items():
        part = frame.loc[sample.group_mask(frame, value)]
        rows.append({"sample": label, "group": name, "observations": len(part), "firms": part[sample.FIRM].nunique(),
            "industries": part[sample.INDUSTRY].nunique(), "first_year": part.year.min(), "last_year": part.year.max(),
            "patent_count_mean": part[sample.COUNT].mean(), "patent_entry_rate": part[sample.ENTRY].mean()})
    return rd.records(pd.DataFrame(rows))
