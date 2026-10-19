from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

def agricultural_reference_decomposition(
    data: pd.DataFrame, outcome: str, cell_column: str
) -> dict[str, float | int]:
    work = data[
        ["证券代码", "year", "是否涉农", cell_column, outcome]
    ].dropna()
    agricultural = work.loc[work["是否涉农"].eq(1)].copy()
    nonagricultural = work.loc[work["是否涉农"].eq(0)].copy()

    agricultural_cells = (
        agricultural.groupby(["year", cell_column], observed=False)
        .agg(
            agricultural_n=("证券代码", "size"),
            agricultural_mean=(outcome, "mean"),
        )
        .reset_index()
    )
    nonagricultural_cells = (
        nonagricultural.groupby(["year", cell_column], observed=False)
        .agg(
            nonagricultural_n=("证券代码", "size"),
            nonagricultural_mean=(outcome, "mean"),
        )
        .reset_index()
    )
    cells = agricultural_cells.merge(
        nonagricultural_cells,
        on=["year", cell_column],
        how="left",
        validate="1:1",
    )
    unsupported = cells["nonagricultural_mean"].isna()
    unsupported_rows = cells.loc[unsupported].copy()
    cells = cells.loc[~unsupported].copy()
    agricultural = agricultural.merge(
        cells[["year", cell_column]],
        on=["year", cell_column],
        how="inner",
        validate="many_to_one",
    )
    cells["agricultural_weight"] = (
        cells["agricultural_n"] / cells["agricultural_n"].sum()
    )
    agricultural_mean = float(agricultural[outcome].mean())
    nonagricultural_mean = float(nonagricultural[outcome].mean())
    within = float(
        (
            cells["agricultural_weight"]
            * (cells["agricultural_mean"] - cells["nonagricultural_mean"])
        ).sum()
    )
    counterfactual = float(
        (cells["agricultural_weight"] * cells["nonagricultural_mean"]).sum()
    )
    between = counterfactual - nonagricultural_mean
    gap = agricultural_mean - nonagricultural_mean
    return {
        "agricultural_mean": agricultural_mean,
        "nonagricultural_mean": nonagricultural_mean,
        "aggregate_gap": gap,
        "within": within,
        "between": between,
        "between_share": between / gap if gap else np.nan,
        "supported_cells": len(cells),
        "unsupported_cells": int(unsupported.sum()),
        "unsupported_firm_years": int(
            unsupported_rows["agricultural_n"].sum()
        )
        if len(unsupported_rows)
        else 0,
    }

def latest_year_decompositions(panel: pd.DataFrame) -> pd.DataFrame:
    outcomes = {
        "green_patent_application_count": "Patent count",
        "log1p_green_patent_application_count": "Log(1 + patent count)",
        "has_green_patent_application": "Patent entry indicator",
    }
    windows = {
        "Full sample, 2007-2024": 2024,
        "Excluding 2024, 2007-2023": 2023,
        "Excluding 2023-2024, 2007-2022": 2022,
    }
    rows: list[dict[str, Any]] = []
    for window, end_year in windows.items():
        data = panel.loc[panel["year"].le(end_year)].copy()
        for column, label in outcomes.items():
            result = agricultural_reference_decomposition(
                data, column, "industry_code_std"
            )
            rows.append({"window": window, "outcome": label, **result})
    return pd.DataFrame(rows)

def tail_decompositions(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for percentile, cap in [(95, 16.0), (97.5, 32.0), (99, 83.0)]:
        data = panel.copy()
        column = f"patent_count_cap_{str(percentile).replace('.', '_')}"
        data[column] = data["green_patent_application_count"].clip(upper=cap)
        result = agricultural_reference_decomposition(
            data, column, "industry_code_std"
        )
        rows.append(
            {
                "Specification": f"Patent count capped at p{percentile:g}",
                "Aggregate gap": result["aggregate_gap"],
                "Within-cell component": result["within"],
                "Between-cell component": result["between"],
                "Between-cell share": result["between_share"],
                "Cap": cap,
            }
        )
    return pd.DataFrame(rows)
