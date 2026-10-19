from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLDS = (1, 2, 3, 5)

def build_supported_cells(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    work = panel[
        ["证券代码", "year", "是否涉农", "industry_code_std", outcome]
    ].dropna()
    cells = (
        work.groupby(
            ["year", "industry_code_std", "是否涉农"],
            observed=False,
        )
        .agg(
            observations=("证券代码", "size"),
            mean_outcome=(outcome, "mean"),
        )
        .reset_index()
        .pivot(
            index=["year", "industry_code_std"],
            columns="是否涉农",
            values=["observations", "mean_outcome"],
        )
    )
    cells.columns = [f"{left}_{right}" for left, right in cells.columns]
    cells = cells.reset_index().rename(
        columns={
            "observations_0": "nonagricultural_n",
            "observations_1": "agricultural_n",
            "mean_outcome_0": "nonagricultural_mean",
            "mean_outcome_1": "agricultural_mean",
        }
    )
    return cells.dropna(
        subset=[
            "nonagricultural_n",
            "agricultural_n",
            "nonagricultural_mean",
            "agricultural_mean",
        ]
    ).copy()

def decompose(
    panel: pd.DataFrame,
    cells: pd.DataFrame,
    outcome: str,
    outcome_label: str,
    threshold: int,
    benchmark: str,
) -> dict[str, object]:
    retained = cells.loc[
        cells["agricultural_n"].ge(threshold)
        & cells["nonagricultural_n"].ge(threshold)
    ].copy()
    if retained.empty:
        raise ValueError(f"No cells remain at threshold {threshold}.")

    weights = retained["agricultural_n"] / retained["agricultural_n"].sum()
    agricultural_mean = float(
        (weights * retained["agricultural_mean"]).sum()
    )

    if benchmark == "Main Table 1 compatible":
        nonagricultural_mean = float(
            panel.loc[panel["是否涉农"].eq(0), outcome].mean()
        )
    elif benchmark == "Retained overlap cells":
        nonagricultural_weights = (
            retained["nonagricultural_n"]
            / retained["nonagricultural_n"].sum()
        )
        nonagricultural_mean = float(
            (
                nonagricultural_weights
                * retained["nonagricultural_mean"]
            ).sum()
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    within = float(
        (
            weights
            * (
                retained["agricultural_mean"]
                - retained["nonagricultural_mean"]
            )
        ).sum()
    )
    counterfactual = float(
        (weights * retained["nonagricultural_mean"]).sum()
    )
    between = counterfactual - nonagricultural_mean
    gap = agricultural_mean - nonagricultural_mean

    baseline_agricultural_n = int(cells["agricultural_n"].sum())
    baseline_nonagricultural_n = int(cells["nonagricultural_n"].sum())
    retained_agricultural_n = int(retained["agricultural_n"].sum())
    retained_nonagricultural_n = int(retained["nonagricultural_n"].sum())

    return {
        "Outcome": outcome_label,
        "Benchmark": benchmark,
        "Minimum observations per group per cell": threshold,
        "Retained cells": int(len(retained)),
        "Retained agricultural firm-years": retained_agricultural_n,
        "Agricultural coverage of baseline supported sample": (
            retained_agricultural_n / baseline_agricultural_n
        ),
        "Retained non-agricultural firm-years in retained cells": (
            retained_nonagricultural_n
        ),
        "Non-agricultural coverage of baseline overlap sample": (
            retained_nonagricultural_n / baseline_nonagricultural_n
        ),
        "Agricultural mean": agricultural_mean,
        "Non-agricultural mean": nonagricultural_mean,
        "Aggregate gap": gap,
        "Within-cell component": within,
        "Between-cell component": between,
        "Between-cell share": between / gap if gap != 0 else np.nan,
    }

def cell_size_distribution(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, column in [
        ("Agricultural observations per cell", "agricultural_n"),
        ("Non-agricultural observations per cell", "nonagricultural_n"),
    ]:
        values = cells[column]
        rows.append(
            {
                "Cell-size measure": group,
                "Supported cells": int(len(cells)),
                "Minimum": float(values.min()),
                "25th percentile": float(values.quantile(0.25)),
                "Median": float(values.median()),
                "75th percentile": float(values.quantile(0.75)),
                "Maximum": float(values.max()),
                "Cells with one observation": int(values.eq(1).sum()),
                "Cells with fewer than three observations": int(
                    values.lt(3).sum()
                ),
                "Cells with fewer than five observations": int(
                    values.lt(5).sum()
                ),
            }
        )
    return pd.DataFrame(rows)

def threshold_retention(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_cells = len(cells)
    baseline_agricultural_n = cells["agricultural_n"].sum()
    baseline_nonagricultural_n = cells["nonagricultural_n"].sum()
    for threshold in THRESHOLDS:
        retained = cells.loc[
            cells["agricultural_n"].ge(threshold)
            & cells["nonagricultural_n"].ge(threshold)
        ]
        rows.append(
            {
                "Minimum observations per group per cell": threshold,
                "Retained cells": int(len(retained)),
                "Share of baseline supported cells": (
                    len(retained) / baseline_cells
                ),
                "Retained agricultural firm-years": int(
                    retained["agricultural_n"].sum()
                ),
                "Agricultural coverage": float(
                    retained["agricultural_n"].sum()
                    / baseline_agricultural_n
                ),
                "Retained non-agricultural firm-years": int(
                    retained["nonagricultural_n"].sum()
                ),
                "Non-agricultural overlap-sample coverage": float(
                    retained["nonagricultural_n"].sum()
                    / baseline_nonagricultural_n
                ),
            }
        )
    return pd.DataFrame(rows)
