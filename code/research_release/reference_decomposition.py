from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = {
    "green_patent_application_count": "Patent count",
    "has_green_patent_application": "Patent entry indicator",
    "log1p_green_patent_application_count": "Log(1 + patent count)",
}

def agricultural_reference_decomposition(
    dataframe: pd.DataFrame,
    outcome: str,
    cell_column: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    work = dataframe[
        [outcome, "是否涉农", "证券代码", "year", cell_column]
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
    unsupported_cells = cells.loc[unsupported].copy()
    cells = cells.loc[~unsupported].copy()

    supported_keys = cells[["year", cell_column]]
    agricultural = agricultural.merge(
        supported_keys,
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
            * (
                cells["agricultural_mean"]
                - cells["nonagricultural_mean"]
            )
        ).sum()
    )
    counterfactual = float(
        (
            cells["agricultural_weight"]
            * cells["nonagricultural_mean"]
        ).sum()
    )
    between = counterfactual - nonagricultural_mean
    gap = agricultural_mean - nonagricultural_mean

    cells["within_contribution"] = cells["agricultural_weight"] * (
        cells["agricultural_mean"] - cells["nonagricultural_mean"]
    )
    cells["between_contribution"] = cells["agricultural_weight"] * (
        cells["nonagricultural_mean"] - nonagricultural_mean
    )

    summary = {
        "Outcome": OUTCOMES.get(outcome, outcome),
        "Cell definition": cell_column,
        "Reference distribution": "Agricultural-reference",
        "Agricultural mean": agricultural_mean,
        "Non-agricultural mean": nonagricultural_mean,
        "Aggregate gap": gap,
        "Within-cell component": within,
        "Between-cell component": between,
        "Between-cell share": between / gap if gap != 0 else np.nan,
        "Supported agricultural cells": int(len(cells)),
        "Unsupported agricultural-only cells": int(unsupported.sum()),
        "Unsupported agricultural firm-years": int(
            unsupported_cells["agricultural_n"].sum()
        )
        if len(unsupported_cells)
        else 0,
        "Agricultural firm-years used": int(len(agricultural)),
        "Non-agricultural firm-years used": int(len(nonagricultural)),
    }
    return summary, cells

def overlap_reference_decompositions(
    dataframe: pd.DataFrame,
    outcome: str,
    cell_column: str,
) -> pd.DataFrame:
    work = dataframe[
        [outcome, "是否涉农", "证券代码", "year", cell_column]
    ].dropna()
    cells = (
        work.groupby(["year", cell_column, "是否涉农"], observed=False)
        .agg(
            observations=("证券代码", "size"),
            mean_outcome=(outcome, "mean"),
        )
        .reset_index()
        .pivot(
            index=["year", cell_column],
            columns="是否涉农",
            values=["observations", "mean_outcome"],
        )
    )
    cells.columns = [f"{left}_{right}" for left, right in cells.columns]
    cells = (
        cells.reset_index()
        .rename(
            columns={
                "observations_0": "nonagricultural_n",
                "observations_1": "agricultural_n",
                "mean_outcome_0": "nonagricultural_mean",
                "mean_outcome_1": "agricultural_mean",
            }
        )
        .dropna(
            subset=[
                "nonagricultural_n",
                "agricultural_n",
                "nonagricultural_mean",
                "agricultural_mean",
            ]
        )
    )
    cells["pooled_n"] = (
        cells["agricultural_n"] + cells["nonagricultural_n"]
    )

    agricultural_mean = float(
        (
            cells["agricultural_n"]
            / cells["agricultural_n"].sum()
            * cells["agricultural_mean"]
        ).sum()
    )
    nonagricultural_mean = float(
        (
            cells["nonagricultural_n"]
            / cells["nonagricultural_n"].sum()
            * cells["nonagricultural_mean"]
        ).sum()
    )
    gap = agricultural_mean - nonagricultural_mean

    references = {
        "Agricultural-reference": (
            cells["agricultural_n"] / cells["agricultural_n"].sum()
        ),
        "Non-agricultural-reference": (
            cells["nonagricultural_n"] / cells["nonagricultural_n"].sum()
        ),
        "Symmetric / pooled reference": (
            cells["pooled_n"] / cells["pooled_n"].sum()
        ),
        "Equal-cell reference": pd.Series(
            np.repeat(1 / len(cells), len(cells)),
            index=cells.index,
        ),
    }

    rows = []
    for reference_name, weights in references.items():
        within = float(
            (
                weights
                * (
                    cells["agricultural_mean"]
                    - cells["nonagricultural_mean"]
                )
            ).sum()
        )
        between = gap - within
        rows.append(
            {
                "Outcome": OUTCOMES.get(outcome, outcome),
                "Reference distribution": reference_name,
                "Aggregate gap": gap,
                "Within-cell component": within,
                "Between-cell component": between,
                "Between-cell share": between / gap if gap != 0 else np.nan,
                "Overlap cells": int(len(cells)),
                "Agricultural firm-years": int(cells["agricultural_n"].sum()),
                "Non-agricultural firm-years": int(
                    cells["nonagricultural_n"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)

def build_decomposition_diagnostics(
    panel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    main_rows = []
    support_rows = []
    alternative_rows = []
    industry_contribution_rows = []

    for outcome in OUTCOMES:
        summary, cells = agricultural_reference_decomposition(
            panel, outcome, "industry_code_std"
        )
        main_rows.append(summary)
        support_rows.append(
            {
                "Outcome": summary["Outcome"],
                "Supported agricultural cells": summary[
                    "Supported agricultural cells"
                ],
                "Unsupported agricultural-only cells": summary[
                    "Unsupported agricultural-only cells"
                ],
                "Unsupported agricultural firm-years": summary[
                    "Unsupported agricultural firm-years"
                ],
                "Agricultural firm-years used": summary[
                    "Agricultural firm-years used"
                ],
                "Non-agricultural firm-years used": summary[
                    "Non-agricultural firm-years used"
                ],
            }
        )
        alternative_rows.append(
            overlap_reference_decompositions(
                panel, outcome, "industry_code_std"
            )
        )

        contributions = (
            cells.groupby("industry_code_std", observed=False)
            .agg(
                Agricultural_weight=("agricultural_weight", "sum"),
                Within_component=("within_contribution", "sum"),
                Between_component=("between_contribution", "sum"),
                Industry_year_cells=("year", "size"),
                Agricultural_firm_years=("agricultural_n", "sum"),
                Nonagricultural_firm_years=(
                    "nonagricultural_n",
                    "sum",
                ),
            )
            .reset_index()
        )
        contributions["Outcome"] = summary["Outcome"]
        contributions["Absolute_between_component"] = contributions[
            "Between_component"
        ].abs()
        contributions["Share_of_absolute_between_contributions"] = (
            contributions["Absolute_between_component"]
            / contributions["Absolute_between_component"].sum()
        )
        industry_contribution_rows.append(contributions)

    winsor_rows = []
    count_column = "green_patent_application_count"
    for quantile in [0.95, 0.975, 0.99]:
        cap = float(panel[count_column].quantile(quantile))
        current = panel.copy()
        current["winsorized_patent_count"] = current[count_column].clip(
            upper=cap
        )
        summary, _ = agricultural_reference_decomposition(
            current,
            "winsorized_patent_count",
            "industry_code_std",
        )
        summary["Outcome"] = f"Patent count capped at p{quantile * 100:g}"
        summary["Upper cap"] = cap
        winsor_rows.append(summary)

    broad_rows = []
    for outcome in OUTCOMES:
        summary, _ = agricultural_reference_decomposition(
            panel, outcome, "broad_industry_sector"
        )
        summary["Cell definition"] = (
            "Broad sector from first letter of industry_code_std"
        )
        broad_rows.append(summary)

    base_summary, _ = agricultural_reference_decomposition(
        panel,
        "green_patent_application_count",
        "industry_code_std",
    )
    loio_rows = []
    for industry_code in sorted(panel["industry_code_std"].dropna().unique()):
        reduced = panel.loc[
            ~panel["industry_code_std"].eq(industry_code)
        ].copy()
        summary, _ = agricultural_reference_decomposition(
            reduced,
            "green_patent_application_count",
            "industry_code_std",
        )
        industry_names = panel.loc[
            panel["industry_code_std"].eq(industry_code),
            "industry_name_std",
        ].dropna()
        loio_rows.append(
            {
                "Excluded industry code": industry_code,
                "Excluded industry name": (
                    industry_names.mode().iloc[0]
                    if not industry_names.empty
                    else ""
                ),
                "Aggregate gap": summary["Aggregate gap"],
                "Within-cell component": summary["Within-cell component"],
                "Between-cell component": summary["Between-cell component"],
                "Between-cell share": summary["Between-cell share"],
                "Change from full-sample share": (
                    summary["Between-cell share"]
                    - base_summary["Between-cell share"]
                ),
            }
        )
    loio = pd.DataFrame(loio_rows)
    loio_summary = pd.DataFrame(
        [
            {
                "Full-sample between-cell share": base_summary[
                    "Between-cell share"
                ],
                "Minimum leave-one-industry-out share": loio[
                    "Between-cell share"
                ].min(),
                "Median leave-one-industry-out share": loio[
                    "Between-cell share"
                ].median(),
                "Maximum leave-one-industry-out share": loio[
                    "Between-cell share"
                ].max(),
                "Largest absolute share change": loio[
                    "Change from full-sample share"
                ].abs().max(),
                "Industries tested": int(len(loio)),
            }
        ]
    )

    return {
        "main_reproduction": pd.DataFrame(main_rows),
        "support_accounting": pd.DataFrame(support_rows),
        "alternative_weights": pd.concat(
            alternative_rows, ignore_index=True
        ),
        "winsorized_counts": pd.DataFrame(winsor_rows),
        "broad_industry": pd.DataFrame(broad_rows),
        "leave_one_industry_summary": loio_summary,
        "leave_one_industry_detail": loio.sort_values(
            "Change from full-sample share",
            key=lambda series: series.abs(),
            ascending=False,
        ),
        "industry_contributions": pd.concat(
            industry_contribution_rows, ignore_index=True
        ).sort_values(
            ["Outcome", "Absolute_between_component"],
            ascending=[True, False],
        ),
    }
