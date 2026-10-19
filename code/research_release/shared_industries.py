from __future__ import annotations

import numpy as np
import pandas as pd
from . import sample
from . import influence

def restrict_to_shared_industries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty or not frame[sample.AGRI].isin([0, 1]).all():
        raise ValueError("The input must contain valid agricultural labels.")
    if frame[sample.INDUSTRY].isna().any():
        raise ValueError("Missing industry codes cannot define shared industries.")
    groups = frame.groupby(sample.INDUSTRY, observed=True)[sample.AGRI].agg(["size", "sum"])
    groups = groups.rename(columns={"size": "observations", "sum": "agricultural_observations"})
    groups["nonagricultural_observations"] = groups.observations - groups.agricultural_observations
    groups["retained"] = groups.agricultural_observations.gt(0) & groups.nonagricultural_observations.gt(0)
    codes = groups.index[groups.retained]
    restricted = frame.loc[frame[sample.INDUSTRY].isin(codes)].copy()
    if restricted.empty or restricted[sample.AGRI].nunique() != 2:
        raise ValueError("No shared industries provide both comparison groups.")
    # Restrict firm-years, not all records of firms that ever enter a removed industry.
    pd.testing.assert_frame_equal(restricted, frame.loc[restricted.index])
    return restricted, groups.sort_index()

def cell_support(frame: pd.DataFrame) -> dict:
    cells = frame.groupby([sample.INDUSTRY, "year"], observed=True)[sample.AGRI].agg(["size", "sum"])
    agri, nonagri = cells["sum"], cells["size"] - cells["sum"]
    definitions = {
        "both_groups": agri.gt(0) & nonagri.gt(0),
        "agricultural_only": agri.gt(0) & nonagri.eq(0),
        "nonagricultural_only": agri.eq(0) & nonagri.gt(0),
    }
    output = {"total_cells": len(cells)}
    for name, mask in definitions.items():
        output[name] = {
            "cells": int(mask.sum()), "observations": int(cells.loc[mask, "size"].sum()),
            "agricultural_observations": int(agri.loc[mask].sum()),
            "nonagricultural_observations": int(nonagri.loc[mask].sum()),
        }
    if sum(value["cells"] for value in output.values() if isinstance(value, dict)) != len(cells):
        raise AssertionError("Industry-year support counts do not exhaust the sample.")
    return output

def comparisons(baseline: list[dict], restricted: list[dict]) -> list[dict]:
    output = []
    for old, new in zip(baseline, restricted, strict=True):
        if old["outcome"] != new["outcome"]:
            raise AssertionError("Outcome order differs between comparison samples.")
        output.append({
            "outcome": old["outcome"],
            "metrics": {key: {"baseline": old[key], "shared_industries": new[key],
                              "difference": new[key] - old[key] if old[key] is not None and new[key] is not None else None}
                        for key in old if key != "outcome"},
        })
    return output

def self_test() -> None:
    frame = pd.DataFrame({
        sample.FIRM: ["000001", "000002", "000003", "000002", "000004", "000005"],
        "year": [2010, 2011, 2010, 2010, 2010, 2011],
        sample.AGRI: [1, 0, 1, 0, 1, 0],
        sample.INDUSTRY: ["C01", "C01", "A01", "C02", "C03", "C03"],
        "unchanged_value": np.arange(6),
    }, index=[10, 11, 12, 13, 14, 15])
    restricted, industries = restrict_to_shared_industries(frame)
    assert list(restricted.index) == [10, 11, 14, 15]
    assert list(industries.index[industries.retained]) == ["C01", "C03"]
    assert restricted[sample.FIRM].eq("000002").sum() == 1
    assert cell_support(restricted)["both_groups"]["cells"] == 0
    pd.testing.assert_frame_equal(restrict_to_shared_industries(restricted)[0], restricted)
    for bad in [frame.assign(**{sample.AGRI: 0}), frame.assign(**{sample.INDUSTRY: None}),
                frame.assign(**{sample.AGRI: 2})]:
        try:
            restrict_to_shared_industries(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid group/industry support was not rejected.")
    influence.self_test()
    print("Shared-industry synthetic selection, row preservation, distinct cell support and accounting checks passed.", flush=True)
