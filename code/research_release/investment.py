from __future__ import annotations

import numpy as np
import pandas as pd
from . import sample
from . import rd
from . import history as patent_history
from . import environment as industry_environment

MEASURES = {
    "CSMAR expense ratio": {
        "field": "ExpenseToGrossRevenue",
        "variable": "lag_csmar_green_expense_ratio_mean_pp",
        "definition": "Database-reported environmental management and R&D expenses / total revenue, in percentage points; not a green-R&D-only measure.",
    },
    "CSMAR capital ratio": {
        "field": "CapitalToTotalAssets",
        "variable": "lag_csmar_green_capital_ratio_mean_pp",
        "definition": "Database-reported environmental CIP, fixed-asset and development amounts / total assets, in percentage points; not annual additions or investment growth.",
    },
}

def select_measure(total: pd.DataFrame, field: str) -> tuple[pd.DataFrame, dict]:
    keys = [sample.FIRM, "year"]
    if not {*keys, field}.issubset(total):
        raise ValueError(f"Required exported ratio absent: {field}")
    if total[keys].isna().any().any() or total.duplicated(keys).any():
        raise ValueError("Overall data require unique, nonmissing firm-year keys.")
    values = pd.to_numeric(total[field], errors="raise")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError(f"Infinite exported ratio: {field}")
    selected = total[keys].copy()
    # Use the exported ratio, including recorded zeros; never infer values from absent records.
    selected["firm_ratio_pp"] = values.where(values.ge(0))
    summary = {
        "exported_field": field,
        "exported_records": len(total),
        "missing_exported_ratios": int(values.isna().sum()),
        "recorded_zero_ratios": int(values.eq(0).sum()),
        "negative_ratios_ineligible": int(values.lt(0).sum()),
        "eligible_ratios": int(selected.firm_ratio_pp.count()),
        "ratios_above_100_pp_retained": int(values.gt(100).sum()),
        "minimum": values.min(), "median": values.median(),
        "p99": values.quantile(.99), "maximum": values.max(),
    }
    return selected, rd.records(pd.DataFrame([summary]))[0]

def compare(merged: pd.DataFrame, variable: str, label: str) -> dict:
    masks, _ = sample.build_masks(merged)
    if [int(mask.sum()) for mask in masks.values()] != sample.EXPECTED_COUNTS:
        raise ValueError("Canonical sample flow changed.")
    common = masks["Common estimation sample"]
    matched = common & merged[variable].notna()
    merged = patent_history.add_calendar_history(merged)
    history = matched & merged[patent_history.HISTORY].notna()
    selected = {
        "Original common sample": common,
        "Input-matched sample": matched,
        "Common sample without eligible input": common & ~matched,
        "Input and own-history intersection": history,
    }
    diagnostics = {
        "overview": [row for name, mask in selected.items()
                     for row in industry_environment.sample_overview(merged.loc[mask], name)],
        "yearly": rd.records(merged.loc[matched].groupby("year").agg(
            observations=(sample.FIRM, "size"), agricultural_observations=(sample.AGRI, "sum")).reset_index()),
        "history_unavailable": int((matched & ~history).sum()),
        "matched_key_sha256": rd.fingerprint(merged.loc[matched]),
        "history_key_sha256": rd.fingerprint(merged.loc[history]),
        "matched_sample_keys": rd.records(merged.loc[matched,
            [sample.FIRM, "year", sample.AGRI, variable, "effective_reporters"]]),
    }
    rows, designs, movements, statuses = [], [], [], []
    for name, mask, history_terms in [
        ("Input-matched sample", matched, []),
        ("Input and own-history intersection", history, [patent_history.HISTORY]),
    ]:
        frame = merged.loc[mask].copy()
        group_firms = frame.groupby(sample.AGRI)[sample.FIRM].nunique()
        if len(group_firms) != 2 or group_firms.min() < 2 or frame[sample.INDUSTRY].nunique() < 2:
            statuses.append({"sample": name, "status": "not estimated: insufficient groups or clusters"})
            continue
        required = [*rd.CONTROLS, *rd.PATENT, *rd.OUTCOMES.values(), variable, *history_terms]
        if frame[required].isna().any().any() or np.isinf(frame[required].to_numpy(dtype=float)).any():
            raise ValueError("Incomplete or infinite fields in the fixed comparison sample.")
        suffix = " + own history" if history_terms else ""
        specs = {
            "Control-adjusted" + suffix: history_terms,
            "Patent GTOS" + suffix: [*history_terms, *rd.PATENT],
            "Non-patent input only" + suffix: [*history_terms, variable],
        }
        digest = rd.fingerprint(frame)
        local = []
        for model, additions in specs.items():
            if model.startswith("Non-patent") and any(term in additions for term in rd.PATENT):
                raise AssertionError("Patent GTOS included in an input-only model.")
            rhs = rd.BASE + (" + " + " + ".join(additions) if additions else "")
            print(f"Environmental-investment {label}; {name}; {model}; N={len(frame)}", flush=True)
            estimates, design = rd.estimate_design(frame, rhs, additions)
            for row in estimates:
                row.update(sample=name, model=model, measure=label, sample_key_sha256=digest)
            local.extend(estimates)
            designs.append({"sample": name, "model": model, **design})
        agricultural = [row for row in local if row["term"] == sample.AGRI]
        for outcome in rd.OUTCOMES:
            endpoints = {row["model"]: row["coefficient"] for row in agricultural if row["outcome"] == outcome}
            initial = endpoints["Control-adjusted" + suffix]
            for model in ["Patent GTOS" + suffix, "Non-patent input only" + suffix]:
                final = endpoints[model]
                movements.append({"sample": name, "outcome": outcome, "model": model,
                    "control_coefficient": initial, "augmented_coefficient": final,
                    "augmented_minus_control": final - initial})
        rows.extend(local)
        statuses.append({"sample": name, "status": "estimated on identical keys for all three specifications and outcomes"})
    return {
        "sample_diagnostics": diagnostics, "model_status": statuses,
        "coefficients": rows, "agricultural_coefficients": [r for r in rows if r["term"] == sample.AGRI],
        "agricultural_coefficient_movements": movements, "designs": designs,
        "inference": "Original six controls and year/province indicators; OLS/LPM; firm and industry cluster SEs with existing finite-sample corrections and normal-reference intervals. No firm/industry fixed effects. Conditional on recorded database measures, exposure construction and selected samples; paired movements have no separate CI.",
    }

def self_test() -> None:
    keys = ["000001", "000002", "000003", "000004", "000005"]
    frame = pd.DataFrame({sample.FIRM: keys, "year": [2010] * 5,
        "ExpenseToGrossRevenue": [0., np.nan, -1., 150., .5]})
    measure, _ = select_measure(frame, "ExpenseToGrossRevenue")
    assert measure.firm_ratio_pp.iloc[0] == 0
    assert measure.firm_ratio_pp.iloc[1:3].isna().all()
    assert measure.firm_ratio_pp.iloc[3] == 150
    try:
        select_measure(pd.concat([frame, frame.iloc[[0]]]), "ExpenseToGrossRevenue")
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate overall keys were not rejected.")
    panel = pd.DataFrame({sample.FIRM: keys * 3, "year": np.repeat([2010, 2011, 2013], 5),
        sample.AGRI: [0, 0, 0, 0, 1] * 3, sample.INDUSTRY: ["C01"] * 15})
    groups = pd.DataFrame({sample.FIRM: keys[:4], "year": [2010] * 4, "firm_ratio_pp": [0., 1., 2., 3.]})
    joined, _, _ = industry_environment.environment(panel, groups, "test_environment")
    assert joined.loc[joined.year.eq(2011) & joined[sample.AGRI].eq(1), "test_environment"].iloc[0] == 1.5
    assert joined.loc[joined.year.eq(2011) & joined[sample.FIRM].eq(keys[0]), "test_environment"].iloc[0] == 2.
    assert joined.loc[joined.year.eq(2013), "test_environment"].isna().all()
    changed = panel.copy()
    changed.loc[changed.year.eq(2011) & changed[sample.FIRM].eq(keys[0]), sample.INDUSTRY] = "C02"
    joined, _, _ = industry_environment.environment(changed, groups, "test_environment")
    assert joined.loc[joined.year.eq(2011) & joined[sample.FIRM].eq(keys[0]), "test_environment"].isna().all()
    print("Environmental-investment checks passed: exported zeros/blanks, duplicate keys, ratios, strict lag and same-industry leave-one-out.", flush=True)
