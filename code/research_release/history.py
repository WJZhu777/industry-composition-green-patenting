from __future__ import annotations

import numpy as np
import pandas as pd
from . import sample
from . import rd

HISTORY = "own_green_patent_log1p_calendar_lag1"

HISTORY_COUNT = "own_green_patent_count_calendar_lag1"

HISTORY_YEAR = "own_patent_source_year"

SAMPLE = "Fixed R&D and own-patent-history sample"

SPECIFICATIONS = {
    "Control-adjusted": [],
    "Patent GTOS": rd.PATENT,
    "Non-patent R&D": [rd.RD_MEAN],
    "Own patent history": [HISTORY],
    "Own history + patent GTOS": [HISTORY, *rd.PATENT],
    "Own history + non-patent R&D": [HISTORY, rd.RD_MEAN],
    "Own history + R&D + patent GTOS": [HISTORY, rd.RD_MEAN, *rd.PATENT],
}

def add_calendar_history(panel: pd.DataFrame) -> pd.DataFrame:
    keys = [sample.FIRM, "year"]
    if panel[keys].isna().any().any() or panel.duplicated(keys).any():
        raise ValueError("Own-patent history requires unique, nonmissing firm-year keys.")
    if (panel.year % 1).ne(0).any():
        raise ValueError("Own-patent history requires integer calendar years.")
    counts = panel[sample.COUNT].dropna()
    if counts.lt(0).any() or (counts % 1).ne(0).any() or np.isinf(counts).any():
        raise ValueError("Own-patent history requires nonnegative integer counts.")
    source = panel[[*keys, sample.COUNT]].rename(columns={
        "year": HISTORY_YEAR, sample.COUNT: HISTORY_COUNT,
    }).copy()
    source["year"] = source[HISTORY_YEAR] + 1
    merged = panel.merge(source, on=keys, how="left", validate="one_to_one", sort=False)
    if len(merged) != len(panel) or not merged[keys].equals(panel[keys].reset_index(drop=True)):
        raise AssertionError("Constructing history changed panel keys or their order.")
    known = merged[HISTORY_YEAR].notna()
    if not merged.loc[known, "year"].eq(merged.loc[known, HISTORY_YEAR] + 1).all():
        raise AssertionError("History is not from the preceding calendar year.")
    merged[HISTORY] = np.log1p(merged[HISTORY_COUNT])
    return merged

def estimate(frame: pd.DataFrame, saved: dict) -> dict:
    rows, designs = [], []
    key_digest = rd.fingerprint(frame)
    for label, additions in SPECIFICATIONS.items():
        rhs = rd.BASE + (" + " + " + ".join(additions) if additions else "")
        print(f"Fitting patent-history model: {label}; N={len(frame)}", flush=True)
        model_rows, design = rd.estimate_design(frame, rhs, additions)
        for row in model_rows:
            row.update(sample=SAMPLE, model=label, sample_key_sha256=key_digest)
        rows.extend(model_rows)
        designs.append({"sample": SAMPLE, "model": label, **design})
    coefficients = pd.DataFrame(rows)
    agricultural = coefficients.loc[coefficients.term.eq(sample.AGRI)].copy()
    comparisons = [
        ("Patent GTOS without own history", "Control-adjusted", "Patent GTOS"),
        ("Non-patent R&D without own history", "Control-adjusted", "Non-patent R&D"),
        ("Own patent history", "Control-adjusted", "Own patent history"),
        ("Patent GTOS conditional on own history", "Own patent history", "Own history + patent GTOS"),
        ("Non-patent R&D conditional on own history", "Own patent history", "Own history + non-patent R&D"),
        ("Patent GTOS conditional on own history and R&D", "Own history + non-patent R&D", "Own history + R&D + patent GTOS"),
        ("Non-patent R&D conditional on own history and GTOS", "Own history + patent GTOS", "Own history + R&D + patent GTOS"),
    ]
    movements = []
    for outcome, part in agricultural.groupby("outcome", sort=False):
        endpoints = part.set_index("model").coefficient
        for label, before, after in comparisons:
            initial, final = float(endpoints[before]), float(endpoints[after])
            movements.append({"outcome": outcome, "comparison": label, "initial_model": before,
                              "augmented_model": after, "initial_coefficient": initial,
                              "augmented_coefficient": final, "augmented_minus_initial": final - initial,
                              "initial_minus_augmented": initial - final})
    reproduction = []
    if key_digest == saved["sample_diagnostics"]["key_sha256"]["R&D primary matched sample"]:
        saved_rows = pd.DataFrame(saved["models"]["coefficients"])
        mapping = {"Control-adjusted": "Control-adjusted", "Patent GTOS": "Patent GTOS",
                   "Non-patent R&D": "Non-patent R&D mean"}
        for label, original in mapping.items():
            for row in coefficients.loc[coefficients.model.eq(label)].to_dict("records"):
                match = saved_rows.loc[saved_rows["sample"].eq("R&D primary matched sample")
                    & saved_rows.model.eq(original) & saved_rows.outcome.eq(row["outcome"])
                    & saved_rows.term.eq(row["term"])]
                if len(match) != 1:
                    raise ValueError("The reference R&D result is missing or ambiguous for a baseline term.")
                for field in ["coefficient", "firm_cluster_se", "industry_cluster_se"]:
                    old = float(match.iloc[0][field])
                    if not np.isclose(row[field], old, rtol=1e-8, atol=1e-10):
                        raise AssertionError(f"R&D baseline did not reproduce: {label}, {row['outcome']}, {field}")
                reproduction.append({"model": label, "outcome": row["outcome"], "term": row["term"],
                                     "coefficient_difference": row["coefficient"] - float(match.iloc[0].coefficient)})
    return {"coefficients": rd.records(coefficients), "agricultural_coefficients": rd.records(agricultural),
            "coefficient_movements": movements, "designs": designs,
            "p2_reproduction": {"same_sample": bool(reproduction), "checked_terms": len(reproduction),
                                "details": reproduction},
            "inference": "OLS with the six firm controls, year and province indicators; no industry or firm effects. Firm and industry cluster covariances use a finite-sample correction and normal-reference intervals. Intervals for individual coefficients do not test changes between models. Coefficient changes are descriptive comparisons, not mediation or identified opportunity effects."}

def self_test() -> None:
    fixture = pd.DataFrame({sample.FIRM: ["000001", "000001", "000001", "000002", "000002"],
                            "year": [2007, 2008, 2010, 2007, 2008],
                            sample.COUNT: [0, 4, 9, 8, 1], sample.INDUSTRY: ["C01", "C02", "C02", "C01", "C01"]})
    merged = add_calendar_history(fixture)
    assert merged.loc[1, HISTORY] == 0
    assert merged.loc[4, HISTORY] == np.log1p(8)
    assert merged.loc[2, [HISTORY, HISTORY_YEAR]].isna().all()
    assert merged.loc[0, HISTORY] != np.log1p(fixture.loc[0, sample.COUNT])
    assert merged.loc[1, sample.INDUSTRY] == "C02"
    for bad in [pd.concat([fixture, fixture.iloc[[0]]]), fixture.assign(**{sample.COUNT: -1})]:
        try:
            add_calendar_history(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid history source was not rejected.")
    missing = fixture.copy()
    missing.loc[3, sample.COUNT] = np.nan
    assert pd.isna(add_calendar_history(missing).loc[4, HISTORY])
    rng = np.random.default_rng(20261009)
    test = pd.DataFrame({sample.FIRM: np.repeat([f"{i:06d}" for i in range(60)], 3),
                         sample.AGRI: np.repeat(np.arange(60) % 2, 3),
                         sample.INDUSTRY: np.repeat([f"C{i % 12:02d}" for i in range(60)], 3),
                         HISTORY: rng.normal(size=180), rd.RD_MEAN: rng.normal(size=180)})
    outcome = 1 + 2 * test[sample.AGRI] + .7 * test[HISTORY] - .4 * test[rd.RD_MEAN] + rng.normal(size=180)
    for field in rd.OUTCOMES.values():
        test[field] = outcome
    rows, _ = rd.estimate_design(test, f"{sample.AGRI} + {HISTORY} + {rd.RD_MEAN}", [HISTORY, rd.RD_MEAN])
    matrix = np.column_stack([np.ones(len(test)), test[sample.AGRI], test[HISTORY], test[rd.RD_MEAN]])
    direct = np.linalg.lstsq(matrix, outcome, rcond=None)[0]
    index = {sample.AGRI: 1, HISTORY: 2, rd.RD_MEAN: 3}
    assert all(np.isclose(row["coefficient"], direct[index[row["term"]]]) for row in rows)
    print("Patent-history synthetic calendar-lag, missingness, key and regression checks passed.", flush=True)
