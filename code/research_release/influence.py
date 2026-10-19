from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.linalg import lstsq
from . import sample
from . import accounting

OUTCOMES = list(accounting.OUTCOMES)

METRICS = [
    "sparse_agri", "control_agri", "full_agri", "full_minus_control",
    "sparse_minus_full", "gelbach_gtos_total", "gelbach_gtos_share_of_change",
    "control_to_full_gtos_partition",
]

def fingerprint(frame: pd.DataFrame) -> str:
    keys = frame[[sample.FIRM, "year"]].sort_values([sample.FIRM, "year"])
    return hashlib.sha256(keys.to_csv(index=False).encode("utf-8")).hexdigest()

def sample_info(frame: pd.DataFrame) -> dict:
    agri = frame[sample.AGRI].eq(1)
    return {
        "observations": len(frame), "firms": int(frame[sample.FIRM].nunique()),
        "agricultural_observations": int(agri.sum()),
        "nonagricultural_observations": int((~agri).sum()),
        "agricultural_firms": int(frame.loc[agri, sample.FIRM].nunique()),
        "nonagricultural_firms": int(frame.loc[~agri, sample.FIRM].nunique()),
        "industry_clusters": int(frame[sample.INDUSTRY].nunique()),
        "industries_with_agricultural_observations": int(frame.loc[agri, sample.INDUSTRY].nunique()),
        "first_year": int(frame.year.min()), "last_year": int(frame.year.max()),
        "sample_key_sha256": fingerprint(frame),
    }

def fit_case(frame: pd.DataFrame) -> tuple[list[dict], dict]:
    if frame[sample.AGRI].nunique() != 2:
        raise ValueError("An agricultural comparison group is empty.")
    design = accounting.build_design(frame)
    x = design["x"]
    y = frame[list(accounting.OUTCOMES.values())].to_numpy(dtype=float)
    blocks, w, z = design["blocks"], design["w"], design["z"]
    controls = np.r_[w, blocks["firm_controls"], blocks["province_fe"]]
    gtos = np.concatenate([blocks[name] for name in accounting.GTOS_BLOCKS])
    gram, xy = x.T @ x, x.T @ y
    full = cho_solve(cho_factor(gram, check_finite=False), xy, check_finite=False)
    sparse_factor = cho_factor(gram[np.ix_(w, w)], check_finite=False)
    control_factor = cho_factor(gram[np.ix_(controls, controls)], check_finite=False)
    sparse = cho_solve(sparse_factor, xy[w], check_finite=False)
    control = cho_solve(control_factor, xy[controls], check_finite=False)
    auxiliary = cho_solve(sparse_factor, gram[np.ix_(w, z)], check_finite=False)
    control_auxiliary = cho_solve(control_factor, gram[np.ix_(controls, gtos)], check_finite=False)
    auxiliary_by_column = np.zeros(x.shape[1])
    auxiliary_by_column[z] = auxiliary[1]
    contributions = {
        name: auxiliary_by_column[blocks[name]] @ full[blocks[name]]
        for name in ["firm_controls", "province_fe", *accounting.GTOS_BLOCKS]
    }
    joint = sum(contributions[name] for name in accounting.GTOS_BLOCKS)
    control_partition = control_auxiliary[1] @ full[gtos]
    total = sparse[1] - full[1]
    movement = full[1] - control[1]
    identity_error = float(np.max(np.abs(sum(contributions.values()) - total)))
    control_identity_error = float(np.max(np.abs(control_partition + movement)))
    normal_error = float(np.max(np.abs(gram @ full - xy)) / max(1., np.max(np.abs(xy))))
    if max(identity_error, control_identity_error) > 1e-8 or normal_error > 1e-10:
        raise ArithmeticError("Accounting identities or normal equations failed.")

    # Check all deletions with unscaled row-level QR fits, not another Gram solve.
    raw = design["original_x"]
    qr_full, _, full_rank, _ = lstsq(raw, y, lapack_driver="gelsy")
    qr_sparse, _, sparse_rank, _ = lstsq(raw[:, w], y, lapack_driver="gelsy")
    qr_control, _, control_rank, _ = lstsq(raw[:, controls], y, lapack_driver="gelsy")
    qr_aux, _, _, _ = lstsq(raw[:, w], raw[:, z], lapack_driver="gelsy")
    qr_control_aux, _, _, _ = lstsq(raw[:, controls], raw[:, gtos], lapack_driver="gelsy")
    if (full_rank, sparse_rank, control_rank) != (raw.shape[1], len(w), len(controls)):
        raise ValueError("A retained-sample model is rank deficient; no pseudoinverse substitution.")
    qr_by_column = np.zeros(raw.shape[1])
    qr_by_column[z] = qr_aux[1]
    qr_joint = sum(qr_by_column[blocks[name]] @ qr_full[blocks[name]] for name in accounting.GTOS_BLOCKS)
    qr_control_partition = qr_control_aux[1] @ qr_full[gtos]
    comparisons = [(sparse[1], qr_sparse[1]), (control[1], qr_control[1]), (full[1], qr_full[1]),
                   (joint, qr_joint), (control_partition, qr_control_partition)]
    qr_error = max(float(np.max(np.abs(left - right))) for left, right in comparisons)
    if not all(np.allclose(left, right, rtol=1e-8, atol=1e-8) for left, right in comparisons):
        raise ArithmeticError(f"Direct unscaled QR fits disagree: {qr_error}")
    output = []
    for index, outcome in enumerate(OUTCOMES):
        row = {
            "outcome": outcome, "sparse_agri": float(sparse[1, index]),
            "control_agri": float(control[1, index]), "full_agri": float(full[1, index]),
            "full_minus_control": float(movement[index]), "sparse_minus_full": float(total[index]),
            "gelbach_gtos_total": float(joint[index]),
            "gelbach_gtos_share_of_change": float(joint[index] / total[index]) if total[index] != 0 else None,
            "control_to_full_gtos_partition": float(control_partition[index]),
        }
        row.update({f"gelbach_{name}": float(value[index]) for name, value in contributions.items()})
        if not all(np.isfinite(value) for key, value in row.items() if key != "outcome" and value is not None):
            raise ArithmeticError("A coefficient or partition is nonfinite.")
        output.append(row)
    diagnostics = {
        "design_columns": design["columns"], "design_rank": x.shape[1],
        "scaled_design_condition_number": float(np.linalg.cond(x)),
        "direct_unscaled_qr_max_error": qr_error, "gelbach_identity_max_error": identity_error,
        "control_to_full_identity_max_error": control_identity_error,
        "normal_equation_relative_error": normal_error,
    }
    return output, diagnostics

def summarize(baseline: list[dict], cases: list[dict]) -> list[dict]:
    result = []
    for reference in baseline:
        outcome = reference["outcome"]
        for metric in METRICS:
            values = [(case["removed_industry"], next(row[metric] for row in case["estimates"]
                       if row["outcome"] == outcome)) for case in cases if case["status"] == "estimated"]
            valid = [(industry, value) for industry, value in values if value is not None]
            if not valid:
                continue
            low, high = min(valid, key=lambda pair: pair[1]), max(valid, key=lambda pair: pair[1])
            base = reference[metric]
            largest = max(valid, key=lambda pair: abs(pair[1] - base)) if base is not None else None
            result.append({
                "outcome": outcome, "metric": metric, "baseline": base,
                "minimum": low[1], "minimum_removed_industry": low[0],
                "maximum": high[1], "maximum_removed_industry": high[0],
                "median": float(np.median([value for _, value in valid])),
                "positive_cases": sum(value > 0 for _, value in valid),
                "negative_cases": sum(value < 0 for _, value in valid),
                "zero_cases": sum(value == 0 for _, value in valid),
                "defined_cases": len(valid), "undefined_cases": len(values) - len(valid),
                "opposite_sign_to_baseline_cases": sum(value * base < 0 for _, value in valid) if base is not None else None,
                "largest_absolute_change": abs(largest[1] - base) if largest else None,
                "largest_change_removed_industry": largest[0] if largest else None,
            })
    return result

def self_test() -> None:
    rng = np.random.default_rng(20261010)
    firms = np.repeat(np.arange(96), 3)
    industries = firms % 6
    frame = pd.DataFrame({
        sample.FIRM: [f"{firm:06d}" for firm in firms], "year": np.tile([2010, 2011, 2012], 96),
        sample.AGRI: (firms % 4 < 2).astype(int), sample.INDUSTRY: [f"C{i:02d}" for i in industries],
        "province": np.where(industries == 5, "exclusive", np.where(firms % 2, "P1", "P0")),
    })
    for field in accounting.CONTROLS + sample.LAGGED:
        frame[field] = rng.normal(size=len(frame))
    frame[sample.FI] = (industries >= 3).astype(int)
    frame[sample.COUNT] = rng.poisson(np.exp(.2 + .4 * frame[sample.AGRI] + .15 * frame[sample.LAGGED[0]]))
    frame[sample.LOG_COUNT] = np.log1p(frame[sample.COUNT])
    frame[sample.ENTRY] = frame[sample.COUNT].gt(0).astype(int)
    baseline, _ = fit_case(frame)
    cases = []
    for industry in sorted(frame[sample.INDUSTRY].unique()):
        selected = frame.loc[frame[sample.INDUSTRY].ne(industry)].copy()
        assert len(selected) + int(frame[sample.INDUSTRY].eq(industry).sum()) == len(frame)
        assert selected[accounting.CONTROLS + sample.LAGGED + [sample.FI]].equals(frame.loc[selected.index, accounting.CONTROLS + sample.LAGGED + [sample.FI]])
        estimates, diagnostics = fit_case(selected)
        cases.append({"removed_industry": industry, "status": "estimated", "estimates": estimates})
        if industry == "C05":
            assert not any("exclusive" in name for name in diagnostics["design_columns"])
    assert len(summarize(baseline, cases)) == len(OUTCOMES) * len(METRICS)
    for invalid in [frame.loc[frame[sample.AGRI].eq(0)], frame.assign(**{sample.FI: 0})]:
        try:
            fit_case(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("An empty comparison or constant GTOS field was not rejected.")
    print("Industry-influence synthetic checks passed: shared sample, fixed proxies, disappearing categories, QR checks and identities.", flush=True)
