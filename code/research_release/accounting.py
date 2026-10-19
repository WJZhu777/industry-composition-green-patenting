from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from . import sample

SEED = 20261008

REPETITIONS = 2000

OUTCOMES = {"patent_count": sample.COUNT, "log1p_patent_count": sample.LOG_COUNT, "has_patent": sample.ENTRY}

CONTROLS = [*sample.FINANCIAL, "soe", "list_age"]

GTOS_BLOCKS = ["gtos_intensity", "gtos_diffusion", "frontier"]

def build_design(frame: pd.DataFrame) -> dict:
    parts = {
        "year_fe": pd.get_dummies(frame.year.astype(int), prefix="year", drop_first=True).astype(float),
        "firm_controls": frame[CONTROLS].astype(float),
        "province_fe": pd.get_dummies(frame.province.astype(str), prefix="prov", drop_first=True).astype(float),
        "gtos_intensity": frame[[sample.LAGGED[0]]].rename(columns={sample.LAGGED[0]: "gtos_intensity"}),
        "gtos_diffusion": frame[[sample.LAGGED[1]]].rename(columns={sample.LAGGED[1]: "gtos_diffusion"}),
        "frontier": frame[[sample.FI]].rename(columns={sample.FI: "frontier"}),
    }
    design = pd.concat([pd.DataFrame({"const": 1., "agri": frame[sample.AGRI]}, index=frame.index), *parts.values()], axis=1)
    blocks = {name: np.array([design.columns.get_loc(column) for column in part.columns]) for name, part in parts.items()}
    original = design.to_numpy(dtype=float)
    matrix = original.copy()
    scales = {}
    # Fixed affine rescaling improves Gram-matrix conditioning without changing any partition.
    for name in ["firm_controls", *GTOS_BLOCKS]:
        for index in blocks[name]:
            mean, sd = float(matrix[:, index].mean()), float(matrix[:, index].std(ddof=0))
            if not sd > 0:
                raise ValueError(f"Constant observed regressor: {design.columns[index]}")
            matrix[:, index] = (matrix[:, index] - mean) / sd
            scales[design.columns[index]] = {"mean": mean, "sd_ddof0": sd}
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite design values; no additional sample deletion is permitted.")
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError("The baseline design is not full rank.")
    w = np.r_[0, 1, blocks["year_fe"]]
    z = np.concatenate([blocks[name] for name in ["firm_controls", "province_fe", *GTOS_BLOCKS]])
    return {"x": matrix, "original_x": original, "columns": list(design.columns), "blocks": blocks,
            "w": w, "z": z, "pooled": np.delete(np.arange(matrix.shape[1]), 1), "scales": scales}

def metrics_from_fits(design: dict, restricted: np.ndarray, full: np.ndarray, auxiliary: np.ndarray,
                      pooled: np.ndarray, mean_difference: np.ndarray, gap: np.ndarray) -> dict:
    blocks, z, pooled_indices = design["blocks"], design["z"], design["pooled"]
    auxiliary_by_full_column = np.zeros(len(design["columns"]))
    auxiliary_by_full_column[z] = auxiliary[1]
    pooled_by_full_column = np.zeros_like(full)
    pooled_by_full_column[pooled_indices] = pooled
    output = {"gelbach.sparse_agri": restricted[1], "gelbach.full_agri": full[1],
              "gelbach.total_change": restricted[1] - full[1], "oaxaca.observed_gap": gap}
    for name in ["firm_controls", "province_fe", *GTOS_BLOCKS]:
        indices = blocks[name]
        output[f"gelbach.{name}"] = auxiliary_by_full_column[indices] @ full[indices]
    output["gelbach.gtos_total"] = sum(output[f"gelbach.{name}"] for name in GTOS_BLOCKS)
    for name in ["firm_controls", "province_fe", "year_fe", *GTOS_BLOCKS]:
        indices = blocks[name]
        output[f"oaxaca.{name}"] = mean_difference[indices] @ pooled_by_full_column[indices]
    output["oaxaca.gtos_total"] = sum(output[f"oaxaca.{name}"] for name in GTOS_BLOCKS)
    output["oaxaca.other_observed"] = sum(output[f"oaxaca.{name}"] for name in ["firm_controls", "province_fe", "year_fe"])
    output["oaxaca.accounted_total"] = output["oaxaca.gtos_total"] + output["oaxaca.other_observed"]
    output["oaxaca.remainder"] = gap - output["oaxaca.accounted_total"]
    output["gelbach.gtos_share_of_change"] = output["gelbach.gtos_total"] / output["gelbach.total_change"]
    output["oaxaca.gtos_share_of_accounted"] = output["oaxaca.gtos_total"] / output["oaxaca.accounted_total"]
    if not np.isfinite(np.array(list(output.values()))).all():
        raise ValueError("A component or ratio is not finite; record rather than replace the draw.")
    return output

def check_identities(metrics: dict) -> dict:
    gelbach_sum = sum(metrics[f"gelbach.{name}"] for name in ["firm_controls", "province_fe", *GTOS_BLOCKS])
    errors = {"gelbach_blocks_minus_change": gelbach_sum - metrics["gelbach.total_change"],
              "oaxaca_components_minus_gap": metrics["oaxaca.accounted_total"] + metrics["oaxaca.remainder"] - metrics["oaxaca.observed_gap"]}
    for name, value in errors.items():
        if np.max(np.abs(value)) > 1e-8:
            raise ArithmeticError(f"Decomposition identity failed: {name}; {value}")
    return {name: float(np.max(np.abs(value))) for name, value in errors.items()}

def direct_metrics(design: dict, y: np.ndarray, row_weights: np.ndarray, *, original: bool = False) -> dict:
    x = design["original_x"] if original else design["x"]
    w, z, pooled = design["w"], design["z"], design["pooled"]
    included = row_weights > 0
    root_weight = np.sqrt(row_weights[included])[:, None]
    weighted_x, weighted_y = x[included] * root_weight, y[included] * root_weight
    full = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)[0]
    restricted = np.linalg.lstsq(weighted_x[:, w], weighted_y, rcond=None)[0]
    auxiliary = np.linalg.lstsq(weighted_x[:, w], weighted_x[:, z], rcond=None)[0]
    pooled_beta = np.linalg.lstsq(weighted_x[:, pooled], weighted_y, rcond=None)[0]
    agricultural = x[:, 1] == 1
    n_a, n_n = row_weights[agricultural].sum(), row_weights[~agricultural].sum()
    if min(n_a, n_n) <= 0:
        raise ValueError("The draw has an empty agricultural or non-agricultural group.")
    difference = row_weights[agricultural] @ x[agricultural] / n_a - row_weights[~agricultural] @ x[~agricultural] / n_n
    gap = row_weights[agricultural] @ y[agricultural] / n_a - row_weights[~agricultural] @ y[~agricultural] / n_n
    return metrics_from_fits(design, restricted, full, auxiliary, pooled_beta, difference, gap)

def cluster_statistics(design: dict, y: np.ndarray, firm: pd.Series) -> dict:
    labels, inverse = np.unique(firm.to_numpy(), return_inverse=True)
    x = design["x"]
    k, q = x.shape[1], y.shape[1]
    upper = np.triu_indices(k)
    rows = []
    n, agricultural = [], []
    for index in range(len(labels)):
        selected = inverse == index
        xx, yy = x[selected], y[selected]
        status = xx[:, 1]
        if not np.all(status == status[0]):
            raise ValueError("Firm-level agricultural status changes within a cluster.")
        rows.append(np.r_[(xx.T @ xx)[upper], (xx.T @ yy).ravel(), xx.sum(axis=0), yy.sum(axis=0)])
        n.append(len(xx))
        agricultural.append(int(status[0]))
    return {"labels": labels, "inverse": inverse, "statistics": np.array(rows), "upper": upper,
            "n": np.array(n), "agricultural": np.array(agricultural), "k": k, "q": q}

def gram_metrics(design: dict, clusters: dict, weights: np.ndarray, totals: np.ndarray | None = None) -> tuple[dict, dict]:
    k, q = clusters["k"], clusters["q"]
    upper_width = len(clusters["upper"][0])
    totals = weights @ clusters["statistics"] if totals is None else totals
    gram = np.zeros((k, k))
    gram[clusters["upper"]] = totals[:upper_width]
    gram += np.triu(gram, 1).T
    xy = totals[upper_width:upper_width + k * q].reshape(k, q)
    group_totals = (weights * clusters["agricultural"]) @ clusters["statistics"][:, upper_width + k * q:]
    sums = totals[upper_width + k * q:]
    n_a = int((weights * clusters["agricultural"]) @ clusters["n"])
    n = int(weights @ clusters["n"])
    n_n = n - n_a
    if min(n_a, n_n) <= 0:
        raise ValueError("The draw has an empty agricultural or non-agricultural group.")
    difference = group_totals[:k] / n_a - (sums[:k] - group_totals[:k]) / n_n
    gap = group_totals[k:] / n_a - (sums[k:] - group_totals[k:]) / n_n
    w, z, pooled = design["w"], design["z"], design["pooled"]
    # A singular fixed design is an unsuccessful draw; never silently substitute a pseudoinverse.
    full = cho_solve(cho_factor(gram, check_finite=False), xy, check_finite=False)
    w_factor = cho_factor(gram[np.ix_(w, w)], check_finite=False)
    restricted = cho_solve(w_factor, xy[w], check_finite=False)
    auxiliary = cho_solve(w_factor, gram[np.ix_(w, z)], check_finite=False)
    pooled_beta = cho_solve(cho_factor(gram[np.ix_(pooled, pooled)], check_finite=False), xy[pooled], check_finite=False)
    metrics = metrics_from_fits(design, restricted, full, auxiliary, pooled_beta, difference, gap)
    errors = check_identities(metrics)
    residual = np.max(np.abs(gram @ full - xy)) / max(1., np.max(np.abs(xy)))
    if residual > 1e-10:
        raise ArithmeticError(f"Weighted normal equations failed: {residual}")
    diagnostics = {"observations": n, "agricultural_observations": n_a,
                   "unique_firms": int(np.count_nonzero(weights)),
                   "unique_agricultural_firms": int(np.count_nonzero(weights * clusters["agricultural"])),
                   "sampled_agricultural_clusters_including_repeats": int(weights @ clusters["agricultural"]),
                   "normal_equation_relative_error": float(residual), **errors}
    return metrics, diagnostics

def compare_metrics(left: dict, right: dict) -> float:
    if left.keys() != right.keys():
        raise AssertionError("Metric sets differ.")
    error = 0.
    for name in left:
        if not np.allclose(left[name], right[name], rtol=1e-8, atol=1e-8):
            raise AssertionError(f"Independent fit disagrees: {name}; {left[name]}; {right[name]}")
        error = max(error, float(np.max(np.abs(left[name] - right[name]))))
    return error

def self_test() -> None:
    rng = np.random.default_rng(SEED)
    n = 360
    frame = pd.DataFrame({sample.FIRM: np.repeat([f"{i:06d}" for i in range(90)], 4),
                          sample.AGRI: np.repeat(np.arange(90) % 2, 4),
                          "year": np.tile([2010, 2011, 2012, 2013], 90),
                          "province": np.repeat(np.array(["A", "B", "C"])[np.arange(90) % 3], 4)})
    for name in [*CONTROLS, *sample.LAGGED, sample.FI]:
        frame[name] = rng.normal(size=n) + .3 * frame[sample.AGRI]
    design = build_design(frame)
    y = np.column_stack([rng.normal(size=n) + j * frame[sample.AGRI] + frame[sample.LAGGED[0]] for j in [1, 2, 3]])
    clusters = cluster_statistics(design, y, frame[sample.FIRM])
    weights = np.bincount(rng.integers(0, 90, size=90), minlength=90)
    row_weights = weights[clusters["inverse"]]
    point, _ = gram_metrics(design, clusters, weights)
    compare_metrics(point, direct_metrics(design, y, row_weights))
    compare_metrics(point, direct_metrics(design, y, row_weights, original=True))
    expanded_indices = np.repeat(np.arange(n), row_weights)
    expanded = {**design, "x": design["x"][expanded_indices], "original_x": design["original_x"][expanded_indices]}
    compare_metrics(point, direct_metrics(expanded, y[expanded_indices], np.ones(len(expanded_indices))))
    assert all(np.unique(row_weights[clusters["inverse"] == i]).size == 1 for i in range(90))
    try:
        gram_metrics(design, clusters, weights * (1 - clusters["agricultural"]))
    except ValueError:
        pass
    else:
        raise AssertionError("An empty group was not rejected.")
    try:
        gram_metrics(design, clusters, weights * (np.arange(90) % 3 != 2))
    except np.linalg.LinAlgError:
        pass
    else:
        raise AssertionError("An absent fixed-design category was not recorded as unestimable.")
    print("Accounting synthetic checks passed: cluster multiplicities, explicit duplication, rescaling, identities and invalid draws.", flush=True)

def summarize(point: dict, draws: list[dict]) -> list[dict]:
    summary = []
    for index, outcome in enumerate(OUTCOMES):
        for metric in point:
            values = np.array([draw["values"][metric][index] for draw in draws])
            low, high = np.quantile(values, [.025, .975], method="linear")
            summary.append({"outcome": outcome, "metric": metric, "point_estimate": float(point[metric][index]),
                            "bootstrap_mean": float(values.mean()), "bootstrap_se_ddof1": float(values.std(ddof=1)),
                            "ci95_percentile_low": float(low), "ci95_percentile_high": float(high),
                            "valid_draws": len(values), "percentile_interval_excludes_zero": bool(low > 0 or high < 0)})
    return summary
