from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from . import sample
from . import accounting

ATOL, RTOL = 1e-8, 1e-8

def control_design(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        pd.DataFrame({"const": 1., "agri": frame[sample.AGRI]}, index=frame.index),
        pd.get_dummies(frame.year.astype(int), prefix="year", drop_first=True).astype(float),
        frame[accounting.CONTROLS].astype(float),
        pd.get_dummies(frame.province.astype(str), prefix="prov", drop_first=True).astype(float),
    ], axis=1).astype(float)

def fit(design: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[np.ndarray, dict]:
    if not design.index.equals(outcomes.index):
        raise ValueError("Design and outcome rows do not align.")
    raw = design.to_numpy(dtype=float)
    y = outcomes.to_numpy(dtype=float)
    if not np.isfinite(raw).all() or not np.isfinite(y).all():
        raise ValueError("Missing or nonfinite values; no additional row deletion is allowed.")
    coefficients, _, rank, _ = lstsq(raw, y, lapack_driver="gelsy")
    if rank != raw.shape[1]:
        raise ValueError("The model is rank deficient.")

    # Check the QR fit with a scaled SVD fit and reversed row order.
    scaled = raw.copy()
    continuous = [*accounting.CONTROLS, "gtos_intensity", "gtos_diffusion", "frontier"]
    scales = {}
    for column in continuous:
        if column not in design:
            continue
        j = design.columns.get_loc(column)
        mean, sd = raw[:, j].mean(), raw[:, j].std(ddof=0)
        if sd == 0:
            raise ValueError(f"Constant regressor: {column}")
        scaled[:, j] = (raw[:, j] - mean) / sd
        scales[j] = (mean, sd)
    checked, _, check_rank, _ = np.linalg.lstsq(scaled[::-1], y[::-1], rcond=None)
    if check_rank != raw.shape[1]:
        raise ValueError("The independent SVD fit is rank deficient.")
    for j, (mean, sd) in scales.items():
        checked[j] /= sd
        checked[0] -= mean * checked[j]
    np.testing.assert_allclose(coefficients, checked, rtol=RTOL, atol=ATOL)
    residual = y - raw @ coefficients
    normal_error = float(np.max(np.abs(scaled.T @ residual)) / max(1., np.max(np.abs(scaled.T @ y))))
    if normal_error > 1e-10:
        raise ArithmeticError("OLS normal equations failed.")
    return coefficients[design.columns.get_loc("agri")], {
        "observations": len(design), "columns": list(design.columns), "rank": int(rank),
        "scaled_condition_number": float(np.linalg.cond(scaled)),
        "qr_svd_max_coefficient_difference": float(np.max(np.abs(coefficients - checked))),
        "normal_equation_relative_error": normal_error,
    }
