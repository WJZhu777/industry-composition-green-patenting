from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
from scipy.stats import norm
from scipy.stats import t
from . import sample
from . import rd
from . import guide_design

MODEL_NAMES = ("Control-adjusted", "Patent GTOS", "Guide complete source profile")

def compare(frame, role):
    profile_columns = [c for c in frame if c.startswith("profile__")]
    blocks = {MODEL_NAMES[0]: [], MODEL_NAMES[1]: rd.PATENT, MODEL_NAMES[2]: profile_columns}
    rows, designs = [], []
    for label, additions in blocks.items():
        rhs = rd.BASE + (" + " + " + ".join(additions) if additions else "")
        print(f"Guide-comparison {role}: {label}; N={len(frame)}; three original outcomes", flush=True)
        # Nominal profile contrasts parameterize conditioning, not separate technology effects.
        reported_terms = rd.PATENT if label == MODEL_NAMES[1] else []
        estimates, design = rd.estimate_design(frame, rhs, reported_terms)
        for row in estimates:
            df = int(row["industry_clusters"]) - 1
            critical = float(t.ppf(.975, df))
            row.update(model=label, sample_key_sha256=guide_design.sample_hash(frame), industry_t_df=df,
                       industry_cluster_p_t=float(2 * t.sf(abs(row["coefficient"] / row["industry_cluster_se"]), df)),
                       industry_cluster_ci95_t_low=row["coefficient"] - critical * row["industry_cluster_se"],
                       industry_cluster_ci95_t_high=row["coefficient"] + critical * row["industry_cluster_se"])
        rows.extend(estimates)
        designs.append({"model": label, "additions": additions, **design})
    agricultural = [r for r in rows if r["term"] == sample.AGRI]
    movements = []
    for outcome in rd.OUTCOMES:
        local = {r["model"]: r["coefficient"] for r in agricultural if r["outcome"] == outcome}
        for label in MODEL_NAMES[1:]:
            movements.append({"outcome": outcome, "model": label, "control_coefficient": local[MODEL_NAMES[0]],
                              "augmented_coefficient": local[label], "augmented_minus_control": local[label] - local[MODEL_NAMES[0]]})
    return {"coefficients": rows, "agricultural_coefficients": agricultural, "coefficient_movements": movements,
            "designs": designs, "empirical_fit_count": 9,
            "profile_parameter_reporting": "All complete-profile indicators are included. Individual reference-dependent profile contrasts are not interpreted or reported as technology effects; the agricultural coefficient is the target comparison."}

def independent_numeric_check(frame, models):
    maxima = {}
    checked, solutions = 0, 0

    def check(name, actual, stored):
        if not np.isfinite([actual, stored]).all() or not np.isclose(actual, stored, rtol=1e-8, atol=1e-9):
            raise AssertionError(f"Independent QR/cluster check failed: {name}, {actual}, {stored}")
        maxima[name] = max(maxima.get(name, 0.), float(abs(actual - stored)))

    for design in models["designs"]:
        matrix = patsy.dmatrix(design["rhs"], frame, return_type="dataframe", NA_action="raise")
        if list(matrix.columns) != design["design_columns"] or len(matrix) != len(frame):
            raise ValueError("Saved design or sample differs.")
        x = matrix.to_numpy(dtype=float)
        q, r = np.linalg.qr(x, mode="reduced")
        inverse_r = np.linalg.solve(r, np.eye(r.shape[0]))
        bread = inverse_r @ inverse_r.T
        for outcome, field in rd.OUTCOMES.items():
            y = frame[field].to_numpy(dtype=float)
            b = np.linalg.solve(r, q.T @ y)
            resid = y - x @ b
            score = x * resid[:, None]
            se, clusters = {}, {}
            for name, field_group in [("firm", sample.FIRM), ("industry", sample.INDUSTRY)]:
                codes, levels = pd.factorize(frame[field_group], sort=True)
                g = len(levels)
                sums = np.zeros((g, x.shape[1]))
                np.add.at(sums, codes, score)
                correction = g / (g - 1) * (len(x) - 1) / (len(x) - x.shape[1])
                se[name] = np.sqrt(np.diag(bread @ (sums.T @ sums) @ bread * correction))
                clusters[name] = g
            for row in models["coefficients"]:
                if row["model"] != design["model"] or row["outcome"] != outcome:
                    continue
                j = matrix.columns.get_loc(row["term"])
                check("coefficient", b[j], row["coefficient"])
                check("r_squared", 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean())), row["r_squared"])
                if row["observations"] != len(x) or row["sample_key_sha256"] != guide_design.sample_hash(frame):
                    raise AssertionError("Saved estimate uses a different sample.")
                for name in se:
                    s = se[name][j]
                    if row[name + "_clusters"] != clusters[name]:
                        raise AssertionError("Cluster count differs.")
                    check(name + "_cluster_se", s, row[name + "_cluster_se"])
                    check(name + "_cluster_p_normal", 2 * norm.sf(abs(b[j] / s)), row[name + "_cluster_p_normal"])
                    for side, sign in [("low", -1), ("high", 1)]:
                        check(name + "_cluster_ci95_" + side, b[j] + sign * norm.ppf(.975) * s,
                              row[name + "_cluster_ci95_" + side])
                df = clusters["industry"] - 1
                if row["industry_t_df"] != df:
                    raise AssertionError("Industry t-reference degrees of freedom differ.")
                check("industry_cluster_p_t", 2 * t.sf(abs(b[j] / se["industry"][j]), df), row["industry_cluster_p_t"])
                for side, sign in [("low", -1), ("high", 1)]:
                    check("industry_cluster_ci95_t_" + side, b[j] + sign * t.ppf(.975, df) * se["industry"][j],
                          row["industry_cluster_ci95_t_" + side])
                checked += 1
            solutions += 1
    expected_ag = [r for r in models["coefficients"] if r["term"] == sample.AGRI]
    if expected_ag != models["agricultural_coefficients"] or solutions != models["empirical_fit_count"]:
        raise AssertionError("Agricultural summary or fit count differs.")
    for movement in models["coefficient_movements"]:
        local = {r["model"]: r["coefficient"] for r in expected_ag if r["outcome"] == movement["outcome"]}
        check("movement_control", local[MODEL_NAMES[0]], movement["control_coefficient"])
        check("movement_augmented", local[movement["model"]], movement["augmented_coefficient"])
        check("movement_difference", local[movement["model"]] - local[MODEL_NAMES[0]], movement["augmented_minus_control"])
    return {"status": "Independent QR solutions and manually aggregated cluster-score covariances passed.",
            "design_outcome_solutions_checked": solutions, "coefficient_rows_checked": checked,
            "maximum_absolute_differences": maxima}

def self_test():
    rng = np.random.default_rng(20261012)
    n = 360
    frame = pd.DataFrame({sample.FIRM: np.repeat([f"{i:06d}" for i in range(120)], 3),
                          "year": np.tile([2018, 2019, 2020], 120),
                          sample.AGRI: np.repeat(np.arange(120) % 2, 3),
                          sample.INDUSTRY: np.repeat([f"C{i % 15:02d}" for i in range(120)], 3),
                          "profile": np.repeat([f"RP_{(i // 2) % 4}" for i in range(120)], 3), "z": rng.normal(size=n)})
    profiles, support = guide_design.profile_design(frame)
    assert len(profiles.columns) == 3 and support[0]["reference"]
    assert not any(r["reference"] for r in support[1:])
    frame = pd.concat([frame, profiles], axis=1)
    y = 2 + 1.5 * frame[sample.AGRI] + .7 * frame.z + rng.normal(size=n)
    for field in rd.OUTCOMES.values():
        frame[field] = y
    rhs = sample.AGRI + " + z + " + " + ".join(profiles.columns)
    rows, design = rd.estimate_design(frame, rhs, [])
    for row in rows:
        df = row["industry_clusters"] - 1
        row.update(model="synthetic", sample_key_sha256=guide_design.sample_hash(frame),
                   industry_t_df=df, industry_cluster_p_t=float(2 * t.sf(abs(row["coefficient"] / row["industry_cluster_se"]), df)),
                   industry_cluster_ci95_t_low=row["coefficient"] - t.ppf(.975, df) * row["industry_cluster_se"],
                   industry_cluster_ci95_t_high=row["coefficient"] + t.ppf(.975, df) * row["industry_cluster_se"])
    models = {"designs": [{"model": "synthetic", **design}], "coefficients": rows,
              "agricultural_coefficients": rows, "coefficient_movements": [], "empirical_fit_count": 3}
    independent_numeric_check(frame, models)
    print("Guide-comparison synthetic checks passed: nominal reference, fixed keys, QR/cluster covariance and normal/t inference.")
