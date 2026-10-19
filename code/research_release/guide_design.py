from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import numpy as np
import pandas as pd

FIRM, AGRI, INDUSTRY = "证券代码", "是否涉农", "industry_code_std"

CONTROLS = ["ln_assets", "roa", "leverage", "ocf_ratio", "soe", "list_age"]

FUNCTIONS = ("P", "W", "E", "R", "K", "M")

ROLES = ("process_rd", "supplier")

FIRST_YEAR, LAST_YEAR = 2018, 2020

NUISANCE = " + ".join(CONTROLS) + " + C(year) + C(province)"

def sample_hash(frame):
    keys = sorted(f"{f}|{int(y)}" for f, y in frame[[FIRM, "year"]].itertuples(index=False, name=None))
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()

def sample_summary(frame):
    ag = frame.loc[frame[AGRI].eq(1)]
    ng = frame.loc[frame[AGRI].eq(0)]
    industries = frame.groupby(INDUSTRY, sort=True)[AGRI].agg(["size", "sum"])
    shared = industries.loc[(industries["sum"] > 0) & (industries["size"] > industries["sum"])].index.tolist()
    return {"observations": len(frame), "firms": int(frame[FIRM].nunique()),
            "industries": int(frame[INDUSTRY].nunique()), "agricultural_observations": len(ag),
            "agricultural_firms": int(ag[FIRM].nunique()), "nonagricultural_observations": len(ng),
            "nonagricultural_firms": int(ng[FIRM].nunique()), "shared_industries": shared,
            "sample_key_sha256": sample_hash(frame)}

def descriptors(source, role):
    branches = source["revised_branch_ledger"]
    families = sorted(source["technology_families"])
    pairs = sorted({(b["technology_family"], f) for b in branches for f in b["functions"]})
    rows = []
    for item in source["industry_role_technology_matrix"]:
        value = item["roles"][role]
        relevant = [b for b in branches if item["code"] in b["conditional_role_codes"][role]]
        if value["status"] == "unknown_or_unlinked_role":
            if relevant or any(v is not None for v in value["function_flags"].values()):
                raise ValueError("Unknown role has a supplied value.")
            continue
        if value["status"] != "conditional_documented_domain" or not relevant:
            raise ValueError("Invalid documented role.")
        evidence = {(b["technology_family"], f) for b in relevant for f in b["functions"]}
        expected_flags = {f: int(any(pair[1] == f for pair in evidence)) for f in FUNCTIONS}
        if expected_flags != value["function_flags"]:
            raise ValueError("Function matrix differs from branch-specific roles.")
        if sorted({p[0] for p in evidence}) != value["technology_families"]:
            raise ValueError("Family matrix differs from branch-specific roles.")
        row = {INDUSTRY: item["code"], "role": role}
        row.update({"function__" + f: expected_flags[f] for f in FUNCTIONS})
        row.update({"family__" + family: int(any(p[0] == family for p in evidence)) for family in families})
        row.update({"pair__" + family + "__" + f: int((family, f) in evidence) for family, f in pairs})
        signature = json.dumps(sorted(evidence), ensure_ascii=False, separators=(",", ":"))
        row["profile"] = "RP_" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
        row["profile_evidence"] = sorted([list(p) for p in evidence])
        rows.append(row)
    if len({r[INDUSTRY] for r in rows}) != len(rows):
        raise ValueError("Duplicate industry descriptor.")
    signatures = {r["profile"]: r["profile_evidence"] for r in rows}
    if any(signatures[r["profile"]] != r["profile_evidence"] for r in rows):
        raise ValueError("Profile-hash collision.")
    return pd.DataFrame([{k: v for k, v in r.items() if k != "profile_evidence"} for r in rows]), rows

def block_columns(table, kind):
    prefix = {"functions": "function__", "families": "family__", "family_functions": "pair__"}[kind]
    return [c for c in table if c.startswith(prefix)]

def normalized(matrix):
    x = np.asarray(matrix, dtype=float)
    scales = np.linalg.norm(x, axis=0)
    scales[scales == 0] = 1.0
    return x / scales, scales

def rank_info(matrix, threshold=None):
    x, _ = normalized(matrix)
    values = np.linalg.svd(x, compute_uv=False)
    default = max(x.shape) * np.finfo(float).eps * (float(values[0]) if len(values) else 1.0)
    tol = default if threshold is None else threshold
    rank = int(np.sum(values > tol))
    return {"columns": int(x.shape[1]), "rank": rank, "full_column_rank": rank == x.shape[1],
            "rank_tolerance": float(tol), "singular_values_normalized": values.tolist(),
            "condition_number_normalized": float(values[0] / values[-1]) if rank == x.shape[1] and len(values) else None}

def matrix_check(frame, base, additions, name, columns=None):
    extra = additions.to_numpy(dtype=float) if isinstance(additions, pd.DataFrame) else np.asarray(additions, dtype=float)
    names = list(additions.columns) if isinstance(additions, pd.DataFrame) else list(columns or [])
    x = np.column_stack([base.to_numpy(dtype=float), extra])
    report = rank_info(x)
    ag_position = list(base.columns).index(AGRI)
    without_ag = rank_info(np.delete(x, ag_position, axis=1))
    report.update({"design": name, "base_columns": list(base.columns), "additional_columns": names,
                   "observations": len(frame), "sample_key_sha256": sample_hash(frame),
                   "agricultural_coefficient_identifiable": report["rank"] == without_ag["rank"] + 1,
                   "rank_without_agricultural_indicator": without_ag["rank"],
                   "dependent_dimensions": report["columns"] - report["rank"]})
    if isinstance(additions, pd.DataFrame):
        constants = {c: float(additions[c].iloc[0]) for c in additions if additions[c].nunique() == 1}
        duplicate_groups = defaultdict(list)
        for c in additions:
            if c not in constants:
                duplicate_groups[additions[c].to_numpy(dtype=float).tobytes()].append(c)
        report["constant_columns_not_removed_from_ledger"] = constants
        report["identical_nonconstant_column_groups"] = [g for g in duplicate_groups.values() if len(g) > 1]
        report["additional_variation_within_industry"] = bool(additions.groupby(frame[INDUSTRY]).nunique().gt(1).any().any()) if names else False
    return report

def preserved_space(nuisance, block):
    n, _ = normalized(nuisance)
    g, _ = normalized(block)
    if not rank_info(n)["full_column_rank"]:
        raise ValueError("Nuisance design is deficient; do not drop controls to manufacture estimability.")
    qn, _ = np.linalg.qr(n, mode="reduced")
    residual = g - qn @ (qn.T @ g)
    u, singular, _ = np.linalg.svd(residual, full_matrices=False)
    reference = np.linalg.svd(np.column_stack([n, g]), compute_uv=False)
    tol = max(n.shape[0], n.shape[1] + g.shape[1]) * np.finfo(float).eps * max(float(reference[0]), 1.0)
    rank = int(np.sum(singular > tol))
    basis = u[:, :rank]
    error = float(np.linalg.norm(residual - basis @ (basis.T @ residual)))
    if error > 1e-9 * max(1.0, float(np.linalg.norm(g))):
        raise ValueError("Guide-block column space not preserved.")
    return basis, {"additional_dimensions_beyond_nuisance": rank,
                   "absolute_svd_tolerance": float(tol), "singular_values_residualized": singular.tolist(),
                   "space_reconstruction_frobenius_error": error,
                   "orthogonalized_against": list(nuisance.columns),
                   "agricultural_indicator_was_not_projected_out": AGRI not in nuisance.columns,
                   "basis_vectors_are_not_measurement_indices": True}

def profile_design(frame):
    levels = sorted(frame["profile"].unique())
    values = pd.DataFrame({"profile__" + p: frame["profile"].eq(p).astype(float) for p in levels[1:]}, index=frame.index)
    rows = []
    for p in levels:
        selected = frame.loc[frame["profile"].eq(p)]
        rows.append({"profile": p, "reference": p == levels[0],
                     "industry_codes": sorted(selected[INDUSTRY].unique()), **sample_summary(selected)})
    return values, rows

def family_support(frame, columns):
    records = []
    for column in columns:
        for flag in (0, 1):
            selected = frame.loc[frame[column].eq(flag)]
            records.append({"descriptor": column, "flag": flag, **sample_summary(selected)})
    return records

def self_test():
    n = np.column_stack([np.ones(12), np.tile([0., 1., 2.], 4)])
    ag = np.tile([0., 0., 1., 1.], 3)
    g = np.column_stack([np.tile([0., 1.], 6), np.tile([0., 1.], 6), np.zeros(12)])
    basis, report = preserved_space(pd.DataFrame(n, columns=["Intercept", "control"]), g)
    assert report["additional_dimensions_beyond_nuisance"] == 1
    assert rank_info(np.column_stack([n, ag, g]))["rank"] == rank_info(np.column_stack([n, ag, basis]))["rank"]
    assert report["agricultural_indicator_was_not_projected_out"]
    absorbed, status = preserved_space(pd.DataFrame(n, columns=["Intercept", "control"]), n)
    assert absorbed.shape[1] == 0 and status["additional_dimensions_beyond_nuisance"] == 0
    # An exposure identical to the group label must make its coefficient nonunique.
    group_design = np.column_stack([n, ag, ag])
    assert rank_info(group_design)["rank"] == rank_info(np.delete(group_design, 2, axis=1))["rank"]
    print("Guide-design synthetic checks passed: aliases, exact absorption, span preservation and group nonidentifiability; no outcomes or fits.")
