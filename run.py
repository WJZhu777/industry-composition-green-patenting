"""Run checks or a selected analysis using inputs supplied outside this directory."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
sys.dont_write_bytecode = True


def release_files(root=ROOT):
    """List release content, leaving local environments and tool caches out."""
    root = Path(root)
    local_directories = {".git", ".idea", ".venv", "venv"}
    for folder, directories, files in os.walk(root):
        folder = Path(folder)
        directories[:] = [
            name for name in directories
            if name != "__pycache__" and not (folder == root and name in local_directories)
        ]
        for name in files:
            path = folder / name
            if path != root / "release_manifest.json":
                yield path


def check():
    modules = sorted((ROOT / "code/research_release").glob("*.py"))
    for path in modules:
        ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        importlib.import_module("research_release." + path.stem)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(f"Imported {len(modules)} modules. Tests used synthetic fixtures and public coding only.")


def canonical_sample(path):
    from research_release import sample
    panel = sample.load_panel(Path(path))
    # The retained leave-one-out helper uses row shifts for own-firm history.
    ordered = panel.sort_values([sample.FIRM, "year"])
    year_steps = ordered.groupby(sample.FIRM).year.diff().dropna()
    if not year_steps.eq(1).all():
        raise ValueError("Legacy leave-one-out own-history shifts require review for this nonconsecutive panel.")
    masks, excluded = sample.build_masks(panel)
    counts = [int(mask.sum()) for mask in masks.values()]
    if counts != sample.EXPECTED_COUNTS:
        raise ValueError(f"Canonical sample flow differs: {counts}; stop rather than select another sample.")
    common = panel.loc[masks["Common estimation sample"]].copy()
    if common[sample.FIRM].nunique() != 4290:
        raise ValueError("The common-sample firm count differs from the published baseline.")
    return panel, common, masks, excluded


def guide_frames(common):
    import pandas as pd
    import patsy
    from research_release import guide_design as gd, rd, sample
    source = json.loads((ROOT / "data/public/guide_coding.json").read_text(encoding="utf-8"))
    period = common.loc[common.year.between(gd.FIRST_YEAR, gd.LAST_YEAR)].copy()
    frames, reports = {}, {}
    for role in gd.ROLES:
        table, descriptors = gd.descriptors(source, role)
        merged = period.merge(table, on=sample.INDUSTRY, how="left", validate="many_to_one", sort=False)
        frame = merged.loc[merged.role.notna()].copy()
        indicators, support = gd.profile_design(frame)
        frame = pd.concat([frame, indicators], axis=1)
        nuisance = patsy.dmatrix(gd.NUISANCE, frame, return_type="dataframe", NA_action="raise")
        pairs = frame[gd.block_columns(table, "family_functions")].to_numpy(float)
        import numpy as np
        matrices = [np.column_stack([nuisance, indicators]), np.column_stack([nuisance, pairs]),
                    np.column_stack([nuisance, pairs, indicators])]
        ranks = [gd.rank_info(m)["rank"] for m in matrices]
        if len(set(ranks)) != 1:
            raise ValueError("Nominal profiles no longer preserve the complete conditional source space.")
        base = patsy.dmatrix(rd.BASE, frame, return_type="dataframe", NA_action="raise")
        for label, extra in [("Control-adjusted", pd.DataFrame(index=frame.index)),
                             ("Patent GTOS", frame[rd.PATENT]), ("Guide complete source profile", indicators)]:
            report = gd.matrix_check(frame, base, extra, label)
            if not report["full_column_rank"] or not report["agricultural_coefficient_identifiable"]:
                raise ValueError(f"Nonidentifiable fixed guide design: {role}, {label}")
        frames[role] = frame
        reports[role] = {"sample": gd.sample_summary(frame), "profile_support": support,
                         "source_descriptors": descriptors, "ranks_profile_pair_combined": ranks}
    return frames, reports


def validate_core_fields(panel):
    from research_release.count_models import post_2015_indicator
    required = ("baseline_nonagri_patent_per_firm", "industry_year_patent_count_nonagri",
                "industry_year_patent_firms_nonagri", "firm_count_nonagri")
    missing = [name for name in required if name not in panel.columns]
    if missing:
        raise ValueError("Missing required core fields: " + ", ".join(missing))
    post_2015_indicator(panel)


def validate_inputs(path):
    from research_release import sample, industry_sources, guide_design
    panel, common, masks, _ = canonical_sample(path)
    validate_core_fields(panel)
    rebuilt, threshold = industry_sources.build_gtos(panel, sample.INDUSTRY)
    industry_sources.verify_hybrid_rebuild(panel, rebuilt)
    frames, guide = guide_frames(common)
    ordered = panel.sort_values([sample.FIRM, "year"])
    gaps = ordered.groupby(sample.FIRM).year.diff().dropna()
    return {"sample_flow": [int(mask.sum()) for mask in masks.values()],
            "firms": int(panel[sample.FIRM].nunique()), "common_firms": int(common[sample.FIRM].nunique()),
            "strict_calendar_gtos": "passed", "rebuilt_fi_threshold": threshold,
            "nonconsecutive_firm_histories": int(gaps.ne(1).sum()),
            "guide_samples": {role: guide_design.sample_summary(frame) for role, frame in frames.items()},
            "estimation_performed": False}


def rd_input(panel, path):
    import pandas as pd
    from research_release import rd
    raw = pd.read_excel(path, header=0, skiprows=[1, 2], dtype={"Symbol": str})
    annual = rd.strict_annual(raw)
    merged, cells, audit = rd.construct_environment(panel, annual)
    selected = rd.samples(merged)
    return merged, selected, {"sample_diagnostics": rd.sample_details(selected)}, audit


def supplementary(panel, common, rd_path, green_path):
    import numpy as np
    import pandas as pd
    from research_release import controls, rd_triangulation as rt, investment_triangulation as gt
    from research_release import industry_sources as sources, sparse_cells, industry_baseline_models as baseline_models
    from research_release import sample
    prepared = panel.copy()
    prepared["industry_name_std"] = prepared["industry_name_std"].fillna(prepared[sample.INDUSTRY]).astype(str)
    rd_raw = pd.read_excel(rd_path, header=0, skiprows=[1, 2], dtype={"Symbol": str})
    rd_raw[sample.FIRM] = controls.standardize_code(rd_raw.Symbol)
    rd_raw["EndDate"] = pd.to_datetime(rd_raw.EndDate, errors="coerce")
    rd_raw["year"] = rd_raw.EndDate.dt.year.astype("Int64")
    for field in ["Source", "StateTypeCode"]:
        rd_raw[field] = pd.to_numeric(rd_raw[field], errors="coerce").astype("Int64")
    for field in rt.RD_NUMERIC_COLUMNS:
        rd_raw[field] = pd.to_numeric(rd_raw[field], errors="coerce")
    rd_raw["year_end_flag"] = rd_raw.EndDate.dt.month.eq(12) & rd_raw.EndDate.dt.day.eq(31)
    rd_raw["rd_field_completeness"] = rd_raw[rt.RD_NUMERIC_COLUMNS].notna().sum(axis=1)
    strict = rt.select_strict_annual(rd_raw)
    merged = rt.merge_rd(prepared, strict, "strict")
    cells, lagged = rt.build_industry_rd_environment(prepared, merged)
    rd_tables = {"coverage": rt.industry_environment_coverage(prepared, common, lagged),
                 "correlations": rt.gtos_correlations(prepared, lagged)[0], "exposure": rt.group_exposure(prepared, lagged)}
    green = pd.read_stata(green_path, convert_categoricals=False, preserve_dtypes=False)
    green[sample.FIRM] = green["股票代码"].astype("Int64").astype(str).str.zfill(6)
    green["year"] = pd.to_numeric(green["年份"], errors="raise").astype(int)
    green_window, matched = gt.build_matched_data(green, prepared)
    cells, lagged = gt.build_industry_year_investment(prepared, matched)
    green_tables = {"coverage": gt.cell_coverage(prepared, common, lagged),
                    "correlations": gt.gtos_correlations(prepared, lagged),
                    "exposure": gt.group_exposure_summary(common, lagged)}
    strict = strict.copy()
    fraction = (strict.RDSpendSumRatio / 100.).where(strict.RDSpendSumRatio.ge(0))
    strict["total_rd_intensity_w"] = controls.winsorize(fraction)
    lagged = strict[[sample.FIRM, "year", "total_rd_intensity_w"]].copy()
    lagged["year"] = lagged.year.astype(int) + 1
    lagged = lagged.rename(columns={"total_rd_intensity_w": "lagged_total_rd_intensity_w"})
    controls_sample = common.merge(lagged, on=[sample.FIRM, "year"], how="left", validate="one_to_one")
    staff = pd.to_numeric(controls_sample.staff_count, errors="coerce")
    controls_sample["ln_staff"] = np.log1p(staff.where(staff.ge(0)))
    valid_std = prepared[sample.INDUSTRY].str.fullmatch(r"[A-Z]\d{2}").fillna(False)
    valid_existing = prepared.industry_code.astype("string").str.fullmatch(r"[A-Z]\d{2}").fillna(False)
    scenarios = [
        ("Hybrid source hierarchy", prepared.loc[valid_std].copy(), sample.INDUSTRY),
        ("Primary financial-library source only", prepared.loc[valid_std & prepared.industry_code_std_source.eq("financial_library")].copy(), sample.INDUSTRY),
        ("Existing-panel source only", prepared.loc[valid_existing].copy(), "industry_code"),
        ("Two-source agreement sample", prepared.loc[valid_std & valid_existing & prepared.industry_code_std_source.eq("financial_library")
            & prepared[sample.INDUSTRY].eq(prepared.industry_code)].copy(), sample.INDUSTRY),
    ]
    source_rows = []
    for label, frame, code in scenarios:
        rebuilt, threshold = sources.build_gtos(frame, code)
        source_rows.append({"Industry-code specification": label, "Frontier threshold": threshold,
                            **sources.decomposition(frame, code), **sources.accounting_models(rebuilt)})
    valid = prepared.loc[valid_std].copy()
    sparse = []
    for outcome, label in [(sample.COUNT, "Patent count"), (sample.ENTRY, "Patent entry indicator"),
                           (sample.LOG_COUNT, "Log(1 + patent count)")]:
        cells = sparse_cells.build_supported_cells(valid, outcome)
        for threshold in sparse_cells.THRESHOLDS:
            for benchmark in ["Main Table 1 compatible", "Retained overlap cells"]:
                sparse.append(sparse_cells.decompose(valid, cells, outcome, label, threshold, benchmark))
    for field in baseline_models.RAW_CONTROLS:
        prepared[field + "_w"] = controls.winsorize(prepared[field])
    return {"rd_triangulation": rd_tables, "investment_triangulation": green_tables,
            "alternative_controls": controls.run_candidate_control_sensitivity(controls_sample),
            "industry_fe_role": controls.run_industry_fe_role_sensitivity(controls_sample),
            "industry_sources": source_rows, "standardized_baseline": sources.standardized_baseline_models(prepared),
            "sparse_cells": sparse, "industry_baseline": baseline_models.run_new_models(prepared)}


def restricted_subsidy_flags(path, ledger_path):
    from research_release import subsidy_titles as titles
    from research_release import sample
    raw, _ = titles.read_source(Path(path))
    annual = raw.loc[raw.year.between(2007, 2024) & raw.report_type.eq(1)].copy()
    annual = annual.loc[annual.Item.notna() & annual.Item.astype(str).str.strip().ne("")].copy()
    saved = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    lookup = annual.set_index("source_excel_row")
    labels = {}
    for item in saved["title_ledger"]:
        ids = item["source_excel_rows"]
        if len(ids) != item["records"] or set(ids) & labels.keys() or not lookup.loc[ids, "Item"].eq(item["title"]).all():
            raise ValueError("Restricted subsidy labels do not match the source rows.")
        labels.update({row: item["category"] for row in ids})
    initial = (annual.Item.astype(str).str.contains(saved["rules"]["raw_rd_pattern"], regex=True)
               & annual.Item.astype(str).str.contains(saved["rules"]["raw_green_pattern"], regex=True))
    if set(annual.loc[initial, "source_excel_row"]) != set(labels):
        raise ValueError("Restricted subsidy candidate universe changed.")
    annual["category"] = annual.source_excel_row.map(labels).fillna("outside_candidate_screen")
    annual["clear_record"] = annual.category.eq("clear_title").astype(int)
    annual["uncertain_record"] = annual.category.eq("uncertain").astype(int)
    flags = annual.groupby([sample.FIRM, "year"], sort=True).agg(
        recorded_title_rows=("source_excel_row", "size"), clear_title_rows=("clear_record", "sum"),
        uncertain_title_rows=("uncertain_record", "sum")).reset_index()
    flags["has_clear_title"] = flags.clear_title_rows.gt(0).astype(int)
    flags["has_uncertain_title"] = flags.uncertain_title_rows.gt(0).astype(int)
    return flags


def restricted_project_records(path, ledger_path):
    import pandas as pd
    from research_release import project_titles as titles
    raw = titles.read_export(Path(path))
    raw["InvestSourceCode"] = pd.to_numeric(raw.InvestSourceCode, errors="raise")
    annual = raw.loc[raw.InvestSourceCode.isin([2, 5]) & raw.year.between(2007, 2024)].copy()
    annual["normalized_title"] = annual.Item.map(titles.normalise)
    saved = json.loads(Path(ledger_path).read_text(encoding="utf-8"))["analysis"]
    labels, adjustments = {}, {}
    lookup = annual.set_index("source_excel_row")
    for item in saved["title_ledger"]:
        ids = item["source_excel_rows"]
        part = lookup.loc[ids]
        if (len(ids) != item["records"] or set(ids) & labels.keys()
                or not part.InvestSourceCode.eq(item["source_code"]).all()
                or not part.normalized_title.eq(item["title"]).all()):
            raise ValueError("Restricted project labels do not match the source rows.")
        labels.update({row: item["category"] for row in ids})
        adjustments.update({row: bool(item["accounting_adjustment_title"]) for row in ids})
    if set(labels) != set(annual.source_excel_row):
        raise ValueError("Restricted project ledger does not cover the selected source.")
    annual["clear_record"] = annual.source_excel_row.map(labels).eq("clear_green_rd_title").astype(int)
    annual["uncertain_record"] = annual.source_excel_row.map(labels).eq("uncertain_or_mixed_title").astype(int)
    annual["accounting_adjustment_title"] = annual.source_excel_row.map(adjustments)
    return annual


def bootstrap(common):
    import numpy as np
    from research_release import accounting as ac, sample
    design = ac.build_design(common)
    y = common[list(ac.OUTCOMES.values())].to_numpy(float)
    clusters = ac.cluster_statistics(design, y, common[sample.FIRM])
    count = len(clusters["labels"])
    point, diagnostics = ac.gram_metrics(design, clusters, np.ones(count, dtype=int))
    ac.compare_metrics(point, ac.direct_metrics(design, y, np.ones(len(common)), original=True))
    rng = np.random.default_rng(ac.SEED)
    draws, failures = [], []
    for first in range(0, ac.REPETITIONS, 100):
        weights = np.array([np.bincount(rng.integers(0, count, size=count), minlength=count)
                            for _ in range(min(100, ac.REPETITIONS - first))])
        totals = weights @ clusters["statistics"]
        for offset, row in enumerate(weights):
            draw_id = first + offset + 1
            try:
                values, audit = ac.gram_metrics(design, clusters, row, totals[offset])
            except (np.linalg.LinAlgError, ValueError, ArithmeticError) as error:
                failures.append({"draw": draw_id, "reason": str(error), "type": type(error).__name__})
                continue
            if draw_id <= 3:
                ac.compare_metrics(values, ac.direct_metrics(design, y, row[clusters["inverse"]], original=True))
            draws.append({"draw": draw_id, **audit, "values": {k: v.tolist() for k, v in values.items()}})
    if len(draws) < 2:
        raise RuntimeError("Fewer than two estimable draws; no interval can be reported.")
    return {"seed": ac.SEED, "requested_repetitions": ac.REPETITIONS,
            "point_estimates": point, "point_diagnostics": diagnostics,
            "summary": ac.summarize(point, draws), "unsuccessful_draws": failures,
            "replicates": draws, "conditional_on_stored_gtos_and_sample": True}


def sample_bridge(panel, masks):
    import numpy as np
    import pandas as pd
    from research_release import sample, accounting, influence, sample_bridge as bridge
    eligible_mask = masks["Static controls and fields complete"] & panel.year.gt(2007)
    common_mask = masks["Common estimation sample"]
    missing = panel[[*sample.LAGGED, sample.FI]].isna().any(axis=1)
    if (common_mask & ~eligible_mask).any() or not (eligible_mask & ~missing).equals(common_mask):
        raise ValueError("The sample difference must be exactly GTOS availability after 2007.")
    eligible = panel.loc[eligible_mask].copy()
    common = panel.loc[common_mask].copy()
    excluded = panel.loc[eligible_mask & ~common_mask].copy()
    a, b = bridge.control_design(eligible), bridge.control_design(common)
    full = accounting.build_design(common)
    c = pd.DataFrame(full["original_x"], index=common.index, columns=full["columns"])
    if list(a.columns) != list(b.columns):
        raise ValueError("Year or province categories differ across comparison samples.")
    pd.testing.assert_frame_equal(a.loc[common.index], b)
    pd.testing.assert_frame_equal(c[b.columns], b)
    estimates, diagnostics = {}, {}
    for label, design, frame in [("A", a, eligible), ("B", b, common), ("C", c, common)]:
        estimates[label], diagnostics[label] = bridge.fit(design, frame[list(accounting.OUTCOMES.values())])
    rows = []
    for j, outcome in enumerate(accounting.OUTCOMES):
        beta_a, beta_b, beta_c = (float(estimates[label][j]) for label in ["A", "B", "C"])
        error = (beta_b - beta_a) + (beta_c - beta_b) - (beta_c - beta_a)
        np.testing.assert_allclose(error, 0., atol=1e-12, rtol=0)
        rows.append({"outcome": outcome, "A_eligible_control": beta_a, "B_common_control": beta_b,
                     "C_common_full": beta_c, "B_minus_A_sample_restriction": beta_b - beta_a,
                     "C_minus_B_add_gtos": beta_c - beta_b, "C_minus_A_total_change": beta_c - beta_a,
                     "path_identity_error": error})
    lag_missing = excluded[sample.LAGGED].isna().any(axis=1)
    fi_missing = excluded[sample.FI].isna()
    missingness = {"lagged_OI_OD_only": int((lag_missing & ~fi_missing).sum()),
                   "FI_only": int((~lag_missing & fi_missing).sum()),
                   "both_lagged_OI_OD_and_FI": int((lag_missing & fi_missing).sum())}
    if sum(missingness.values()) != len(excluded):
        raise AssertionError("GTOS missingness categories do not exhaust the excluded sample.")
    return {"samples": {name: influence.sample_info(frame) for name, frame in
                         [("A_eligible", eligible), ("B_C_common", common), ("A_minus_common", excluded)]},
            "excluded_missingness": missingness, "estimates": rows, "fit_diagnostics": diagnostics,
            "interpretation": "Path-specific point comparison, not a selection correction or causal decomposition."}


def wild_cluster_cases(common, own_history, guide_samples):
    import numpy as np
    from research_release import sample, rd, history
    full_rhs = rd.BASE + " + " + " + ".join(rd.PATENT)
    fe_rhs = rd.BASE + " + C(industry_code_std) + " + " + ".join(sample.LAGGED)
    cases = []
    for name, rhs in [("Full accounting", full_rhs), ("Industry FE plus OI/OD", fe_rhs)]:
        for outcome, field in rd.OUTCOMES.items():
            cases.append((name, outcome, common, rhs, field))
    positive = common.loc[common[sample.COUNT].gt(0)].copy()
    positive["log_positive_count"] = np.log(positive[sample.COUNT])
    cases.append(("Full positive-count intensity", "Log(count) given count > 0", positive,
                  full_rhs, "log_positive_count"))
    cases.append(("Own history + patent GTOS", "Patent count", own_history,
                  full_rhs + " + " + history.HISTORY, sample.COUNT))
    for role in ["process_rd", "supplier"]:
        frame = guide_samples[role]
        columns = [c for c in frame if c.startswith("profile__")]
        rhs = rd.BASE + (" + " + " + ".join(columns) if columns else "")
        for outcome, field in rd.OUTCOMES.items():
            cases.append(("Guide profiles: " + role, outcome, frame, rhs, field))
    return cases


def industry_wild_cluster(common, own_history, guide_samples):
    import numpy as np
    import patsy
    from scipy.stats import t
    from research_release import sample, rd, wild_cluster as wild
    rows = []
    for index, (name, outcome, frame, rhs, field) in enumerate(
            wild_cluster_cases(common, own_history, guide_samples)):
        matrix = patsy.dmatrix(rhs, frame, return_type="dataframe", NA_action="raise")
        if not matrix.index.equals(frame.index):
            raise ValueError("Design construction changed sample or row alignment.")
        x, y = matrix.to_numpy(float), frame[field].to_numpy(float)
        target = matrix.columns.get_loc(sample.AGRI)
        fit = wild.prepare(x, y, frame[sample.INDUSTRY], target)
        g = len(fit["levels"])
        row = {"model": name, "outcome": outcome, "n": len(frame), "industries": g,
               "firms": int(frame[sample.FIRM].nunique()), "sample_sha256": rd.fingerprint(frame),
               "rhs": rhs, "design_columns": list(matrix.columns),
               "coefficient": fit["coefficient"], "industry_se": fit["industry_se"],
               "industry_cr1_t": fit["t"], "industry_cr1_p_t": float(2 * t.sf(abs(fit["t"]), g - 1)),
               "wild_cluster": []}
        for weight_index, distribution in enumerate(["Rademacher", "Webb"]):
            stream = 2 * index + weight_index
            weights, exhaustive = wild.draw_weights(g, distribution, stream)
            values = wild.bootstrap_t(fit, weights)
            selected = np.linspace(0, len(values) - 1, 3, dtype=int)
            error = wild.direct_check(x, y, fit, target, weights[:, selected])
            exceedances = int(np.sum(np.abs(values) >= abs(fit["t"]) - 1e-10))
            row["wild_cluster"].append({"weights": distribution, "seed_stream": stream,
                "draws": len(values), "exhaustive_sign_distribution": exhaustive,
                "exceedances": exceedances, "p_value": exceedances / len(values),
                "invalid_draws": 0, "direct_refits_checked": 3,
                "max_t_error_against_direct_refit": error})
        rows.append(row)
    return {"plan": wild.plan(), "results": rows}


def analysis(args):
    import numpy as np
    import pandas as pd
    from research_release import sample, baseline, structural_utils as utils, diagnostics, decomposition
    panel, common, masks, excluded = canonical_sample(args.panel)
    mode = args.command
    if mode == "penalties":
        from research_release import penalties
        return penalties.estimate(panel, common)
    if mode == "sample-bridge":
        return sample_bridge(panel, masks)
    if mode == "wild-cluster":
        from research_release import history, rd
        merged, _, _, _ = rd_input(panel, args.rd)
        enriched = history.add_calendar_history(merged)
        chosen = enriched.loc[masks["Common estimation sample"] & enriched[rd.RD_MEAN].notna()
                              & enriched[history.HISTORY].notna()].copy()
        frames, _ = guide_frames(common)
        return industry_wild_cluster(common, chosen, frames)
    if mode == "supplementary":
        return supplementary(panel, common, args.rd, args.legacy_investment)
    if mode == "descriptive":
        from research_release import descriptive_alignment
        valid = panel.loc[panel[sample.INDUSTRY].str.fullmatch(r"[A-Z]\d{2}")].copy()
        return {"tail_decompositions": descriptive_alignment.tail_decompositions(valid),
                "latest_year_decompositions": descriptive_alignment.latest_year_decompositions(valid)}
    if mode == "sample":
        return {"flow": sample.sample_flow(panel, masks), "first_failures": sample.first_failures(panel, masks),
                "comparisons": sample.compare_samples(panel, masks["Common estimation sample"], excluded),
                "clusters": sample.industry_clusters(panel, masks["Common estimation sample"])}
    if mode == "guide":
        from research_release import guides
        frames, reports = guide_frames(common)
        return {role: {"support": reports[role], "models": guides.compare(frame, role)} for role, frame in frames.items()}
    if mode in ["rd", "history"]:
        from research_release import rd, history
        merged, selected, saved, audit = rd_input(panel, args.rd)
        if mode == "rd":
            return {"construction": audit, **saved, "models": rd.compare_models(selected)}
        enriched = history.add_calendar_history(merged)
        chosen = enriched.loc[masks["Common estimation sample"] & enriched[rd.RD_MEAN].notna()
                              & enriched[history.HISTORY].notna()].copy()
        return history.estimate(chosen, saved)
    if mode == "investment":
        from research_release import environment, investment
        total = environment.read_export(Path(args.investment_total))
        result = {}
        for label, definition in investment.MEASURES.items():
            groups, audit = investment.select_measure(total, definition["field"])
            merged, cells, construction = environment.environment(panel, groups, definition["variable"])
            result[label] = {"measure_audit": audit, "construction": construction,
                             "comparison": investment.compare(merged, definition["variable"], label)}
        return result
    if mode in ["subsidy", "projects"]:
        if mode == "subsidy":
            from research_release import subsidy
            flags = restricted_subsidy_flags(args.subsidy, args.labels)
            merged, cells, construction = subsidy.environment(panel, flags)
            from research_release import investment
            return {"construction": construction, "audit": subsidy.audit(merged),
                    "models": investment.compare(merged, subsidy.VARIABLE,
                        "Title-defined green R&D subsidy share among recorded non-agricultural firms")}
        from research_release import projects, investment
        records = restricted_project_records(args.investment_detail, args.labels)
        result = {}
        for code, definition in projects.MEASURES.items():
            flags, audit = projects.firm_flags(records, code)
            merged, cells, construction = projects.environment(panel, flags, definition["variable"])
            result[str(code)] = {"source_summary": audit, "construction": construction,
                                "audit": projects.audit(merged, definition["variable"]),
                                "models": investment.compare(merged, definition["variable"], definition["label"])}
        return result
    if mode == "bootstrap":
        return bootstrap(common)
    if mode == "nb2":
        from research_release import nb2
        y = common[sample.COUNT]
        alpha = max((float(y.var(ddof=1)) - float(y.mean())) / float(y.mean()) ** 2, 1e-6)
        result = {}
        for name, design in nb2.designs(common).items():
            fitted, fixed = nb2.fixed_fit(common, design, alpha)
            result[name] = {"fixed": fixed, "joint": nb2.joint_fit(common, design, fitted, alpha)}
        return result
    if mode in ["influence", "shared"]:
        from research_release import influence, shared_industries as shared
        coefficients, audit = influence.fit_case(common)
        if mode == "shared":
            retained, report = shared.restrict_to_shared_industries(common)
            rows, diagnostics_ = influence.fit_case(retained)
            return {"restriction": report, "results": rows, "comparison": shared.comparisons(coefficients, rows),
                    "diagnostics": diagnostics_}
        cases = []
        for industry in sorted(common[sample.INDUSTRY].unique()):
            retained = common.loc[common[sample.INDUSTRY].ne(industry)].copy()
            rows, diagnostics_ = influence.fit_case(retained)
            cases.append({"dropped_industry": industry, "sample": influence.sample_info(retained),
                          "results": rows, "diagnostics": diagnostics_})
        return {"baseline": coefficients, "baseline_diagnostics": audit, "cases": cases}
    if mode == "reassignment":
        return baseline.reassignment_checks(common, utils)
    if mode == "tails":
        regime, regime_cells = diagnostics.build_core_gtos_regime(common)
        return {"tail_sensitivity": diagnostics.high_regime_tail_sensitivity(regime),
                "alternative_definitions": diagnostics.alternative_high_definitions(regime),
                "gap_bootstrap": diagnostics.firm_cluster_bootstrap_gap(
                    regime, "high_frontier_opportunity",
                    seed=20260726 + list(diagnostics.REGIME_LABELS).index("high_frontier_opportunity"))}
    if mode == "core":
        from research_release import count_models, support, reference_decomposition as reference, sign_diagnostics
        validate_core_fields(panel)
        cells = panel.drop_duplicates([sample.INDUSTRY, "year"])
        valid = panel.loc[panel[sample.INDUSTRY].str.fullmatch(r"[A-Z]\d{2}")].copy()
        valid["broad_industry_sector"] = valid[sample.INDUSTRY].astype(str).str.slice(0, 1)
        return {"nested_coefficients": baseline.nested_coefficients(panel, utils, "Strict-calendar corrected construction"),
                "reference_decomposition": reference.build_decomposition_diagnostics(valid),
                "exposures": baseline.exposure_tables(panel, cells, common, "Strict-calendar corrected construction")[:2],
                "partitions": baseline.decomposition_tables(common, decomposition, "Strict-calendar corrected construction"),
                "two_part": count_models.run_two_part(panel), "ppml": count_models.run_ppml(panel),
                "fixed_nb2": count_models.run_negative_binomial(panel),
                "overlap": baseline.support_summary(panel, support, "Strict-calendar corrected construction")[0],
                "clustering": diagnostics.clustering_sensitivity(common, utils),
                "post2015_clustering": diagnostics.post2015_clustering_sensitivity(common, utils),
                "late_window": diagnostics.frontier_2013_sensitivity(common, utils),
                "leave_one_out": baseline.leave_one_out_check(panel, utils),
                "condition_indices": sign_diagnostics.condition_index_table(
                    sign_diagnostics.residualized_dataframe(common, [sample.AGRI, *utils.CORE_GTOS_VARS]), "Background-adjusted"),
                "vifs": sign_diagnostics.vif_from_matrix(
                    sign_diagnostics.residualized_dataframe(common, [sample.AGRI, *utils.CORE_GTOS_VARS]), "Background-adjusted")}
    raise ValueError(f"Unknown analysis: {mode}")


def serializable(value):
    import numpy as np
    import pandas as pd
    if isinstance(value, pd.DataFrame):
        if value.index.name is not None and value.index.name not in value.columns:
            value = value.reset_index()
        return serializable(value.to_dict("records"))
    if isinstance(value, pd.Series):
        return serializable(value.to_dict())
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Import and synthetic checks only; no licensed data or paper models.")
    validation = sub.add_parser("validate-inputs", help="Read-only core-field, sample, GTOS and guide checks; no estimates.")
    validation.add_argument("--panel", required=True)
    for command in ["sample", "core", "guide", "rd", "history", "investment", "subsidy", "projects",
                    "bootstrap", "nb2", "influence", "shared", "reassignment", "tails", "supplementary", "descriptive",
                    "sample-bridge", "wild-cluster", "penalties"]:
        p = sub.add_parser(command)
        p.add_argument("--panel", required=True)
        p.add_argument("--output", required=True, help="A new JSON file OUTSIDE the release directory.")
        if command in ["rd", "history", "supplementary", "wild-cluster"]:
            p.add_argument("--rd", required=True)
        if command == "supplementary":
            p.add_argument("--legacy-investment", required=True)
        if command == "investment":
            p.add_argument("--investment-total", required=True)
        if command in ["subsidy", "projects"]:
            p.add_argument("--labels", required=True, help="Restricted frozen classification ledger; not included.")
            p.add_argument("--subsidy" if command == "subsidy" else "--investment-detail", required=True)
    args = parser.parse_args()
    if args.command == "check":
        check()
        return
    if args.command == "validate-inputs":
        print(json.dumps(serializable(validate_inputs(args.panel)), ensure_ascii=False, indent=2, allow_nan=False))
        return
    output = Path(args.output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Keep local, potentially restricted outputs outside the release directory.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    inputs = [Path(v).resolve() for k, v in vars(args).items() if k not in ["command", "output"] and v]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    result = analysis(args)
    if before != {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}:
        raise RuntimeError("An input changed during the analysis.")
    text = json.dumps(serializable(result), ensure_ascii=False, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(f"Saved {args.command} results to {output}; input files were not modified.")


if __name__ == "__main__":
    main()
