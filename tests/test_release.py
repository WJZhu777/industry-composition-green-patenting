"""Small numerical checks; no empirical data or paper estimates are used."""

import ast
import builtins
import copy
import csv
import dis
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location("release_runner_test", ROOT / "run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_high_regime_bootstrap_uses_original_group_seed(self):
        from research_release import diagnostics
        from argparse import Namespace
        frame = pd.DataFrame({"synthetic": [1]})
        with patch.object(self.runner, "canonical_sample", return_value=(frame, frame, {}, None)), \
             patch.object(diagnostics, "build_core_gtos_regime", return_value=(frame, pd.DataFrame())), \
             patch.object(diagnostics, "high_regime_tail_sensitivity", return_value=[]), \
             patch.object(diagnostics, "alternative_high_definitions", return_value=[]), \
             patch.object(diagnostics, "firm_cluster_bootstrap_gap", return_value={}) as bootstrap:
            self.runner.analysis(Namespace(command="tails", panel="unused"))
            bootstrap.assert_called_once_with(frame, "high_frontier_opportunity", seed=20260728)

    def test_green_matching_returns_window_then_matched_panel(self):
        from research_release import investment_triangulation as green
        raw = pd.DataFrame({"证券代码": ["100001"], "year": [2008], "绿色总投资": [1.],
                            "资本化绿色投资额": [1.]})
        panel = pd.DataFrame({"证券代码": ["100001"], "year": [2008], "是否涉农": [1],
                              "industry_code_std": ["C26"], "industry_name_std": ["Synthetic"]})
        window, matched = green.build_matched_data(raw, panel)
        self.assertNotIn("是否涉农", window)
        self.assertEqual(matched["是否涉农"].tolist(), [1])


class DiagnosticCacheTests(unittest.TestCase):
    def setUp(self):
        from research_release import sign_diagnostics
        self.diagnostic = sign_diagnostics
        self.frame = pd.DataFrame({"marker": [1.]})

    def fit(self, formula, frame):
        terms = [self.diagnostic.AGRI, *self.diagnostic.CORE_GTOS_VARS]
        return SimpleNamespace(params=pd.Series(float(frame.marker.iloc[0]), index=terms),
                               bse=pd.Series(0.1, index=terms), pvalues=pd.Series(0.1, index=terms),
                               nobs=len(frame), rsquared=0.)

    def coefficient(self, frame):
        return self.diagnostic.fit_specification(
            frame, "green_patent_application_count", "Synthetic", self.diagnostic.SPARSE_BASE
        )["Agricultural coefficient"]

    def test_separate_frames_do_not_share_a_fit(self):
        with patch.object(self.diagnostic, "clustered_ols", side_effect=self.fit) as fit:
            self.assertEqual(self.coefficient(self.frame), 1.)
            self.assertEqual(self.coefficient(self.frame.assign(marker=2.)), 2.)
            self.assertEqual(fit.call_count, 2)

    def test_revising_a_frame_does_not_reuse_its_old_fit(self):
        with patch.object(self.diagnostic, "clustered_ols", side_effect=self.fit) as fit:
            self.assertEqual(self.coefficient(self.frame), 1.)
            self.frame.loc[:, "marker"] = 3.
            self.assertEqual(self.coefficient(self.frame), 3.)
            self.assertEqual(fit.call_count, 2)

    def test_order_checks_reuse_fits_only_within_the_call(self):
        with patch.object(self.diagnostic, "clustered_ols", side_effect=self.fit) as fit:
            first = self.diagnostic.build_order_sensitivity(self.frame)
            formulas = [call.args[0] for call in fit.call_args_list]
            self.assertEqual(len(formulas), len(set(formulas)))
            self.assertLess(len(formulas), len(first))
            fit.reset_mock()
            second = self.diagnostic.build_order_sensitivity(self.frame.assign(marker=2.))
            self.assertEqual(fit.call_count, len(formulas))
        self.assertTrue(first["Agricultural coefficient"].eq(1.).all())
        self.assertTrue(second["Agricultural coefficient"].eq(2.).all())

    def test_fwl_direct_fit_uses_the_current_sample(self):
        d = self.diagnostic
        rng = np.random.default_rng(887)
        frame = pd.DataFrame({"证券代码": np.repeat([f"{i:06d}" for i in range(40)], 3),
                              "year": np.tile([2020, 2021, 2022], 40),
                              d.AGRI: np.repeat(np.arange(40) % 2, 3),
                              "province": np.repeat(rng.choice(["A", "B", "C"], 40), 3)})
        for name in [*d.CORE_FIRM_CONTROLS, *d.CORE_GTOS_VARS]:
            frame[name] = rng.normal(size=len(frame))
        count, log_count, entry = d.PATENT_OUTCOMES
        frame[count] = rng.poisson(1.2, size=len(frame))
        frame[log_count] = np.log1p(frame[count])
        frame[entry] = frame[count].gt(0).astype(int)
        first = d.build_fwl_diagnostics(frame)
        frame[count] += 2 * frame[d.AGRI]
        frame[log_count] = np.log1p(frame[count])
        frame[entry] = frame[count].gt(0).astype(int)
        second = d.build_fwl_diagnostics(frame)
        for results in [first, second]:
            self.assertTrue(np.isfinite(results["Clustered standard error"]).all())
            np.testing.assert_allclose(results["FWL agricultural coefficient"],
                                       results["Direct agricultural coefficient"], rtol=0, atol=1e-10)
        count_rows = first.Outcome.eq(d.OUTCOME_LABELS[count])
        np.testing.assert_allclose(
            second.loc[count_rows, "Direct agricultural coefficient"]
            - first.loc[count_rows, "Direct agricultural coefficient"], 2., rtol=0, atol=1e-10
        )


class ReleaseFilesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def files(self, names):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic fixture", encoding="utf-8")
            return {p.relative_to(root).as_posix() for p in self.runner.release_files(root)}

    def test_root_tool_directories_are_not_release_content(self):
        names = ["run.py", ".idea/workspace.xml", "venv/Lib/module.py",
                 ".venv/pyvenv.cfg", ".git/config", "code/__pycache__/module.pyc",
                 "release_manifest.json"]
        self.assertEqual(self.files(names), {"run.py"})

    def test_unlisted_data_is_not_hidden_by_gitignore_rules(self):
        names = ["data/private/panel.csv", "results/local/estimates.json", "export.xlsx", ".env"]
        self.assertEqual(self.files(names), set(names))

    def test_nested_ide_name_and_manifest_are_not_excluded(self):
        names = ["data/.idea/panel.csv", "data/venv/panel.csv", "data/release_manifest.json"]
        self.assertEqual(self.files(names), set(names))


class CountConvergenceTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"证券代码": ["100001", "100002", "100003"],
                                   "green_patent_application_count": [0, 1, 4]})
        self.result = SimpleNamespace(converged=True, params=np.array([0.2]),
                                      bse=np.array([0.1]), pvalues=np.array([0.04]), llf=-5.)

    def test_converged_fit_keeps_original_result_and_dispersion(self):
        from research_release import count_models
        before = self.frame.copy(deep=True)
        expected = max((self.frame.green_patent_application_count.var()
                        - self.frame.green_patent_application_count.mean())
                       / self.frame.green_patent_application_count.mean() ** 2, 1e-6)
        with patch.object(count_models.smf, "glm") as model:
            model.return_value.fit.return_value = self.result
            actual, alpha = count_models.fit_nb_glm("green_patent_application_count ~ 1", self.frame)
            self.assertIs(actual, self.result)
            self.assertEqual(alpha, expected)
            model.return_value.fit.assert_called_once()
        pd.testing.assert_frame_equal(self.frame, before)

    def test_unconverged_fit_is_rejected(self):
        from research_release import count_models
        self.result.converged = False
        with patch.object(count_models.smf, "glm") as model:
            model.return_value.fit.return_value = self.result
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                count_models.fit_nb_glm("green_patent_application_count ~ 1", self.frame)

    def test_nonfinite_fit_is_rejected(self):
        from research_release import count_models
        for field in ["params", "bse", "pvalues", "llf"]:
            with self.subTest(field=field):
                result = SimpleNamespace(**vars(self.result))
                setattr(result, field, np.array([np.nan]) if field != "llf" else -np.inf)
                with patch.object(count_models.smf, "glm") as model:
                    model.return_value.fit.return_value = result
                    with self.assertRaisesRegex(ArithmeticError, "nonfinite"):
                        count_models.fit_nb_glm("green_patent_application_count ~ 1", self.frame)

    def test_reported_convergence_reads_the_result(self):
        from research_release import count_models
        for converged in [True, False]:
            with self.subTest(converged=converged), \
                 patch.object(count_models, "common_sample", return_value=self.frame), \
                 patch.object(count_models, "fit_nb_glm", return_value=(SimpleNamespace(converged=converged, nobs=len(self.frame)), 1.)), \
                 patch.object(count_models, "extract_rows", side_effect=lambda *args: [{"coef": 0.2, "se": 0.1}]):
                rows, _ = count_models.run_negative_binomial(self.frame)
                self.assertEqual(rows.converged.tolist(), [converged] * 4)


class NumericalTests(unittest.TestCase):
    def test_final_descriptive_caps_are_fixed(self):
        from research_release import descriptive_alignment
        frame = pd.DataFrame({"证券代码": ["100001", "100002"] * 3,
                              "year": [2022, 2022, 2023, 2023, 2024, 2024],
                              "是否涉农": [1, 0] * 3, "industry_code_std": ["C26"] * 6,
                              "green_patent_application_count": [2, 100, 3, 200, 4, 300]})
        frame["log1p_green_patent_application_count"] = np.log1p(frame.green_patent_application_count)
        frame["has_green_patent_application"] = frame.green_patent_application_count.gt(0).astype(int)
        original = frame.copy(deep=True)
        self.assertEqual(descriptive_alignment.tail_decompositions(frame).Cap.tolist(), [16., 32., 83.])
        latest = descriptive_alignment.latest_year_decompositions(frame)
        self.assertEqual(len(latest), 9)
        self.assertEqual(latest.window.nunique(), 3)
        pd.testing.assert_frame_equal(frame, original)

    def test_sample_rules(self):
        from research_release import sample
        sample.self_test()

    def test_rd_rules(self):
        from research_release import rd
        rd.self_test()

    def test_accounting_identities(self):
        from research_release import accounting
        accounting.self_test()

    def test_nb2_likelihood(self):
        from research_release import nb2
        nb2.self_test()

    def test_history_calendar_lags(self):
        from research_release import history
        history.self_test()

    def test_investment_missingness(self):
        from research_release import investment
        investment.self_test()

    def test_industry_influence_algebra(self):
        from research_release import influence
        influence.self_test()

    def test_shared_industry_restriction(self):
        from research_release import shared_industries
        shared_industries.self_test()

    def test_subsidy_environment(self):
        from research_release import subsidy
        subsidy.self_test()

    def test_project_environment(self):
        from research_release import projects
        projects.self_test()

    def test_sector_a_support(self):
        from research_release import sector_a
        sector_a.self_test()

    def test_guide_column_space(self):
        from research_release import guide_design
        guide_design.self_test()

    def test_guide_cluster_covariance(self):
        from research_release import guides
        guides.self_test()

    def test_public_guide_coding(self):
        from research_release import guide_design
        data = json.loads((ROOT / "data/public/guide_coding.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["revised_branch_ledger"]), 97)
        self.assertEqual(len(data["technology_families"]), 31)
        for role in guide_design.ROLES:
            table, records = guide_design.descriptors(data, role)
            self.assertFalse(table.empty)
            self.assertFalse(table[guide_design.INDUSTRY].duplicated().any())
            self.assertEqual(len(records), len(table))
        original = data["industry_role_technology_matrix"]
        for row in original:
            for role in guide_design.ROLES:
                if row["roles"][role]["status"] == "unknown_or_unlinked_role":
                    self.assertTrue(all(v is None for v in row["roles"][role]["function_flags"].values()))

    def test_missing_private_overrides_fail_closed(self):
        from research_release import project_titles, subsidy_titles
        with self.assertRaisesRegex(RuntimeError, "restricted"):
            project_titles.classify("synthetic title")
        with self.assertRaisesRegex(RuntimeError, "restricted"):
            subsidy_titles.classify_title("synthetic title")

    def test_strict_calendar_gtos(self):
        from research_release import industry_sources as gtos
        frame = pd.DataFrame({"证券代码": ["100001", "100002", "100003", "100001", "100002", "100003"],
                              "year": [2007, 2007, 2007, 2009, 2009, 2009],
                              "是否涉农": [0, 0, 1, 0, 0, 1], "industry_code_std": ["C26"] * 6,
                              "green_patent_application_count": [2, 0, 99, 4, 0, 99],
                              "has_green_patent_application": [1, 0, 1, 1, 0, 1]})
        result, threshold = gtos.build_gtos(frame, "industry_code_std")
        self.assertTrue(result.loc[result.year.eq(2009), "opportunity_intensity"].isna().all())


class GuideDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        public = ROOT / "data/public"
        cls.data = json.loads((public / "guide_coding.json").read_text(encoding="utf-8"))
        cls.sources = json.loads((public / "official_sources.json").read_text(encoding="utf-8"))

    def test_source_registry_and_page_bounds(self):
        sources = {s["source_id"]: s for s in self.sources if "source_id" in s}
        self.assertEqual(len(sources), 7)
        self.assertEqual(sum(s["pdf_pages"] for s in sources.values()), 117)
        for s in self.sources:
            self.assertTrue(s["document_title"])
            self.assertRegex(s["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(s["url"].startswith("https://"))
        standards = [s for s in self.sources if "source_record" in s]
        self.assertEqual([s["source_record"] for s in standards], ["GB/T 4754-2011"])
        for row in [*self.data["direction_screening"], *self.data["revised_branch_ledger"]]:
            self.assertEqual(row["source_id"], row["direction_id"].split(":")[0])
            self.assertGreaterEqual(row["page_start_1based"], 1)
            self.assertLessEqual(row["page_start_1based"], sources[row["source_id"]]["pdf_pages"])

    def test_complete_screening_and_branch_membership(self):
        directions = {r["direction_id"]: r for r in self.data["direction_screening"]}
        branches = {r["branch_id"]: r for r in self.data["revised_branch_ledger"]}
        self.assertEqual(len(directions), len(self.data["direction_screening"]))
        self.assertEqual(len(directions), 139)
        self.assertEqual(len(branches), len(self.data["revised_branch_ledger"]))
        self.assertEqual(len(branches), 97)
        statuses = [r["status"] for r in directions.values()]
        self.assertEqual(statuses.count("has_qualified_branch"), 75)
        self.assertEqual(statuses.count("outside_narrow_technical_scope"), 59)
        for direction_id, row in directions.items():
            expected = sorted(b["branch_id"] for b in branches.values() if b["direction_id"] == direction_id)
            self.assertEqual(sorted(row["branch_ids"]), expected)
            self.assertEqual(row["status"] == "has_qualified_branch", bool(expected))
            self.assertTrue(row["title"])
        for b in branches.values():
            self.assertEqual(b["direction_title"], directions[b["direction_id"]]["title"])
            self.assertTrue(b["literal_source_anchors"])
            self.assertTrue(b["scope_and_role_conditions"])
            self.assertTrue(set(b["functions"]) <= set(self.data["function_definitions"]))
            self.assertEqual(set(b["conditional_role_codes"]), set(self.data["role_definitions"]))
            for codes in b["conditional_role_codes"].values():
                self.assertEqual(len(codes), len(set(codes)))
                for code in codes:
                    self.assertRegex(code, r"^[A-S][0-9]{2}$")
            for code in b["standard_activity_codes"]:
                self.assertRegex(str(code), r"^[0-9]{4}$")
        self.assertEqual(len(self.data["pending_reviews"]), 9)
        for row in self.data["pending_reviews"]:
            self.assertIn(row["direction_id"], directions)

    def test_family_membership(self):
        branches = self.data["revised_branch_ledger"]
        families = self.data["technology_families"]
        self.assertEqual(set(families), {b["technology_family"] for b in branches})
        for family, value in families.items():
            expected = sorted(b["branch_id"] for b in branches if b["technology_family"] == family)
            self.assertEqual(value["source_branch_ids"], expected)
            self.assertEqual(value["branch_records_not_technology_count"], len(expected))

    def test_all_roles_reconstruct_from_branches(self):
        branches = self.data["revised_branch_ledger"]
        matrix = self.data["industry_role_technology_matrix"]
        expected_codes = {c for b in branches for codes in b["conditional_role_codes"].values() for c in codes}
        self.assertEqual({r["code"] for r in matrix}, expected_codes)
        self.assertEqual(len(matrix), len(expected_codes))
        for row in matrix:
            self.assertEqual(set(row["roles"]), set(self.data["role_definitions"]))
            for role, value in row["roles"].items():
                relevant = [b for b in branches if row["code"] in b["conditional_role_codes"][role]]
                by_function, by_family = {}, {}
                for b in relevant:
                    by_family.setdefault(b["technology_family"], []).append(b["branch_id"])
                    for f in b["functions"]:
                        by_function.setdefault(f, []).append(b["branch_id"])
                flags = {f: int(f in by_function) if relevant else None for f in self.data["function_definitions"]}
                status = "conditional_documented_domain" if relevant else "unknown_or_unlinked_role"
                self.assertEqual(value["status"], status)
                self.assertEqual(value["function_flags"], flags)
                self.assertEqual(value["technology_families"], sorted(by_family))
                self.assertEqual(value["source_branch_ids_by_function"], {k: sorted(v) for k, v in by_function.items()})
                self.assertEqual(value["source_branch_ids_by_family"], {k: sorted(v) for k, v in by_family.items()})

    def test_csv_projections_and_missing_roles(self):
        with (ROOT / "data/public/guide_branches.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(self.data["revised_branch_ledger"]))
        for actual, expected in zip(rows, self.data["revised_branch_ledger"]):
            self.assertEqual(set(actual), set(expected))
            for key, value in expected.items():
                parsed = json.loads(actual[key]) if isinstance(value, (dict, list, int)) else actual[key]
                self.assertEqual(parsed, value)
        with (ROOT / "data/public/guide_industry_roles.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        matrix = {(r["code"], role): (r["name"], value)
                  for r in self.data["industry_role_technology_matrix"] for role, value in r["roles"].items()}
        self.assertEqual(len(rows), len(matrix))
        self.assertEqual({(r["industry_code"], r["role"]) for r in rows}, set(matrix))
        for row in rows:
            name, value = matrix[row["industry_code"], row["role"]]
            self.assertEqual(row["industry_name"], name)
            self.assertEqual(row["status"], value["status"])
            self.assertEqual(json.loads(row["technology_families"]), value["technology_families"])
            for f, flag in value["function_flags"].items():
                self.assertEqual(row[f], "" if flag is None else str(flag))

    def test_profiles_use_pair_sets_not_record_counts(self):
        from research_release import guide_design
        changed = copy.deepcopy(self.data)
        changed["revised_branch_ledger"].reverse()
        changed["revised_branch_ledger"].append(copy.deepcopy(changed["revised_branch_ledger"][0]))
        for role in guide_design.ROLES:
            original_table, original_records = guide_design.descriptors(self.data, role)
            changed_table, changed_records = guide_design.descriptors(changed, role)
            pd.testing.assert_frame_equal(original_table, changed_table)
            self.assertEqual(original_records, changed_records)
            for row in original_records:
                linked = [b for b in self.data["revised_branch_ledger"]
                          if row[guide_design.INDUSTRY] in b["conditional_role_codes"][role]]
                pairs = sorted({(b["technology_family"], f) for b in linked for f in b["functions"]})
                signature = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))
                self.assertEqual(row["profile"], "RP_" + hashlib.sha256(signature.encode()).hexdigest()[:16])


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_no_unresolved_global_dependencies(self):
        for path in (ROOT / "code/research_release").glob("*.py"):
            module = importlib.import_module("research_release." + path.stem)
            def walk(code):
                for instruction in dis.get_instructions(code):
                    if instruction.opname == "LOAD_GLOBAL":
                        self.assertTrue(instruction.argval in vars(module) or instruction.argval in vars(builtins),
                                        f"{path.stem}: {instruction.argval}")
                for constant in code.co_consts:
                    if inspect.iscode(constant):
                        walk(constant)
            for item in vars(module).values():
                if inspect.isfunction(item) and item.__module__ == module.__name__:
                    walk(item.__code__)
                elif inspect.isclass(item) and item.__module__ == module.__name__:
                    for member in vars(item).values():
                        if inspect.isfunction(member):
                            walk(member.__code__)

    def test_retained_function_logic(self):
        provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
        total = 0
        for row in provenance["modules"]:
            path = ROOT / "code/research_release" / (row["module"] + ".py")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            changes = row.get("readability_changes", {})
            aliases = {new: old for old, new in changes.get("module_aliases", {}).items()}
            messages = {new: old for old, new in changes.get("console_messages", {}).items()}
            # Retain the source hashes; reverse only the declared presentation edits.
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    node.id = aliases.get(node.id, node.id)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    node.value = messages.get(node.value, node.value)
            nodes = {n.name: n for n in tree.body
                     if isinstance(n, ast.FunctionDef)}
            for function in row["functions"]:
                checksum = hashlib.sha256(ast.dump(nodes[function["function"]], include_attributes=False).encode()).hexdigest()
                self.assertEqual(checksum, function["ast_sha256"])
                total += 1
            classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
            for item in row["class_definitions"]:
                checksum = hashlib.sha256(ast.dump(classes[item["class"]], include_attributes=False).encode()).hexdigest()
                self.assertEqual(checksum, item["ast_sha256"])
        self.assertGreater(total, 170)

    def test_no_local_paths_or_embedded_inputs(self):
        folders = [ROOT / folder for folder in ["code", "data/public", "results/reference"]]
        for path in self.runner.release_files(ROOT):
            if not any(path.is_relative_to(folder) for folder in folders):
                continue
            self.assertIn(path.suffix, [".py", ".json", ".csv"])
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]|C:\\\\Users|D:\\\\Desktop", text), path.name)
            self.assertNotIn("original_pdf_base64", text)
            self.assertNotIn("sample_keys_and_profiles", text)
        self.assertFalse((ROOT / "LICENSE").exists())

    def test_tables_are_aggregate_snapshots(self):
        tables = json.loads((ROOT / "results/reference/paper_tables.json").read_text(encoding="utf-8"))
        self.assertEqual(len(tables["manuscript"]["tables"]), 13)
        self.assertEqual(len(tables["supplement"]["tables"]), 66)
        for section in ["manuscript", "supplement"]:
            for table in tables[section]["tables"]:
                self.assertTrue(table["rows"])
                self.assertTrue(table["caption"], f"Missing caption on {section} object {table['table_object']}")

    def test_release_allowlist_and_hashes(self):
        manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
        actual = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.runner.release_files(ROOT)}
        expected = manifest["files_sha256"]
        self.assertEqual(set(actual), set(expected), "Missing or unreviewed release files.")
        for name, checksum in expected.items():
            self.assertEqual(actual[name], checksum, f"Reviewed file changed: {name}")


if __name__ == "__main__":
    unittest.main()
