"""Checks for the two final appendix diagnostics; synthetic inputs only."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_release import load_runner
from research_release import accounting, history, rd, sample, sample_bridge, wild_cluster


class RevisionDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def panel(self):
        rng = np.random.default_rng(391)
        n = 180
        frame = pd.DataFrame({sample.FIRM: [f"{i:06d}" for i in range(n)],
                              "year": np.tile([2008, 2009, 2010], n // 3),
                              sample.AGRI: rng.integers(0, 2, n),
                              sample.INDUSTRY: np.tile(["A01", "C13", "C26"], n // 3),
                              "province": rng.choice(["A", "B"], n),
                              sample.COUNT: rng.poisson(2, n)})
        for column in [*accounting.CONTROLS, *sample.LAGGED, sample.FI]:
            frame[column] = rng.normal(size=n)
        frame[sample.LOG_COUNT] = np.log1p(frame[sample.COUNT])
        frame[sample.ENTRY] = frame[sample.COUNT].gt(0).astype(int)
        return frame

    def test_sample_bridge_keeps_rows_and_separates_the_two_changes(self):
        frame = self.panel()
        frame.loc[[3, 5], sample.LAGGED[0]] = np.nan
        frame.loc[[6, 7], sample.FI] = np.nan
        frame.loc[8, [sample.LAGGED[0], sample.FI]] = np.nan
        masks = {"Static controls and fields complete": pd.Series(True, index=frame.index),
                 "Common estimation sample": frame[[*sample.LAGGED, sample.FI]].notna().all(axis=1)}
        result = self.runner.sample_bridge(frame, masks)
        self.assertEqual(result["samples"]["A_eligible"]["observations"], len(frame))
        self.assertEqual(result["samples"]["A_minus_common"]["observations"], 5)
        self.assertEqual(result["excluded_missingness"],
                         {"lagged_OI_OD_only": 2, "FI_only": 2, "both_lagged_OI_OD_and_FI": 1})
        for row in result["estimates"]:
            self.assertAlmostEqual(row["B_minus_A_sample_restriction"] + row["C_minus_B_add_gtos"],
                                   row["C_minus_A_total_change"], places=12)

    def test_sample_bridge_rejects_a_different_selection_rule(self):
        frame = self.panel()
        common = pd.Series(True, index=frame.index)
        common.iloc[0] = False
        with self.assertRaisesRegex(ValueError, "exactly GTOS availability"):
            self.runner.sample_bridge(frame, {"Static controls and fields complete": ~common | common,
                                              "Common estimation sample": common})

    def test_bridge_fit_matches_direct_ols_and_rejects_misalignment(self):
        frame = self.panel()
        design = sample_bridge.control_design(frame)
        y = frame[list(accounting.OUTCOMES.values())]
        values, _ = sample_bridge.fit(design, y)
        direct = np.linalg.lstsq(design.to_numpy(), y.to_numpy(), rcond=None)[0]
        np.testing.assert_allclose(values, direct[design.columns.get_loc("agri")], atol=1e-10)
        with self.assertRaisesRegex(ValueError, "do not align"):
            sample_bridge.fit(design, y.iloc[::-1])

    def test_wild_test_order_preserves_random_streams_and_positive_log(self):
        frame = self.panel()
        history = frame.iloc[10:].copy()
        guides = {"process_rd": frame.iloc[:80].assign(profile__one=1),
                  "supplier": frame.iloc[80:].assign(profile__one=1)}
        cases = self.runner.wild_cluster_cases(frame, history, guides)
        self.assertEqual(len(cases), 14)
        self.assertEqual([case[0] for case in cases],
                         ["Full accounting"] * 3 + ["Industry FE plus OI/OD"] * 3
                         + ["Full positive-count intensity", "Own history + patent GTOS"]
                         + ["Guide profiles: process_rd"] * 3 + ["Guide profiles: supplier"] * 3)
        positive = cases[6][2]
        self.assertTrue(positive[sample.COUNT].gt(0).all())
        np.testing.assert_array_equal(positive.log_positive_count, np.log(positive[sample.COUNT]))
        self.assertNotIn(sample.FI, cases[3][3])
        self.assertIn(sample.INDUSTRY, cases[3][3])
        self.assertIs(cases[7][2], history)
        self.assertIs(cases[8][2], guides["process_rd"])
        self.assertNotIn(sample.LAGGED[0], cases[8][3])

    def test_wild_algorithm_matches_refits_and_is_row_order_invariant(self):
        result = wild_cluster.synthetic_checks()
        self.assertEqual(result["direct_refits"], 18)
        self.assertLess(result["max_t_error"], 2e-8)

    def test_wild_weights_are_exhaustive_or_seeded_as_declared(self):
        signs, exact = wild_cluster.draw_weights(13, "Rademacher", 0)
        self.assertTrue(exact)
        self.assertEqual(signs.shape, (13, 8192))
        self.assertEqual(len(np.unique(signs.T, axis=0)), 8192)
        webb, exact = wild_cluster.draw_weights(13, "Webb", 1)
        self.assertFalse(exact)
        self.assertEqual(webb.shape, (13, 9999))
        np.testing.assert_array_equal(webb, wild_cluster.draw_weights(13, "Webb", 1)[0])

    def test_new_cli_routes_do_not_fall_through_to_core_models(self):
        from argparse import Namespace
        frame = self.panel()
        masks = {"Common estimation sample": pd.Series(True, index=frame.index)}
        with patch.object(self.runner, "canonical_sample", return_value=(frame, frame, masks, None)), \
             patch.object(self.runner, "sample_bridge", return_value={"synthetic": True}) as bridge:
            self.assertEqual(self.runner.analysis(Namespace(command="sample-bridge", panel="unused")),
                             {"synthetic": True})
            bridge.assert_called_once_with(frame, masks)

    def test_wild_cli_retains_the_rd_and_calendar_history_intersection(self):
        from argparse import Namespace
        frame = self.panel()
        common = pd.Series(True, index=frame.index)
        common.iloc[0] = False
        enriched = frame.assign(**{rd.RD_MEAN: 1., history.HISTORY: 1.})
        enriched.loc[1, rd.RD_MEAN] = np.nan
        enriched.loc[2, history.HISTORY] = np.nan
        guides = {"synthetic": frame}
        with patch.object(self.runner, "canonical_sample", return_value=(frame, frame, {"Common estimation sample": common}, None)), \
             patch.object(self.runner, "rd_input", return_value=(frame, {}, {}, {})), \
             patch.object(history, "add_calendar_history", return_value=enriched), \
             patch.object(self.runner, "guide_frames", return_value=(guides, {})), \
             patch.object(self.runner, "industry_wild_cluster", return_value={"synthetic": True}) as estimate:
            result = self.runner.analysis(Namespace(command="wild-cluster", panel="unused", rd="unused"))
            self.assertEqual(result, {"synthetic": True})
            pd.testing.assert_frame_equal(estimate.call_args.args[1], enriched.iloc[3:])
            self.assertIs(estimate.call_args.args[2], guides)


if __name__ == "__main__":
    unittest.main()
