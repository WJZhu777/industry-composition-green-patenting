"""Coverage and inference checks for the contextual penalty comparison."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_release import penalties


def fixture():
    rng = np.random.default_rng(407)
    frame = pd.MultiIndex.from_product(
        [[f"{i:06d}" for i in range(40)], range(2007, 2025)],
        names=[penalties.FIRM, "year"],
    ).to_frame(index=False)
    firm = frame[penalties.FIRM].astype(int)
    frame[penalties.AGRI] = (firm % 3 == 0).astype(int)
    frame[penalties.INDUSTRY] = (firm % 5).map(lambda x: f"C{x:02d}")
    frame["province"] = (firm % 4).astype(str)
    for variable in penalties.CONTROLS + penalties.GTOS:
        frame[variable] = rng.normal(size=len(frame))
    frame[penalties.OUTCOME] = rng.binomial(1, 0.2, size=len(frame))
    return frame


class PenaltyTests(unittest.TestCase):
    def test_coverage_excludes_unobserved_and_partial_years_without_mutation(self):
        frame = fixture()
        original = frame.copy(deep=True)
        selected = penalties.covered_sample(frame)
        self.assertEqual(set(selected.year), set(range(2009, 2024)))
        pd.testing.assert_frame_equal(frame, original)
        self.assertTrue(selected[penalties.OUTCOME].eq(0).any())

    def test_all_table_samples_use_covered_years(self):
        frame = fixture()
        common = frame.loc[frame.year.ge(2008)].copy()
        original = frame.copy(deep=True)
        seen = []

        def capture(selected, rhs, terms):
            seen.append(selected.copy())
            return {"n": len(selected)}

        with patch.object(penalties, "fit", side_effect=capture):
            result = penalties.estimate(frame, common)
        self.assertEqual(len(seen), 12)
        self.assertTrue(all(s.year.between(2009, 2023).all() for s in seen))
        self.assertEqual(len(seen[6]), 40 * 6)
        self.assertEqual(len(seen[8]), 40 * 9)
        self.assertEqual(result["sample"]["original_panel_n"], len(frame))
        self.assertEqual(seen[6].index.tolist(), seen[7].index.tolist())
        self.assertEqual(seen[8].index.tolist(), seen[9].index.tolist())
        self.assertEqual(seen[10].index.tolist(), seen[11].index.tolist())
        pd.testing.assert_frame_equal(frame, original)

    def test_clustered_fit_matches_direct_matrix_calculation(self):
        frame = penalties.covered_sample(fixture())
        rhs = penalties.AGRI + " + ln_assets"
        result = penalties.fit(frame, rhs, [penalties.AGRI])
        x = np.column_stack([np.ones(len(frame)), frame[penalties.AGRI], frame.ln_assets])
        y = frame[penalties.OUTCOME].to_numpy()
        inverse = np.linalg.inv(x.T @ x)
        beta = inverse @ x.T @ y
        scores = x * (y - x @ beta)[:, None]
        grouped = pd.DataFrame(scores).groupby(frame[penalties.FIRM].to_numpy()).sum().to_numpy()
        n, k = x.shape
        g = frame[penalties.FIRM].nunique()
        covariance = inverse @ grouped.T @ grouped @ inverse * (g / (g - 1)) * ((n - 1) / (n - k))
        term = result["terms"][penalties.AGRI]
        self.assertAlmostEqual(term["coefficient"], beta[1], places=11)
        self.assertAlmostEqual(term["se"], np.sqrt(covariance[1, 1]), places=11)

    def test_winsor_cutoffs_do_not_change_with_penalty_period(self):
        frame = fixture()
        prepared, cutoffs = penalties.prepare_baseline(frame)
        expected = frame.ln_assets.quantile([0.01, 0.99]).tolist()
        self.assertEqual(cutoffs["ln_assets"], expected)
        selected = penalties.covered_sample(prepared)
        np.testing.assert_allclose(selected.ln_assets_w, frame.loc[selected.index, "ln_assets"].clip(*expected))

    def test_legacy_baseline_penalty_api_applies_same_coverage(self):
        from research_release import industry_baseline_models as legacy
        frame, _ = penalties.prepare_baseline(fixture())
        seen = []
        original_fit = legacy.fit_clustered_ols

        def checked_fit(selected, formula):
            self.assertTrue(selected.year.between(2009, 2023).all())
            seen.append(len(selected))
            return original_fit(selected, formula)

        with patch.object(legacy, "OUTCOME_SPECS", [(penalties.OUTCOME, "penalized_dummy", "LPM")]), \
             patch.object(legacy, "fit_clustered_ols", side_effect=checked_fit):
            result, _ = legacy.run_new_models(frame)
        self.assertEqual(len(result), 4)
        self.assertEqual(seen, [600] * 4)


if __name__ == "__main__":
    unittest.main()
