"""
tests/test_evaluation_metrics.py

Unit tests for clinical consensus metrics (TIR, TAR, TBR, LBGI, HBGI, MG)
and Paired Two One-Sided Tests (TOST) statistical equivalence testing.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_population_model import window_outcomes_batch, paired_tost


class TestEvaluationMetrics(unittest.TestCase):
    def test_clinical_metrics_boundary_conditions(self):
        # Case 1: All values in range [70, 180] -> TIR = 100%, TAR = 0%, TBR = 0%
        bg_in_range = np.full((1, 61), 120.0)
        out1 = window_outcomes_batch(bg_in_range)
        self.assertAlmostEqual(out1["TIR"][0], 100.0)
        self.assertAlmostEqual(out1["TAR"][0], 0.0)
        self.assertAlmostEqual(out1["TBR"][0], 0.0)
        self.assertAlmostEqual(out1["MG"][0], 120.0)

        # Case 2: All values above range (> 180) -> TAR = 100%
        bg_above = np.full((1, 61), 250.0)
        out2 = window_outcomes_batch(bg_above)
        self.assertAlmostEqual(out2["TAR"][0], 100.0)
        self.assertAlmostEqual(out2["TIR"][0], 0.0)
        self.assertGreater(out2["HBGI"][0], 0.0)
        self.assertAlmostEqual(out2["LBGI"][0], 0.0)

        # Case 3: All values below range (< 70) -> TBR = 100%
        bg_below = np.full((1, 61), 50.0)
        out3 = window_outcomes_batch(bg_below)
        self.assertAlmostEqual(out3["TBR"][0], 100.0)
        self.assertAlmostEqual(out3["TIR"][0], 0.0)
        self.assertGreater(out3["LBGI"][0], 0.0)
        self.assertAlmostEqual(out3["HBGI"][0], 0.0)

    def test_paired_tost_equivalent(self):
        # Create identical simulated and actual series -> diff is 0, p-value must be < 0.05
        actual = np.random.uniform(70.0, 180.0, size=100)
        sim = actual + np.random.normal(0.0, 0.1, size=100)  # tiny noise within margin 5.0

        sim_mean, sim_std, act_mean, act_std, p_val, is_equiv = paired_tost(sim, actual, margin=5.0)
        self.assertTrue(is_equiv)
        self.assertLess(p_val, 0.05)

    def test_paired_tost_non_equivalent(self):
        # Create series with large systematic shift exceeding margin
        actual = np.random.uniform(70.0, 180.0, size=100)
        sim = actual + 15.0  # shift exceeds margin 5.0

        sim_mean, sim_std, act_mean, act_std, p_val, is_equiv = paired_tost(sim, actual, margin=5.0)
        self.assertFalse(is_equiv)
        self.assertGreater(p_val, 0.05)


if __name__ == "__main__":
    unittest.main()
