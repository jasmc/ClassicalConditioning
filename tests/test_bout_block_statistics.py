import unittest
import numpy as np
import pandas as pd
from scipy.stats import binomtest, brunnermunzel
from classical_conditioning.analysis.bout_block_statistics import exact_sign, independent_test, corrected_b_statistics, VALUE


class BoutBlockStatisticsTests(unittest.TestCase):
    def test_sign_test_ignores_magnitude_and_handles_zero_pairs(self):
        result = exact_sign([-1000., -.01, -.1, -.2, 0., 1.])
        self.assertAlmostEqual(result["p_raw"], binomtest(1,5,.5).pvalue)
        self.assertEqual(result["zero_differences"], 1)
        self.assertEqual(exact_sign([0.,0.])["p_raw"], 1.)

    def test_independent_test_has_heterogeneous_shapes_and_reverses_effect(self):
        t = np.linspace(-1,1,20)**3 - .4
        c = np.linspace(-.2,.3,18)
        result = independent_test(t,c)
        self.assertAlmostEqual(result["p_raw"], brunnermunzel(t,c,distribution="t").pvalue)
        self.assertAlmostEqual(result["rank_biserial"], -independent_test(c,t)["rank_biserial"])

    def test_family_has_36_tests_and_rejects_duplicate_fish(self):
        rng = np.random.default_rng(17)
        panels = {}
        for panel, condition in [("D","delay"),("E","trace")]:
            rows = []
            for c in (condition,"control"):
                for fish in range(16):
                    base = rng.normal(0,.1)
                    for block in range(3):
                        rows.append(dict(condition_id=c,fish_id=f"{c}{fish}", **{"Selected block order":block,VALUE:base+rng.normal(0,.05)-(0.1 if block==1 and c==condition else 0),"Eligible":True}))
            panels[panel] = pd.DataFrame(rows)
        tests, changes = corrected_b_statistics(panels)
        self.assertEqual(len(tests),36)
        self.assertTrue(tests.p_holm36.ge(tests.p_raw-1e-12).all())
        self.assertEqual(len(changes),192)
        panels["D"] = pd.concat([panels["D"],panels["D"].iloc[[0]]])
        with self.assertRaisesRegex(ValueError,"pseudoreplicate"):
            corrected_b_statistics(panels)
