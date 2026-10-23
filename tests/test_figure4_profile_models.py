"""Known-answer checks for the exploratory Figure 4 profile adapters."""
import unittest
import numpy as np
import pandas as pd
from scipy.linalg import block_diag

from classical_conditioning.analysis.review.figure4_profiles import (
    GROUPS, FORMULA, model_data, fit, prediction_design, differences, peaks,
    stratified_weights, individual_predictions,
)
from classical_conditioning.analysis.inference.panel_g_fast_lmm import prepare_fast


class ProfileModelTests(unittest.TestCase):
    def test_contrast_sign_and_peak_grid(self):
        curves = np.array([[0, 0, 0], [-1, -2, -1], [0, -.5, 0]])
        d = differences(curves)
        np.testing.assert_array_equal(d, [[1, 2, 1], [0, .5, 0], [1, 1.5, 1]])
        amplitude, latency = peaks(d, [.25, .75, 1.25])
        np.testing.assert_array_equal(amplitude, [2, .5, 1.5])
        np.testing.assert_array_equal(latency, [.75, .75, .75])

    def test_bootstrap_preserves_declared_strata_and_duplicate_clusters(self):
        keys = [f"f{i}" for i in range(12)]
        labels = {k: GROUPS[i // 4] for i, k in enumerate(keys)}
        weights = stratified_weights(keys, labels, np.random.default_rng(12))
        self.assertEqual(weights.sum(), 12)
        self.assertTrue(np.any(weights > 1))
        for i in range(3): self.assertEqual(weights[4 * i:4 * i + 4].sum(), 4)

    def test_missing_intensity_is_not_imputed_and_duplicate_bins_fail(self):
        fish = pd.DataFrame({"fish_id": ["a", "b"], "condition_id": ["control", "delay"],
            "display_group": [GROUPS[0], GROUPS[1]], "time_s": [1., 1.], "signed": [np.nan, 0.]})
        data = model_data(fish)
        self.assertEqual(data.fish_id.tolist(), ["b"])
        self.assertEqual(data.signed.tolist(), [0.])
        with self.assertRaises(ValueError): model_data(pd.concat([fish, fish]))

    def test_fixed_and_individual_predictions_match_dense_gaussian_equations(self):
        rng = np.random.default_rng(4)
        rows = []
        times = np.arange(-4.75, 12, .5)
        for g, name in enumerate(GROUPS):
            for i in range(8):
                offset = rng.normal(0, .25)
                for t in times:
                    if rng.random() < .05: continue
                    y = .04 * t + .12 * g * np.sin(t / 3) + offset + rng.normal(0, .08)
                    rows.append({"fish_id": f"{g}-{i}", "condition_id": "control" if g == 0 else "delay",
                                 "display_group": name, "time_s": t, "signed": y})
        data = model_data(pd.DataFrame(rows))
        result, attempts = fit(data, random="1")
        self.assertIsNotNone(result, attempts)
        diagnostics = next(a for a in reversed(attempts) if a["diagnostic_status"] == "ok")
        noise = diagnostics["residual_variance"]
        tau = float(result.cov_re[0, 0])
        prepared = prepare_fast(data, FORMULA, random="1")
        v = block_diag(*[noise * np.eye(len(i)) + tau for i in prepared.indices])
        # prepare_fast sorts by group membership; the adapter supplies contiguous fish rows.
        vx = np.linalg.solve(v, prepared.x)
        beta = np.linalg.solve(prepared.x.T @ vx, vx.T @ prepared.y)
        np.testing.assert_allclose(result.fe_params, beta, rtol=1e-6, atol=1e-8)
        grid, design = prediction_design(result, times)
        self.assertEqual(grid.display_group.iloc[0], GROUPS[0])
        conditional = individual_predictions(result, data, times, "1", noise)
        for fish, positions in data.groupby("fish_id", sort=False).indices.items():
            group = GROUPS.index(str(data.iloc[positions[0]].display_group))
            r = prepared.y[positions] - prepared.x[positions] @ beta
            gamma = tau * r.sum() / (noise + len(r) * tau)
            expected = design[group * len(times):(group + 1) * len(times)] @ beta + gamma
            np.testing.assert_allclose(conditional[fish], expected, rtol=1e-6, atol=1e-8)


if __name__ == "__main__": unittest.main()
