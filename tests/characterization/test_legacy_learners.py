from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_script(filename: str, module_name: str) -> ModuleType:
    path = REPOSITORY_ROOT / filename
    repository_path = str(REPOSITORY_ROOT)
    if repository_path not in sys.path:
        sys.path.insert(0, repository_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class LegacyLearnerCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original = load_script(
            "6_LearnersQuantification.py",
            "legacy_learners_original",
        )
        cls.new = load_script(
            "6_LearnersQuantification_new.py",
            "legacy_learners_new",
        )
        cls.improved = load_script(
            "6_LearnersQuantification_improved.py",
            "legacy_learners_improved",
        )
        cls.wip = load_script(
            "6_LearnersQuantification_WIP.py",
            "legacy_learners_wip",
        )

    def test_active_variants_have_incompatible_decision_defaults(self) -> None:
        self.assertFalse(self.original.RUN_EXPORT_RESULTS)
        self.assertTrue(self.new.RUN_EXPORT_RESULTS)
        self.assertEqual(self.new.MIN_FISH_WITH_ALL_FEATURES, 10)
        self.assertEqual(self.new.N_BOOTSTRAP, 50)
        self.assertEqual(self.improved.ALPHA_TARGET, 0.10)
        self.assertEqual(self.wip.ALPHA_TARGET, 0.05)
        self.assertTrue(self.wip.USE_LOG_TRANSFORM)
        self.assertEqual(self.wip.POOLED_DATA_NAN_FILTER, "no_nanFracFilt")
        self.assertFalse(self.wip.APPLY_FISH_DISCARD)

    def test_original_and_new_use_different_directional_votes(self) -> None:
        original_source = inspect.getsource(self.original.classify_learners)
        new_source = inspect.getsource(self.new.classify_learners)
        self.assertIn("se_multiplier_for_voting", original_source)
        self.assertIn("votes_conservative", original_source)
        self.assertNotIn("p_learning", original_source)
        self.assertIn("probability_threshold", new_source)
        self.assertIn("votes_probabilistic", new_source)
        self.assertIn("p_learning", new_source)

    def test_improved_variants_use_leave_one_out_control_references(self) -> None:
        values = np.array([[1.0, 10.0], [3.0, 30.0], [100.0, 1_000.0]])
        is_control = np.array([True, True, False])
        expected_full = np.array([2.0, 20.0])
        expected_reference = np.array(
            [[3.0, 30.0], [1.0, 10.0], [2.0, 20.0]]
        )

        for module in (self.improved, self.wip):
            full, reference = module._build_reference_means_loocv(
                values,
                is_control,
            )
            np.testing.assert_allclose(full, expected_full)
            np.testing.assert_allclose(reference, expected_reference)

    def test_improved_variants_use_finite_sample_empirical_p_values(self) -> None:
        scores = np.array([0.5, 2.0, 4.0])
        control_scores = np.array([1.0, 2.0, 3.0])
        expected = np.array([1.0, 0.75, 0.25])
        for module in (self.improved, self.wip):
            np.testing.assert_allclose(
                module._empirical_p_values(scores, control_scores),
                expected,
            )

    def test_improved_and_wip_model_different_response_quantities(self) -> None:
        improved_source = inspect.getsource(self.improved.extract_change_feature)
        wip_source = inspect.getsource(self.wip.extract_change_feature)
        self.assertIn(
            "Log_Response ~ Log_Baseline + Epoch",
            improved_source,
        )
        self.assertIn(
            "Log_Response_Adj ~ Log_Baseline",
            improved_source,
        )
        self.assertIn("Log_Normalized_Vigor ~ Epoch", wip_source)
        self.assertIn("Q('Vigor_Adj') ~ 1", wip_source)

    def test_all_variants_require_complete_multifeature_cohorts(self) -> None:
        original_source = inspect.getsource(
            self.original.run_multivariate_lme_pipeline
        )
        new_source = inspect.getsource(self.new.run_multivariate_lme_pipeline)
        improved_source = inspect.getsource(
            self.improved.run_multivariate_lme_pipeline
        )
        wip_source = inspect.getsource(self.wip.run_multivariate_lme_pipeline)
        self.assertIn("len(common_fish) < 10", original_source)
        for source in (new_source, improved_source, wip_source):
            self.assertIn("MIN_FISH_WITH_ALL_FEATURES", source)


if __name__ == "__main__":
    unittest.main()
