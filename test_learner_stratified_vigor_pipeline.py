import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from learner_stratified_vigor_pipeline import (
    BLOCK_COLUMN,
    CATCH_TRIALS,
    PipelineConfig,
    TIME_BIN_COLUMN,
    add_block_names,
    aggregate_temporal_bins,
    aggregate_trial_summary,
    checkpoint_is_current,
    checkpoint_signature,
    calculate_catch_timing_metrics,
    create_manifest,
    legacy_pandas_array_pickle_compat,
    prepare_analysis6_input,
    read_pickle_compat,
    restore_source_fish_ids,
    robust_profile_ylim,
    run_analysis6_classifier,
    summarize_catch_profiles,
    summarize_profiles,
)


class LegacyStringArrayProxy:
    """Serialize a StringArray with the two-item state found in the source files."""

    def __reduce__(self):
        array = pd.array(["legacy-a", "legacy-b"], dtype="string[python]")
        constructor, constructor_args, state = array.__reduce__()
        legacy_state = state[:2]
        return constructor, constructor_args, legacy_state


class LegacyCategoricalProxy:
    """Serialize a Categorical with the two-item state found in the source files."""

    def __reduce__(self):
        array = pd.Categorical(
            ["Pre-train", "Train 1"],
            categories=["Pre-train", "Train 1"],
            ordered=True,
        )
        constructor, constructor_args, state = array.__reduce__()
        legacy_state = state[:2]
        return constructor, constructor_args, legacy_state


class LearnerStratifiedVigorPipelineTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for fish in ("20260101_01", "20260101_02"):
            for trial in range(5, 25):
                for time_value in (-10.0, -1.0, 1.0, 5.0, 12.0, 15.0):
                    rows.append(
                        {
                            "Fish": fish,
                            "Trial number": trial,
                            "Trial time (s)": time_value,
                            "Vigor (deg/ms)": 2.0 if time_value <= 0 else 1.0,
                            "Scaled vigor (AU)": 0.75,
                            "Bout": time_value != 15.0,
                        }
                    )
        self.data = pd.DataFrame(rows)
        self.config = PipelineConfig(
            trace_path="trace.pkl",
            control_path="control.pkl",
            output_dir="out",
            bootstrap_iterations=20,
        )

    def test_trial_summary_uses_configured_windows_and_blocks(self):
        summary = aggregate_trial_summary(
            self.data,
            "Fish",
            self.data["Trial time (s)"],
            "3sTrace",
        )
        self.assertEqual(summary["Fish"].nunique(), 2)
        self.assertEqual(set(summary[BLOCK_COLUMN].astype(str)), {"Pre-train", "Train 1"})
        self.assertIn("Median 15 s before", summary.columns)
        self.assertIn("Median CR", summary.columns)
        np.testing.assert_allclose(summary["Normalized vigor"], -1.0)

    def test_temporal_bins_preserve_fish_and_trial_units(self):
        temporal = aggregate_temporal_bins(
            self.data,
            "Fish",
            self.data["Trial time (s)"],
            "3sTrace",
            self.config,
        )
        self.assertEqual(temporal["Fish_ID"].nunique(), 2)
        self.assertEqual(temporal["Trial number"].nunique(), 20)
        self.assertTrue(temporal[TIME_BIN_COLUMN].between(-15, 21).all())
        self.assertTrue(temporal["Movement probability"].between(0, 1).all())

    def test_manifest_keeps_controls_as_reference(self):
        all_fish = pd.DataFrame(
            {
                "Exp.": ["control", "3sTrace", "3sTrace"],
                "Fish": ["c1", "t1", "t2"],
            }
        )
        classification = pd.DataFrame(
            {
                "Condition": ["control", "3sTrace"],
                "Fish_ID": ["c1", "t1"],
                "Is_Learner": [True, True],
            }
        )
        manifest = create_manifest(classification, all_fish, "run-1", 0.05)
        strata = manifest.set_index("Fish_ID")["Analysis_Stratum"].to_dict()
        self.assertEqual(strata["c1"], "Reference")
        self.assertEqual(strata["t1"], "Conditioned learner")
        self.assertEqual(strata["t2"], "Conditioned unclassified")

    def test_analysis6_keys_keep_same_fish_id_separate_across_conditions(self):
        trial_summary = pd.DataFrame(
            {
                "Exp.": ["control", "3sTrace"],
                "Fish": ["20260101_01", "20260101_01"],
                "Trial number": [5, 5],
            }
        )
        prepared, key_map = prepare_analysis6_input(trial_summary)
        self.assertEqual(prepared["Fish"].nunique(), 2)
        self.assertEqual(
            set(prepared["Fish"]),
            {"control::20260101_01", "3sTrace::20260101_01"},
        )

        classification = pd.DataFrame(
            {
                "Condition": ["control", "3sTrace"],
                "Fish_ID": ["control::20260101_01", "3sTrace::20260101_01"],
                "Is_Learner": [False, True],
            }
        )
        restored = restore_source_fish_ids(classification, key_map)
        self.assertEqual(restored["Fish_ID"].tolist(), ["20260101_01", "20260101_01"])
        self.assertEqual(restored["Classifier_Fish_ID"].nunique(), 2)

    def test_profile_summary_aggregates_trials_within_fish_first(self):
        temporal = pd.DataFrame(
            {
                "Analysis_Stratum": ["Reference"] * 4,
                "Fish_ID": ["a", "a", "b", "b"],
                BLOCK_COLUMN: ["Pre-train"] * 4,
                TIME_BIN_COLUMN: [0.25] * 4,
                "Trial number": [5, 6, 5, 6],
                "Conditional vigor": [1.0, 3.0, 5.0, 7.0],
            }
        )
        profiles, coverage = summarize_profiles(
            temporal, "Conditional vigor", self.config
        )
        self.assertEqual(profiles.loc[0, "Median"], 4.0)
        self.assertEqual(coverage.loc[0, "Contributing_Fish"], 2)

    def test_catch_profiles_include_only_configured_catch_trials(self):
        rows = []
        for fish in ("a", "b"):
            for trial in (*CATCH_TRIALS, 30):
                rows.append(
                    {
                        "Analysis_Stratum": "Conditioned learner",
                        "Fish_ID": fish,
                        "Trial number": trial,
                        TIME_BIN_COLUMN: 1.0,
                        "Conditional vigor": float(trial),
                    }
                )
        profiles, _ = summarize_catch_profiles(
            pd.DataFrame(rows),
            "Conditional vigor",
            self.config,
        )
        self.assertEqual(set(profiles["Trial number"]), set(CATCH_TRIALS))
        self.assertNotIn(30, profiles["Trial number"].tolist())

    def test_catch_timing_metrics_detect_strengthening_near_expected_us(self):
        rows = []
        for fish in ("learner-a", "learner-b"):
            for trial in CATCH_TRIALS:
                for time_value, outcome_value in (
                    (-4.0, 1.0),
                    (-1.0, 1.0),
                    (1.0, 0.9),
                    (3.0, 0.8),
                    (7.0, 0.6),
                    (9.0, 0.5),
                    (10.5, 0.4),
                    (12.5, 0.2),
                ):
                    rows.append(
                        {
                            "Condition": "3sTrace",
                            "Analysis_Stratum": "Conditioned learner",
                            "Fish_ID": fish,
                            "Trial number": trial,
                            TIME_BIN_COLUMN: time_value,
                            "Conditional vigor": outcome_value,
                            "Movement probability": outcome_value,
                        }
                    )
        per_fish, summary = calculate_catch_timing_metrics(
            pd.DataFrame(rows),
            self.config,
        )
        self.assertTrue(per_fish["Near_US_Strengthening"].gt(0).all())
        self.assertTrue(per_fish["Anticipatory_Suppression_Slope"].gt(0).all())
        strengthening = summary[summary["Metric"].eq("Near_US_Strengthening")]
        self.assertTrue(strengthening["CI_Low"].gt(0).all())

    def test_robust_y_limits_ignore_extreme_confidence_band(self):
        profiles = pd.DataFrame(
            {
                "Median": [-0.1, 0.0, 0.2],
                "CI_Low": [-100.0, -1.0, -1.0],
                "CI_High": [100.0, 1.0, 1.0],
            }
        )
        low, high = robust_profile_ylim(profiles, "Conditional vigor")
        self.assertGreater(low, -1.0)
        self.assertLess(high, 1.0)

    def test_block_mapping_excludes_trials_outside_analysis_blocks(self):
        data = pd.DataFrame({"Trial number": [1, 5, 94, 95]})
        mapped = add_block_names(data)
        self.assertEqual(mapped["Trial number"].tolist(), [5, 94])

    def test_checkpoint_signature_rejects_changed_binning(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.pkl"
            source.write_bytes(b"test")
            metadata = Path(temporary) / "checkpoint.json"
            signature = checkpoint_signature(source, "3sTrace", self.config)
            metadata.write_text(
                __import__("json").dumps(signature),
                encoding="utf-8",
            )
            self.assertTrue(checkpoint_is_current(metadata, signature))
            changed = dict(signature, time_bin_s=1.0)
            self.assertFalse(checkpoint_is_current(metadata, changed))

    def test_legacy_string_array_pickle_state_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.pkl"
            with path.open("wb") as handle:
                pickle.dump(pd.DataFrame({"value": [LegacyStringArrayProxy()]}), handle)
            with self.assertRaises(NotImplementedError):
                pd.read_pickle(path)
            loaded = read_pickle_compat(path)
        self.assertIsInstance(loaded, pd.DataFrame)
        self.assertEqual(loaded.shape, (1, 1))

    def test_legacy_string_array_patch_is_restored(self):
        from pandas.core.arrays import Categorical
        from pandas.core.arrays.string_ import StringArray

        originals = {
            StringArray: StringArray.__setstate__,
            Categorical: Categorical.__setstate__,
        }
        with legacy_pandas_array_pickle_compat():
            for array_class, original in originals.items():
                self.assertIsNot(array_class.__setstate__, original)
        for array_class, original in originals.items():
            self.assertIs(array_class.__setstate__, original)

    def test_legacy_categorical_pickle_state_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-categorical.pkl"
            with path.open("wb") as handle:
                pickle.dump(pd.DataFrame({"value": [LegacyCategoricalProxy()]}), handle)
            with self.assertRaises(NotImplementedError):
                pd.read_pickle(path)
            loaded = read_pickle_compat(path)
        self.assertIsInstance(loaded.loc[0, "value"], pd.Categorical)
        self.assertEqual(
            loaded.loc[0, "value"].tolist(),
            ["Pre-train", "Train 1"],
        )

    def test_temporal_binning_handles_duplicate_source_indexes(self):
        duplicated = self.data.copy()
        duplicated.index = np.zeros(len(duplicated), dtype=int)
        temporal = aggregate_temporal_bins(
            duplicated,
            "Fish",
            duplicated["Trial time (s)"],
            "3sTrace",
            self.config,
        )
        self.assertEqual(temporal["Fish_ID"].nunique(), 2)
        self.assertEqual(temporal["Trial number"].nunique(), 20)

    @unittest.skipUnless(
        importlib.util.find_spec("statsmodels"),
        "Analysis 6 integration requires the pinned runtime environment.",
    )
    def test_analysis6_integration_returns_classification_rows(self):
        rng = np.random.default_rng(7)
        rows = []
        for condition in ("control", "3sTrace"):
            for fish_number in range(6):
                fish = f"{condition}_{fish_number:02d}"
                for trial in range(5, 95):
                    if condition == "control":
                        normalized = rng.normal(0, 0.05)
                    elif fish_number < 3 and trial < 15:
                        normalized = 0.2 + rng.normal(0, 0.05)
                    elif fish_number < 3 and trial < 75:
                        normalized = -0.45 + rng.normal(0, 0.05)
                    elif fish_number < 3:
                        normalized = 0.1 + rng.normal(0, 0.05)
                    else:
                        normalized = rng.normal(0, 0.05)
                    rows.append(
                        {
                            "Exp.": condition,
                            "Fish": fish,
                            "Trial number": trial,
                            "Median 15 s before": 0.0,
                            "Median CR": normalized,
                            "Normalized vigor": normalized,
                        }
                    )
        summary = add_block_names(pd.DataFrame(rows))
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            input_path = temporary_path / "trial_summary.csv"
            summary.to_csv(input_path, index=False)
            classification = run_analysis6_classifier(
                input_path,
                temporary_path / "output",
                Path(__file__).resolve().parent,
                0.05,
            )
        self.assertEqual(
            {
                "Fish_ID",
                "Condition",
                "Is_Learner",
                "T_joint",
                "p_empirical",
            }.difference(classification.columns),
            set(),
        )
        self.assertGreaterEqual(classification["Fish_ID"].nunique(), 10)


if __name__ == "__main__":
    unittest.main()
