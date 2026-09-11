from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd

import file_utils

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_script(filename: str, module_name: str) -> ModuleType:
    path = REPOSITORY_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class LegacyDownstreamCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grouping = load_script("3_FishGrouping.py", "legacy_fish_grouping")
        cls.normalized = load_script(
            "5_NormalizedVigorPlotting.py",
            "legacy_normalized_vigor",
        )
        cls.normalized.config = object()
        cls.normalized.fish_ids_to_discard = []

        cls.scaled_temporary = tempfile.TemporaryDirectory()
        root = Path(cls.scaled_temporary.name)
        folder_tuple = tuple(root / f"folder-{index}" for index in range(18))
        for folder in folder_tuple:
            folder.mkdir()
        with patch.object(file_utils, "create_folders", return_value=folder_tuple):
            cls.scaled = load_script(
                "4_ScaledVigorPlotting.py",
                "legacy_scaled_vigor",
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.scaled_temporary.cleanup()

    def test_stage_three_uses_early_baseline_percentiles(self) -> None:
        module = self.grouping
        frame = pd.DataFrame(
            {
                module.FISH_COL: ["fish-1"] * 4,
                module.TRIAL_NUMBER_COL: [1] * 4,
                module.TIME_COL: [-30, -20, -10, 0],
                module.VIGOR_COL: [1.0, 3.0, 5.0, 7.0],
                module.SCALED_VIGOR_COL: [0.0] * 4,
                module.BOUT_COL: [True] * 4,
            }
        )
        result = module.process_data(
            frame,
            window_size=1,
            downsample_factor=1,
            grouping_cols=[module.FISH_COL, module.TRIAL_NUMBER_COL],
            time_col=module.TIME_COL,
            baseline_window_frames=15,
        )
        expected = (np.array([1.0, 3.0, 5.0, 7.0]) - 1.2) / (2.8 - 1.2)
        np.testing.assert_allclose(
            result[module.SCALED_VIGOR_COL].to_numpy(dtype=float),
            expected,
        )

    def test_stage_three_masks_out_of_bout_vigor_as_missing(self) -> None:
        module = self.grouping
        frame = pd.DataFrame(
            {
                module.FISH_COL: ["fish-1"] * 4,
                module.TRIAL_NUMBER_COL: [1] * 4,
                module.TIME_COL: [-30, -20, -10, 0],
                module.VIGOR_COL: [1.0, 3.0, 5.0, 7.0],
                module.SCALED_VIGOR_COL: [0.0] * 4,
                module.BOUT_COL: [True, True, False, False],
            }
        )
        result = module.process_data(
            frame,
            window_size=1,
            downsample_factor=1,
            grouping_cols=[module.FISH_COL, module.TRIAL_NUMBER_COL],
            time_col=module.TIME_COL,
            baseline_window_frames=15,
        )
        self.assertTrue(result.loc[result[module.TIME_COL] >= -10, module.VIGOR_COL].isna().all())

    def test_stage_four_time_bins_force_zero_edge_and_extend_range(self) -> None:
        bins = self.scaled.compute_time_bins((-20, 20), 0.5)
        self.assertIn(0.0, bins)
        self.assertLessEqual(min(bins), -21.0)
        self.assertGreaterEqual(max(bins), 22.0)
        np.testing.assert_allclose(np.diff(bins), 0.5)

    def test_stage_four_baseline_display_transform_includes_time_zero(self) -> None:
        module = self.scaled
        frame = pd.DataFrame(
            {
                "Trial time (s)": [-15.0, -1.0, 0.0, 1.0],
                "Scaled vigor (AU)": [1.0, 3.0, 100.0, 5.0],
            }
        )
        module.DO_BASELINE_SUBTRACT = True
        result = module.apply_baseline_and_clip(frame.copy(), [-100.0, 100.0])
        expected_baseline = np.median([1.0, 3.0, 100.0])
        np.testing.assert_allclose(
            result["Scaled vigor (AU)"].to_numpy(),
            np.clip(
                np.array([1.0, 3.0, 100.0, 5.0]) - expected_baseline,
                -100.0,
                100.0,
            ),
        )

    def test_stage_four_filters_configured_fish_ids(self) -> None:
        module = self.scaled
        frame = pd.DataFrame(
            {
                "Fish": ["keep", "discard", "keep"],
                "value": [1, 2, 3],
            }
        )
        with (
            patch.object(module, "APPLY_FISH_DISCARD", True),
            patch.object(module, "fish_ids_to_discard", ["discard"]),
        ):
            result = module.filter_discarded_fish_ids(frame, source="test")

        self.assertEqual(result["Fish"].tolist(), ["keep", "keep"])
        self.assertEqual(frame["Fish"].tolist(), ["keep", "discard", "keep"])

    def test_stage_four_build_uses_two_stage_aggregation_and_fish_count(self) -> None:
        module = self.scaled
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            all_fish = root / "all-fish"
            pooled = root / "pooled"
            figures = root / "figures"
            all_fish.mkdir()
            pooled.mkdir()
            figures.mkdir()

            times = [-3.75, -3.25, -0.75, -0.25, 0.25, 0.75]
            values = {
                "fish-1": [0.0, 1.0, 10.0, 20.0, 30.0, 40.0],
                "fish-2": [100.0, 2.0, 10.0, 20.0, 30.0, 40.0],
            }
            rows = []
            for fish, fish_values in values.items():
                for time, value in zip(times, fish_values):
                    rows.append(
                        {
                            "Exp.": "delay",
                            module.time_frame_col: time,
                            "CS beg": 0,
                            "CS end": 1,
                            "Trial number": 1,
                            "Block name": "Train",
                            "Scaled vigor (AU)": value,
                            "Vigor (deg/ms)": value,
                            "Fish": fish,
                            "Bout": True,
                        }
                    )
            input_path = all_fish / "pooled_delay_CS.pkl"
            pd.DataFrame(rows).to_pickle(input_path, compression="gzip")

            def use_supplied_seconds(frame: pd.DataFrame) -> pd.DataFrame:
                return frame.rename(
                    columns={module.time_frame_col: "Trial time (s)"}
                )

            with (
                patch.object(module, "path_all_fish", all_fish),
                patch.object(module, "path_pooled_data", pooled),
                patch.object(module, "path_scaled_vigor_fig", figures),
                patch.object(module, "APPLY_FISH_DISCARD", False),
                patch.object(module, "fish_ids_to_discard", []),
                patch.object(module, "binning_windows", [1.0]),
                patch.object(module, "x_lim", (-4.0, 1.0)),
                patch.object(module, "csus", "CS"),
                patch.object(module, "ensure_time_seconds", use_supplied_seconds),
            ):
                module.run_build_pooled_outputs()

            line_path = next(pooled.glob("SV lineplot 1.0s bins*.pkl"))
            count_path = next(pooled.glob("Count heatmap 1.0s bins*.pkl"))
            heatmap_path = next(pooled.glob("SV heatmap 1.0s bins*.pkl"))
            line = pd.read_pickle(line_path, compression="gzip")
            count = pd.read_pickle(count_path, compression="gzip")
            heatmap = pd.read_pickle(heatmap_path, compression="gzip")

            first_bin = line.loc[
                np.isclose(line["Trial time (s)"], -3.5),
                "Scaled vigor (AU)",
            ].iloc[0]
            self.assertAlmostEqual(first_bin, 25.75)
            self.assertTrue(np.allclose(line["Count"], 2.0))
            self.assertTrue(np.allclose(count.drop(columns="Exp."), 2.0))
            self.assertAlmostEqual(float(heatmap.loc[1, -3.5]), 1.0)
            self.assertAlmostEqual(float(heatmap.loc[1, -0.5]), 0.0)
            self.assertAlmostEqual(float(heatmap.loc[1, 0.5]), 1.0)

    def test_stage_four_artifact_and_catch_selection_rules(self) -> None:
        module = self.scaled
        self.assertTrue(module._stem_matches_csus("pooled_delay_CS", "CS"))
        self.assertTrue(
            module._stem_matches_csus("pooled_delay_CS_selectedFish", "CS")
        )
        self.assertTrue(module._stem_matches_csus("pooled_delay_CS_allFish", "CS"))
        self.assertFalse(module._stem_matches_csus("pooled_delay_US", "CS"))

        with patch.object(module, "EXPERIMENT", "allDelay"):
            trials, names = module.resolve_catch_trials()
        self.assertEqual(trials, [25, 39, 53, 59, 65])
        self.assertEqual(names, module.trial_names)

        with patch.object(module, "EXPERIMENT", "longDelay"):
            long_trials, long_names = module.resolve_catch_trials()
        self.assertEqual(long_trials[-2:], module.catch_trials_retraining)
        self.assertEqual(long_names[-2:], module.retraining_trial_names)

    def test_stage_five_keeps_fish_passing_only_one_block(self) -> None:
        module = self.normalized
        module.minimum_trials_per_fish_per_block = 6
        rows = []
        for block_name, count in (("Block A", 6), ("Block B", 1)):
            for trial in range(count):
                rows.append(
                    {
                        "Exp.": "delay",
                        "Fish": "fish-A",
                        "Block name": block_name,
                        "Trial number": trial + 1,
                        "Mean 15 s before": 2.0,
                        "Mean CR": 1.0,
                        "Normalized vigor": 0.5,
                    }
                )
        for block_name in ("Block A", "Block B"):
            for trial in range(5):
                rows.append(
                    {
                        "Exp.": "delay",
                        "Fish": "fish-B",
                        "Block name": block_name,
                        "Trial number": trial + 1,
                        "Mean 15 s before": 2.0,
                        "Mean CR": 1.0,
                        "Normalized vigor": 0.5,
                    }
                )

        result = module.prepare_main_df(
            pd.DataFrame(rows),
            apply_fish_discard=False,
        )
        self.assertEqual(set(result["Fish_ID"]), {"fish-A"})
        self.assertEqual(
            int((result["Block_name"] == "Block B").sum()),
            1,
        )

    def test_stage_five_cs_windows_include_zero_in_both_means(self) -> None:
        module = self.normalized
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            all_fish = root / "all-fish"
            pooled = root / "pooled"
            all_fish.mkdir()
            pooled.mkdir()
            time_column = module.gen_config.time_trial_frame_label
            frame = pd.DataFrame(
                {
                    "Strain": ["strain"] * 3,
                    "Age (dpf)": [7] * 3,
                    "Exp.": ["delay"] * 3,
                    "ProtocolRig": ["rig"] * 3,
                    "Day": ["day"] * 3,
                    "Fish no.": [1] * 3,
                    "Fish": ["fish-1"] * 3,
                    "Block name": ["Train"] * 3,
                    "Trial number": [1] * 3,
                    time_column: [-2.0, 0.0, 1.0],
                    "Scaled vigor (AU)": [0.0, 0.0, 0.0],
                    "Vigor (deg/ms)": [2.0, 10.0, 4.0],
                    "US beg": [0.0, 0.0, 0.0],
                }
            )
            frame.to_pickle(all_fish / "delay_CS_input.pkl", compression="gzip")

            def use_supplied_seconds(data: pd.DataFrame) -> pd.DataFrame:
                return data.rename(columns={time_column: "Trial time (s)"})

            with (
                patch.object(module, "path_all_fish", all_fish),
                patch.object(module, "path_pooled_data", pooled),
                patch.object(module, "cond_types", ["delay"]),
                patch.object(module, "cr_window", [0.0, 1.0]),
                patch.object(module, "blocks_dict", {}),
                patch.object(module, "csus", "CS"),
                patch.object(module, "RUN_PROCESS", True),
                patch.object(module, "APPLY_FISH_DISCARD", False),
                patch.object(module, "APPLY_MAX_NAN_FRAC_PER_WINDOW", False),
                patch.object(module.gen_config, "baseline_window", 2),
                patch.object(
                    module.analysis_utils,
                    "convert_time_from_frame_to_s",
                    use_supplied_seconds,
                ),
                patch.object(
                    module.analysis_utils,
                    "identify_blocks_trials",
                    side_effect=lambda data, _: data,
                ),
            ):
                result = module.run_data_aggregation()

            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result["Mean 2 s before"].iloc[0], 6.0)
            self.assertAlmostEqual(result["Mean CR"].iloc[0], 7.0)
            self.assertAlmostEqual(result["Normalized vigor"].iloc[0], 7.0 / 6.0)

    def test_stage_five_us_windows_share_negative_response_endpoint(self) -> None:
        module = self.normalized
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            all_fish = root / "all-fish"
            pooled = root / "pooled"
            all_fish.mkdir()
            pooled.mkdir()
            time_column = module.gen_config.time_trial_frame_label
            frame = pd.DataFrame(
                {
                    "Strain": ["strain"] * 3,
                    "Age (dpf)": [7] * 3,
                    "Exp.": ["delay"] * 3,
                    "ProtocolRig": ["rig"] * 3,
                    "Day": ["day"] * 3,
                    "Fish no.": [1] * 3,
                    "Fish": ["fish-1"] * 3,
                    "Block name": ["Train"] * 3,
                    "Trial number": [1] * 3,
                    time_column: [-3.0, -1.0, 0.0],
                    "Scaled vigor (AU)": [0.0, 0.0, 0.0],
                    "Vigor (deg/ms)": [2.0, 10.0, 4.0],
                    "US beg": [0.0, 0.0, 0.0],
                }
            )
            frame.to_pickle(all_fish / "delay_US_input.pkl", compression="gzip")

            with (
                patch.object(module, "path_all_fish", all_fish),
                patch.object(module, "path_pooled_data", pooled),
                patch.object(module, "cond_types", ["delay"]),
                patch.object(module, "cr_window", [0.0, 1.0]),
                patch.object(module, "blocks_dict", {}),
                patch.object(module, "csus", "US"),
                patch.object(module, "RUN_PROCESS", True),
                patch.object(module, "APPLY_FISH_DISCARD", False),
                patch.object(module, "APPLY_MAX_NAN_FRAC_PER_WINDOW", False),
                patch.object(module.gen_config, "baseline_window", 2),
                patch.object(
                    module.analysis_utils,
                    "convert_time_from_frame_to_s",
                    side_effect=lambda data: data.rename(
                        columns={time_column: "Trial time (s)"}
                    ),
                ),
                patch.object(
                    module.analysis_utils,
                    "identify_blocks_trials",
                    side_effect=lambda data, _: data,
                ),
            ):
                result = module.run_data_aggregation()

            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result["Mean 2 s before"].iloc[0], 6.0)
            self.assertAlmostEqual(result["Mean CR"].iloc[0], 7.0)
            self.assertAlmostEqual(result["Normalized vigor"].iloc[0], 7.0 / 6.0)

    def test_stage_five_pooled_loaders_use_first_or_newest_match(self) -> None:
        module = self.normalized
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "NV per trial per fish_a_b_CS_allFish.pkl"
            newest = root / "NV per trial per fish_c_d_CS_allFish.pkl"
            pd.DataFrame({"source": ["first"]}).to_pickle(first, compression="gzip")
            pd.DataFrame({"source": ["newest"]}).to_pickle(newest, compression="gzip")
            first.touch()
            newest.touch()
            newest_mtime = newest.stat().st_mtime + 10
            os.utime(newest, (newest_mtime, newest_mtime))
            glob_order = [
                path
                for path in root.glob("*.pkl")
                if "NV per trial per fish" in path.stem
                and "_allFish" in path.stem
                and path.stem.endswith("_CS_allFish")
            ]

            with (
                patch.object(module, "path_pooled_data", root),
                patch.object(module, "APPLY_FISH_DISCARD", False),
                patch.object(module, "APPLY_MAX_NAN_FRAC_PER_WINDOW", False),
                patch.object(module, "csus", "CS"),
            ):
                first_loaded = module.load_first_pooled()
                newest_loaded = module.load_latest_pooled()

            expected_first = pd.read_pickle(glob_order[0], compression="gzip")
            pd.testing.assert_frame_equal(first_loaded, expected_first)
            self.assertEqual(newest_loaded["source"].tolist(), ["newest"])

    def test_stage_five_mixed_model_errors_are_returned_not_raised(self) -> None:
        module = self.normalized
        frame = pd.DataFrame({"Fish_ID": ["fish-1"], "value": [1.0]})

        with patch.object(module.smf, "mixedlm", side_effect=RuntimeError("fit failed")):
            result, error = module.run_mixed_model(
                frame,
                "value ~ 1",
                "Fish_ID",
                re_formula="~value",
                method="lbfgs",
            )

        self.assertIsNone(result)
        self.assertEqual(error, "fit failed")

    def test_stage_five_model_formulas_and_corrections_are_fixed(self) -> None:
        module = self.normalized
        source = inspect.getsource(module.run_trial_by_trial)
        self.assertIn(
            "Log_Response ~ Log_Baseline + C(Condition, Treatment('{ref_cond}')) * C(Block_name)",
            source,
        )
        self.assertIn(
            "Log_Response ~ Log_Baseline + C(Condition, Treatment('{ref_cond}')) * Trial_Centered",
            source,
        )
        self.assertIn(
            "Log_Response ~ Log_Baseline + C(Condition, Treatment('{ref_cond}'))",
            source,
        )
        self.assertIn('method="fdr_bh"', source)
        self.assertFalse(module.APPLY_MAX_NAN_FRAC_PER_WINDOW)
        self.assertEqual(module.n_boot, 100)
        self.assertEqual(module.TRIAL_LINE_KW["seed"], 10)


if __name__ == "__main__":
    unittest.main()
