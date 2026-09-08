from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.legacy_standard_main import (
    RECIPE_ID,
    SCALED_VIGOR_COLUMN,
    TIME_COLUMN,
    VIGOR_COLUMN,
    LegacyStandardMainConfig,
    build_legacy_standard_main,
    transform_legacy_standard_main,
)
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cli import build_parser
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_legacy_grouping() -> ModuleType:
    path = REPOSITORY_ROOT / "3_FishGrouping.py"
    spec = importlib.util.spec_from_file_location(
        "legacy_standard_main_reference",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_samples(
    *,
    trial_type: str = "CS",
    trial_number: int = 5,
    times: list[int] | None = None,
) -> pd.DataFrame:
    times = times or [-20, -16, -15, -10, 0, 1]
    size = len(times)
    vigor = np.resize(
        np.asarray([1, 3, 5, 7, 9, 11], dtype="float32"),
        size,
    )
    bouts = np.resize(
        np.asarray([True, True, False, True, True, False]),
        size,
    )
    return pd.DataFrame(
        {
            "Strain": ["strain"] * size,
            "Age (dpf)": ["6"] * size,
            "Exp.": ["delay-special"] * size,
            "ProtocolRig": ["rig"] * size,
            "Day": ["20221115"] * size,
            "Fish no.": ["04"] * size,
            TIME_COLUMN: times,
            "CS beg": np.zeros(size, dtype="int32"),
            "CS end": np.zeros(size, dtype="int32"),
            "US beg": np.zeros(size, dtype="int32"),
            "US end": np.zeros(size, dtype="int32"),
            "Trial type": [trial_type] * size,
            "Trial number": np.full(size, trial_number, dtype="int32"),
            "Block name": [""] * size,
            "Angle of point 15 (deg)": np.arange(size, dtype="float32"),
            VIGOR_COLUMN: vigor,
            SCALED_VIGOR_COLUMN: np.zeros(size, dtype="float32"),
            "Bout beg": np.zeros(size, dtype=bool),
            "Bout end": np.zeros(size, dtype=bool),
            "Bout": bouts,
        }
    )


class LegacyStandardMainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.legacy = load_legacy_grouping()

    def test_matches_executable_standard_main_transform(self) -> None:
        source = make_samples()
        config = LegacyStandardMainConfig(
            rolling_window_frames=3,
            downsample_factor=2,
            baseline_window_frames=15,
        )
        actual = transform_legacy_standard_main(source, config=config)["CS"]

        reference = source.copy()
        reference["Fish"] = (
            reference["Day"].astype("string")
            + "_"
            + reference["Fish no."].astype("string")
        )
        reference["Exp."] = "delay"
        reference["Block name"] = "Pre-train"
        reference = reference.drop(columns="Trial type")
        expected = self.legacy.process_data(
            reference,
            window_size=config.rolling_window_frames,
            downsample_factor=config.downsample_factor,
            grouping_cols=["Fish", "Trial number"],
            time_col=TIME_COLUMN,
            baseline_window_frames=config.baseline_window_frames,
        )
        expected.loc[
            ~expected["Bout"],
            [VIGOR_COLUMN, SCALED_VIGOR_COLUMN],
        ] = np.nan
        expected = expected.loc[:, actual.columns].reset_index(drop=True)

        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_categorical=False,
            check_dtype=False,
        )
        self.assertEqual(actual[VIGOR_COLUMN].dtype, expected[VIGOR_COLUMN].dtype)
        self.assertEqual(
            actual[SCALED_VIGOR_COLUMN].dtype,
            expected[SCALED_VIGOR_COLUMN].dtype,
        )

    def test_filters_trials_outside_configured_analysis_blocks(self) -> None:
        outside = make_samples(trial_number=4)
        inside = make_samples(trial_number=5)
        source = pd.concat([outside, inside], ignore_index=True)

        result = transform_legacy_standard_main(
            source,
            config=LegacyStandardMainConfig(
                rolling_window_frames=1,
                downsample_factor=1,
                baseline_window_frames=15,
            ),
        )

        self.assertEqual(set(result["CS"]["Trial number"]), {5})
        self.assertEqual(result["CS"]["Block name"].astype(str).unique().tolist(), ["Pre-train"])

    def test_keeps_cs_and_us_with_overlapping_trial_numbers_separate(self) -> None:
        cs = make_samples(trial_type="CS", trial_number=18)
        us = make_samples(trial_type="US", trial_number=18)
        result = transform_legacy_standard_main(
            pd.concat([cs, us], ignore_index=True),
            config=LegacyStandardMainConfig(
                rolling_window_frames=1,
                downsample_factor=1,
                baseline_window_frames=15,
            ),
        )

        self.assertEqual(set(result), {"CS", "US"})
        self.assertEqual(result["CS"]["Block name"].astype(str).unique().tolist(), ["Train 1"])
        self.assertEqual(result["US"]["Block name"].astype(str).unique().tolist(), ["Train 1"])

    def test_builder_verifies_lineage_and_publishes_two_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            recording_id = "recording-1"
            source_dir = project_dir / "Processed data" / recording_id
            quality_dir = project_dir / "Quality checks" / recording_id
            metadata_dir = project_dir / "Metadata"
            source_dir.mkdir(parents=True)
            quality_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)

            times = list(range(-10_510, -10_498)) + list(range(-5, 7))
            cs = make_samples(
                trial_type="CS",
                trial_number=5,
                times=times,
            )
            us = make_samples(
                trial_type="US",
                trial_number=18,
                times=times,
            )
            source = pd.concat([cs, us], ignore_index=True)
            source_path = source_dir / "samples_legacy-v1.parquet"
            pq.write_table(
                pa.Table.from_pandas(source, preserve_index=False),
                source_path,
                compression="zstd",
            )
            source_hash = sha256_file(source_path)
            source_summary_path = quality_dir / "legacy-v1_preprocessing_summary.json"
            source_summary_path.write_text(
                json.dumps(
                    {
                        "recipe": "legacy-paper-v1",
                        "recording_id": recording_id,
                        "experiment": "allDelay",
                        "artifact": {"sha256": source_hash},
                    }
                ),
                encoding="utf-8",
            )
            source_marker_path = (
                metadata_dir / f"{recording_id}_legacy-v1_complete.json"
            )
            source_marker_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "recipe": "legacy-paper-v1",
                        "recording_id": recording_id,
                        "samples_sha256": source_hash,
                        "summary_sha256": sha256_file(source_summary_path),
                    }
                ),
                encoding="utf-8",
            )

            result = build_legacy_standard_main(
                project_dir,
                recording_id,
                read_batch_rows=7,
            )

            expected = transform_legacy_standard_main(source)
            self.assertEqual(set(result.samples_paths), {"CS", "US"})
            self.assertEqual(result.trial_counts, {"CS": 1, "US": 1})
            self.assertEqual(result.row_counts, {"CS": 3, "US": 3})
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            marker = json.loads(
                result.completion_marker_path.read_text(encoding="utf-8")
            )
            self.assertEqual(summary["recipe"], RECIPE_ID)
            self.assertFalse(summary["config"]["discard_list_applied_to_rows"])
            self.assertEqual(
                summary["config_sha256"],
                "8ee7ea95671f9dbafc8a150853c15ad301eba30647d06634b96eb53bd0408936",
            )
            self.assertEqual(
                summary["legacy_paper_config_sha256"],
                "0c4f133a874d9d87cf9fda1e7be7b3e07b148eaaf86a2b5e647bc4c63d24d6e9",
            )
            for alignment, path in result.samples_paths.items():
                pd.testing.assert_frame_equal(
                    pq.read_table(path).to_pandas(),
                    expected[alignment],
                )
                self.assertEqual(
                    sha256_file(path),
                    marker["artifact_sha256"][alignment],
                )
                self.assertTrue(
                    summary["artifacts"][alignment]["compression_lossless"]
                )
            self.assertEqual(
                sha256_file(result.summary_path),
                marker["summary_sha256"],
            )

            with self.assertRaises(FileExistsError):
                build_legacy_standard_main(
                    project_dir,
                    recording_id,
                    read_batch_rows=5,
                )

            cs_only = source.loc[source["Trial type"] == "CS"].reset_index(drop=True)
            pq.write_table(
                pa.Table.from_pandas(cs_only, preserve_index=False),
                source_path,
                compression="zstd",
            )
            source_hash = sha256_file(source_path)
            source_summary_path.write_text(
                json.dumps(
                    {
                        "recipe": "legacy-paper-v1",
                        "recording_id": recording_id,
                        "experiment": "allDelay",
                        "artifact": {"sha256": source_hash},
                    }
                ),
                encoding="utf-8",
            )
            source_marker_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "recipe": "legacy-paper-v1",
                        "recording_id": recording_id,
                        "samples_sha256": source_hash,
                        "summary_sha256": sha256_file(source_summary_path),
                    }
                ),
                encoding="utf-8",
            )

            replaced = build_legacy_standard_main(
                project_dir,
                recording_id,
                read_batch_rows=5,
                overwrite=True,
            )

            self.assertEqual(set(replaced.samples_paths), {"CS"})
            self.assertFalse(result.samples_paths["US"].exists())
            replaced_summary = json.loads(
                replaced.summary_path.read_text(encoding="utf-8")
            )
            self.assertEqual(set(replaced_summary["artifacts"]), {"CS"})

    def test_builder_rejects_tampered_source_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            recording_id = "recording-1"
            source_dir = project_dir / "Processed data" / recording_id
            quality_dir = project_dir / "Quality checks" / recording_id
            metadata_dir = project_dir / "Metadata"
            source_dir.mkdir(parents=True)
            quality_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)
            source_path = source_dir / "samples_legacy-v1.parquet"
            pq.write_table(
                pa.Table.from_pandas(make_samples(), preserve_index=False),
                source_path,
            )
            summary_path = quality_dir / "legacy-v1_preprocessing_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "recipe": "legacy-paper-v1",
                        "recording_id": recording_id,
                        "experiment": "allDelay",
                        "artifact": {"sha256": "tampered"},
                    }
                ),
                encoding="utf-8",
            )
            marker_path = metadata_dir / f"{recording_id}_legacy-v1_complete.json"
            marker_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "recipe": "legacy-paper-v1",
                        "recording_id": recording_id,
                        "samples_sha256": "tampered",
                        "summary_sha256": sha256_file(summary_path),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ArtifactIntegrityError):
                build_legacy_standard_main(project_dir, recording_id)

    def test_frozen_builder_rejects_parameter_changes(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "frozen configuration"):
            build_legacy_standard_main(
                Path("."),
                "recording",
                config=LegacyStandardMainConfig(rolling_window_frames=11),
            )

    def test_cli_exposes_only_explicit_standard_main_recipe(self) -> None:
        args = build_parser().parse_args(
            [
                "legacy-standard-main",
                "--project-dir",
                ".",
                "--recording-id",
                "recording",
            ]
        )
        self.assertEqual(args.recipe, RECIPE_ID)
        self.assertEqual(args.experiment, "allDelay")


if __name__ == "__main__":
    unittest.main()
