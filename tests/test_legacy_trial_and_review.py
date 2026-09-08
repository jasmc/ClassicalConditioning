from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.figures import (
    FigureMode,
    build_legacy_preprocessing_review_figure,
)
from classical_conditioning.preprocessing.legacy_characterization import (
    make_presegment_fixture,
    make_stimulus_timeline_fixture,
)
from classical_conditioning.preprocessing.legacy_equivalence import (
    compare_block_assignment,
    compare_scaled_vigor_roundtrip,
    compare_stimulus_annotation,
    compare_trial_segmentation,
)
from classical_conditioning.preprocessing.legacy_v1 import (
    TIME_COLUMN,
    VIGOR_COLUMN,
    SCALED_VIGOR_COLUMN,
    assign_blocks_legacy,
    segment_trials_legacy,
)


class StimulusAndTrialEquivalenceTests(unittest.TestCase):
    def test_stimulus_annotation_matches_analysis_utils(self) -> None:
        frame, protocol = make_stimulus_timeline_fixture()
        comparison = compare_stimulus_annotation(frame, protocol)
        self.assertTrue(comparison.equal, comparison.detail)

    def test_segmentation_blocks_and_scaling_on_presegment_fixture(self) -> None:
        frame, config = make_presegment_fixture()
        segmentation = compare_trial_segmentation(frame, config)
        self.assertTrue(segmentation.equal, segmentation.detail)

        segmented = segment_trials_legacy(frame, config)
        self.assertGreater(len(segmented), 0)
        self.assertEqual(
            set(segmented["Trial type"].astype(str)),
            {"CS", "US"},
        )

        blocks = compare_block_assignment(segmented, experiment_name="allDelay")
        self.assertTrue(blocks.equal, blocks.detail)
        assigned = assign_blocks_legacy(segmented, "allDelay")
        self.assertIn("Pre-train", set(assigned["Block name"].astype(str)))
        self.assertIn("Train 1", set(assigned["Block name"].astype(str)))

        # Ensure baseline window exists inside each segmented trial for scaling.
        scaled_input = assigned.copy()
        scaled_input[VIGOR_COLUMN] = np.linspace(
            0.05,
            0.55,
            len(scaled_input),
            dtype=np.float32,
        )
        scaling = compare_scaled_vigor_roundtrip(scaled_input, config)
        self.assertTrue(scaling.equal, scaling.detail)


class LegacyReviewFigureTests(unittest.TestCase):
    def test_builds_static_review_figure_from_synthetic_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            recording_id = "20260101_01"
            processed = project / "Processed data" / recording_id
            processed.mkdir(parents=True)
            rows = 40
            frame = pd.DataFrame(
                {
                    TIME_COLUMN: np.tile(np.arange(-10, 10), 2),
                    VIGOR_COLUMN: np.linspace(0.0, 1.0, rows, dtype=np.float32),
                    SCALED_VIGOR_COLUMN: np.linspace(0.0, 1.0, rows, dtype=np.float32),
                    "Trial type": ["CS"] * 20 + ["US"] * 20,
                    "Trial number": [5] * 20 + [18] * 20,
                    "Block name": ["Pre-train"] * 20 + ["Train 1"] * 20,
                    "Bout": [False] * 10 + [True] * 5 + [False] * 25,
                }
            )
            pq.write_table(
                pa.Table.from_pandas(frame, preserve_index=False),
                processed / "samples_legacy-v1.parquet",
                compression="zstd",
            )
            result = build_legacy_preprocessing_review_figure(
                project,
                recording_id,
                mode=FigureMode.STATIC,
                overwrite=True,
            )
            self.assertEqual(result.trial_counts.get("CS"), 20)
            self.assertTrue(any(path.suffix == ".png" for path in result.outputs))
            self.assertTrue(result.summary_path.is_file())


if __name__ == "__main__":
    unittest.main()
