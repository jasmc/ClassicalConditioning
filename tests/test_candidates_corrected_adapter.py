from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from classical_conditioning.preprocessing.candidates_corrected_v1 import (
    apply_corrected_validity_mask,
)
from classical_conditioning.preprocessing.candidates_v1 import (
    CANDIDATE_COLUMNS,
    CandidateMetricConfig,
    calculate_candidate_metrics,
)


class CorrectedCandidateAdapterTests(unittest.TestCase):
    def test_mask_invalidates_long_interval_kept_by_candidate_alone(self) -> None:
        config = CandidateMetricConfig(point_count=4)
        frame_ids = np.array([10, 11, 12], dtype=np.int64)
        # FrameStep==1 throughout, but last interval is longer than corrected max.
        elapsed = np.array([0.0, 1.4, 20.0], dtype=np.float64)
        x = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 4.0],
            ],
            dtype=np.float64,
        )
        y = np.zeros_like(x)
        angles = np.zeros_like(x)
        metrics, _ = calculate_candidate_metrics(
            frame_ids,
            elapsed,
            x,
            y,
            angles,
            config=config,
        )
        self.assertTrue(bool(metrics.loc[2, "valid_derivative"]))

        corrected_valid = np.array([False, True, False])
        corrected_step = np.array([0, 1, 1], dtype=np.int64)
        corrected_delta = np.array([np.nan, 1.4, 18.6], dtype=np.float64)
        masked = apply_corrected_validity_mask(
            metrics,
            corrected_derivative_valid=corrected_valid,
            corrected_frame_step=corrected_step,
            corrected_delta_time_ms=corrected_delta,
        )
        self.assertEqual(
            masked["valid_derivative"].tolist(),
            [False, True, False],
        )
        self.assertTrue(np.isnan(masked.loc[2, CANDIDATE_COLUMNS[0]]))
        self.assertFalse(np.isnan(masked.loc[1, CANDIDATE_COLUMNS[2]]))
        self.assertEqual(int(masked.loc[2, "FrameStep"]), 1)
        self.assertAlmostEqual(float(masked.loc[2, "DeltaTimeMs"]), 18.6)

    def test_mask_length_mismatch_is_rejected(self) -> None:
        metrics = pd.DataFrame(
            {
                "valid_derivative": [True, True],
                "FrameStep": [1, 1],
                "DeltaTimeMs": [1.0, 1.0],
                CANDIDATE_COLUMNS[0]: [0.1, 0.2],
                CANDIDATE_COLUMNS[1]: [0.1, 0.2],
                CANDIDATE_COLUMNS[2]: [0.1, 0.2],
                CANDIDATE_COLUMNS[3]: [0.1, 0.2],
                CANDIDATE_COLUMNS[4]: [0.1, 0.2],
                CANDIDATE_COLUMNS[5]: [0.1, 0.2],
            }
        )
        with self.assertRaisesRegex(ValueError, "length"):
            apply_corrected_validity_mask(
                metrics,
                corrected_derivative_valid=np.array([True]),
                corrected_frame_step=np.array([1]),
                corrected_delta_time_ms=np.array([1.0]),
            )


if __name__ == "__main__":
    unittest.main()
