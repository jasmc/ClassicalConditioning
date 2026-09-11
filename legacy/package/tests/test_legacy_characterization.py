from __future__ import annotations

import unittest

import pandas as pd

from classical_conditioning.preprocessing.legacy_characterization import (
    make_camera_tracking_sync_fixture,
    make_interpolation_fixture,
    make_legacy_angle_frame,
)
from classical_conditioning.preprocessing.legacy_equivalence import (
    analysis_utils_reference_steps,
    compare_synchronize_and_interpolate,
    operation_comparisons_to_dict,
    run_legacy_angle_operation_chain,
)
from classical_conditioning.preprocessing.legacy_v1 import (
    LegacyPreprocessingConfig,
    interpolate_legacy,
    synchronize_legacy,
)


class LegacyCharacterizationEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LegacyPreprocessingConfig(
            angle_point_count=4,
            temporal_filter_frames=3,
            bout_max_window_frames=3,
            bout_min_window_frames=3,
            bout_threshold_primary_deg_per_ms=0.5,
            minimum_bout_duration_frames=0,
            minimum_interbout_frames=1,
        )
        self.references = analysis_utils_reference_steps(self.config)

    def test_angle_chain_matches_analysis_utils_for_characterization_cases(self) -> None:
        cases = (
            "constant",
            "one_moving_point",
            "all_moving_together",
            "opposing_local_points",
            "no_movement",
            "short_bout_pulse",
            "two_close_pulses",
        )
        for case in cases:
            with self.subTest(case=case):
                frame = make_legacy_angle_frame(
                    case,
                    row_count=60,
                    point_count=4,
                )
                report = operation_comparisons_to_dict(
                    run_legacy_angle_operation_chain(
                        frame,
                        self.config,
                        self.references,
                    )
                )
                self.assertTrue(
                    report["all_equal"],
                    f"{case}: {report['first_divergence_operation']} "
                    f"{report['operations']}",
                )

    def test_synchronize_and_interpolate_match_analysis_utils(self) -> None:
        tracking, camera = make_camera_tracking_sync_fixture()
        comparisons = compare_synchronize_and_interpolate(
            tracking,
            camera,
            expected_framerate=700.0,
            predicted_framerate=700.0,
        )
        report = operation_comparisons_to_dict(comparisons)
        self.assertTrue(report["all_equal"], report)

        # Direct fixture path also stays self-consistent.
        synced = synchronize_legacy(*make_camera_tracking_sync_fixture())
        interpolated = interpolate_legacy(synced, 700.0, 700.0)
        self.assertGreater(len(interpolated), 0)
        self.assertIn("Trial time (frame) [700 FPS]", interpolated.columns)

        source = make_interpolation_fixture()
        again = interpolate_legacy(source, 700.0, 700.0)
        self.assertEqual(
            list(again.columns),
            [
                "Trial time (frame) [700 FPS]",
                "AbsoluteTime",
                "Angle of point 0 (deg)",
                "Angle of point 1 (deg)",
            ],
        )


if __name__ == "__main__":
    unittest.main()
