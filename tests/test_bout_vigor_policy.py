import unittest
import numpy as np
import pandas as pd
from classical_conditioning.analysis.bout_vigor import mask_bout_vigor, bout_only_trial_outcomes, bout_only_profiles
from classical_conditioning.analysis.temporal_profiles import aggregate_event_profiles, CANDIDATE_COLUMNS, TemporalProfileConfig
from classical_conditioning.figures.example_traces import prepare_example_trace_data


class BoutVigorPolicyTests(unittest.TestCase):
    def test_rest_and_invalid_extremes_never_enter_aggregation(self):
        source = np.array([2., 1e99, 4., 1e99])
        masked = mask_bout_vigor(source, [True, True, True, False], [True, False, True, True])
        self.assertEqual(np.nanmean(masked), 3.)
        self.assertAlmostEqual(np.nanmedian(np.log(masked)), np.log(8)/2)
        self.assertEqual(source[1], 1e99)
        self.assertTrue(np.isnan(mask_bout_vigor(source, [True]*4, [False]*4)).all())

    def test_old_outcome_aliases_use_bout_fields_and_preserve_empty_windows(self):
        old = pd.DataFrame({"baseline_conditional_intensity": [2., np.nan], "conditional_intensity": [4., np.nan], "baseline_total_activity": [1., 0.], "response_total_activity": [2., 0.]})
        new = bout_only_trial_outcomes(old)
        self.assertEqual(new.response_total_activity[0]/new.baseline_total_activity[0], 2.)
        self.assertTrue(np.isnan(new.response_total_activity[1]))
        self.assertEqual(old.response_total_activity[1], 0.)

    def test_old_scaling_is_not_silently_reused(self):
        data = bout_only_profiles(pd.DataFrame({"Conditional intensity mean": [2., np.nan], "Total activity mean": [1., 0.], "Scaled total activity": [.5, 0.]}))
        self.assertTrue(data["Scaled total activity"].isna().all())
        self.assertTrue(np.isnan(data["Total activity mean"][1]))

    def test_temporal_means_and_scaling_ignore_rest_before_calculation(self):
        times = np.arange(-20000, 2000, 100)
        frames = pd.DataFrame({"AbsoluteTime": times, "FrameStep": 1, "DeltaTimeMs": 100., **{column: np.where(np.arange(len(times))%2, 1e99, np.linspace(1., 4., len(times))) for column in CANDIDATE_COLUMNS}})
        movement = pd.DataFrame({"AbsoluteTime": times, "valid": True, "moving": np.arange(len(times))%2 == 0, "bout_id": np.where(np.arange(len(times))%2 == 0, np.arange(len(times))+1, 0)})
        protocol = pd.DataFrame({"Type": ["Cycle"], "Beg": [0], "End": [100]})
        config = TemporalProfileConfig(window_start_s=-20, window_end_s=2, bin_width_s=.5)
        a = aggregate_event_profiles(frames, protocol, config=config, movement_state=movement)
        changed = frames.copy()
        changed.loc[~movement.moving, list(CANDIDATE_COLUMNS)] = -1e99
        b = aggregate_event_profiles(changed, protocol, config=config, movement_state=movement)
        np.testing.assert_allclose(a[["Total activity mean", "Scaled total activity"]], b[["Total activity mean", "Scaled total activity"]], equal_nan=True)
        self.assertTrue(a["Total activity mean"].lt(5).all())
        with self.assertRaisesRegex(ValueError, "shared movement"):
            aggregate_event_profiles(frames, protocol, config=config)

    def test_trace_keeps_rest_as_gap(self):
        frames = pd.DataFrame({"FrameID": [1,2,3], "AbsoluteTime": [-1000,0,11000], "angle0": [0.,1.,0.]})
        metrics = frames[["FrameID", "AbsoluteTime"]].assign(legacy_distal_angular_speed_rad_per_ms=[2.,1e99,4.])
        movement = frames[["FrameID", "AbsoluteTime"]].assign(valid=True, moving=[True,False,True])
        protocol = pd.DataFrame({"Type": ["Cycle"], "Beg": [0], "End": [10000]})
        data,_ = prepare_example_trace_data(frames, metrics, protocol, metric_id="legacy_distal_angular_speed", trial_numbers=[1], tail_point=0, movement_state=movement)
        self.assertTrue(np.isnan(data.Vigor.iloc[1]))
