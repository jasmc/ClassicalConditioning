import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from classical_conditioning.preprocessing.acquisition_timing import (
    estimate_camera_cadence, presumed_acquisition_times)


class AcquisitionTimingTests(unittest.TestCase):
    def camera(self):
        dt = np.full(79,1.4)
        # Delay and catch-up leave frame cadence unchanged without losing IDs.
        dt[19] = 8.4
        dt[20:27] = .4
        elapsed = 100+np.r_[0,np.cumsum(dt)]
        return pd.DataFrame({'FrameID':np.arange(1000,1080), 'ElapsedTime':elapsed,
                             'AbsoluteTime':10000+elapsed},index=np.arange(80)*3)

    def test_buffered_arrival_is_not_lost_frames(self):
        camera=self.camera()
        cadence=estimate_camera_cadence(camera)
        self.assertAlmostEqual(cadence.interval_ms,1.4)
        self.assertFalse(cadence.has_frame_loss_evidence)
        elapsed,absolute=presumed_acquisition_times(camera,cadence)
        np.testing.assert_allclose(np.diff(elapsed),1.4)
        np.testing.assert_allclose(np.diff(absolute),1.4)
        self.assertAlmostEqual(cadence.maximum_delay_ms,7.)

    def test_missing_ids_detected_and_positions_do_not_assume_contiguity(self):
        camera=self.camera().drop(index=self.camera().index[40])
        cadence=estimate_camera_cadence(camera)
        self.assertAlmostEqual(cadence.interval_ms,1.4)
        self.assertEqual(cadence.missing_frame_ids,1)
        self.assertTrue(cadence.has_frame_loss_evidence)
        with self.assertRaisesRegex(ValueError,'Frame-loss'):
            presumed_acquisition_times(camera,cadence)

    def test_rejects_no_stable_anchor_and_duplicate_ids(self):
        camera=pd.DataFrame({'FrameID':np.arange(8),'ElapsedTime':np.cumsum([1,2]*4)})
        with self.assertRaisesRegex(ValueError,'stable'):
            estimate_camera_cadence(camera)
        camera=self.camera()
        camera.iloc[3,camera.columns.get_loc('FrameID')]=camera.FrameID.iloc[2]
        with self.assertRaisesRegex(ValueError,'strictly increase'):
            estimate_camera_cadence(camera)

    def test_buffer_capacity_evidence_separate_from_missing_ids(self):
        cadence=estimate_camera_cadence(self.camera(),buffer_size=4)
        self.assertEqual(cadence.missing_frame_ids,0)
        self.assertTrue(cadence.buffer_capacity_exceeded)

    def test_resampled_clock_and_protocol_share_expected_rate(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'legacy/helpers'))
        import analysis_utils
        # Source is 800 FPS, output 700 FPS; arrival timestamp includes jitter.
        source=pd.DataFrame({'FrameID':np.arange(81),
                             'AbsoluteTime':10000+np.arange(81)*1.25,
                             'ElapsedTime':np.arange(81)*1.25,
                             'Angle of point 0 (deg)':np.arange(81,dtype=float)})
        source.loc[20,'AbsoluteTime']+=1.
        output=analysis_utils.interpolate_data(source,700.,800.)
        np.testing.assert_allclose(np.diff(output.AbsoluteTime),1000/700,atol=1e-10)
        # Frame 40 acquired at 50 ms maps to expected-grid index 35.
        self.assertAlmostEqual(output.loc[35,'AbsoluteTime'],10050.)
        self.assertAlmostEqual(output.loc[35,'Angle of point 0 (deg)'],40.)
        protocol=pd.DataFrame({'beg (ms)':[10049.9], 'end (ms)':[10059.9]},index=['Cycle'])
        labelled=analysis_utils.stim_in_data(output,protocol)
        self.assertEqual(labelled.index[np.asarray(labelled['CS beg'])==1].tolist(),[35])
        np.testing.assert_allclose(np.diff(labelled.AbsoluteTime),1000/700,atol=1e-10)
        self.assertEqual(analysis_utils.framerate_and_reference_frame(self.camera(),'synthetic')[2],False)

    def test_camera_reference_survives_missing_leading_tracking_rows(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'legacy/helpers'))
        import analysis_utils
        camera=self.camera(); cadence=estimate_camera_cadence(camera)
        elapsed,absolute=presumed_acquisition_times(camera,cadence)
        original_arrival=float(camera.AbsoluteTime.iloc[20])
        camera['ElapsedTime']=elapsed; camera['AbsoluteTime']=absolute
        tracking=pd.DataFrame({'FrameID':camera.FrameID.iloc[20:40].to_numpy(),
                               'Angle of point 0 (deg)':np.arange(20,dtype=float)})
        merged=analysis_utils.merge_camera_with_data(tracking,camera)
        output=analysis_utils.interpolate_data(merged,700,cadence.framerate)
        self.assertAlmostEqual(output.AbsoluteTime.iloc[0],absolute[20])
        self.assertGreater(original_arrival-output.AbsoluteTime.iloc[0],1.)


if __name__=='__main__': unittest.main()
