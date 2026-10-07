"""Scientific invariants for the revised per-trial displayed-bin reference."""
import unittest
import numpy as np
import pandas as pd
from display_bin_centring import centre_trial_heatmap_bins

class DisplayReferenceTests(unittest.TestCase):
    def test_trial_specific_reference_and_baseline_boundaries(self):
        p=pd.DataFrame({'trial':[1]*4+[2]*4,
            'bin_center_s':[-15.25,-14.75,-.25,.25]*2,
            'signed_bout_log_bin':[100.,-3.,-1.,200.,-100.,5.,9.,20.],
            'baseline_log_median':[-2.]*4+[3.]*4})
        q=centre_trial_heatmap_bins(p)
        np.testing.assert_allclose(q.signed_bout_log_bin,[102.,-1.,1.,202.,-107.,-2.,2.,13.])
        self.assertTrue(q.display_baseline_bin_count.eq(2).all())
        np.testing.assert_allclose(q.display_baseline_log_bin_median,[-4.]*4+[10.]*4)
        # Preserve within-trial amplitude differences, not only zero median.
        for t in [1,2]:
            np.testing.assert_allclose(np.diff(q[q.trial.eq(t)].signed_bout_log_bin),np.diff(p[p.trial.eq(t)].signed_bout_log_bin))

    def test_finite_only_missingness_and_missing_reference(self):
        p=pd.DataFrame({'trial':[1]*4+[2]*4,'bin_center_s':[-14.75,-.25,.25,.75]*2,
            'signed_bout_log_bin':[np.nan,2.,np.inf,7.,np.nan,np.nan,4.,np.nan],
            'baseline_log_median':[0.]*8})
        q=centre_trial_heatmap_bins(p)
        np.testing.assert_allclose(q.signed_bout_log_bin,[np.nan,0.,np.nan,5.,np.nan,np.nan,np.nan,np.nan],equal_nan=True)
        self.assertTrue(q[q.trial.eq(2)].display_baseline_bin_count.eq(0).all())

    def test_recovery_of_uncentred_bins_and_input_preservation(self):
        p=pd.DataFrame({'trial':[1]*3,'bin_center_s':[-14.75,-.25,.25],
            'signed_bout_log_bin':[-.6,.2,.7],'baseline_log_median':[-2.]*3,
            'eligible_frames':[1,900,20]})
        original=p.copy(deep=True);q=centre_trial_heatmap_bins(p)
        # Reference gives each DISPLAYED bin equal weight, independent of support count.
        self.assertAlmostEqual(q.iloc[0].display_centre_offset_from_previous,-.2)
        np.testing.assert_allclose(q.signed_bout_log_bin,q.uncentred_bout_log_bin-q.display_baseline_log_bin_median)
        pd.testing.assert_frame_equal(p,original)
        np.testing.assert_array_equal(q.eligible_frames,p.eligible_frames)

    def test_duplicate_and_mixed_references_rejected(self):
        p=pd.DataFrame({'trial':[1,1],'bin_center_s':[-.25,-.25],
            'signed_bout_log_bin':[0.,1.],'baseline_log_median':[0.,0.]})
        with self.assertRaisesRegex(ValueError,'Duplicate'):centre_trial_heatmap_bins(p)
        p.loc[1,'bin_center_s']=.25;p.loc[1,'baseline_log_median']=1.
        with self.assertRaisesRegex(ValueError,'inconsistent'):centre_trial_heatmap_bins(p)

if __name__=='__main__':unittest.main()
