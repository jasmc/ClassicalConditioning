import unittest
import numpy as np
from quantile_candidates import trial_quantile_variants

class QuantileTests(unittest.TestCase):
    def test_skewed_baseline_is_not_half_under_legacy_formula(self):
        v=np.array([-1.,-.2,0.,.1,2.,np.nan,5.]);mask=np.array([1,1,1,1,1,0,0],bool)
        a,b,m=trial_quantile_variants(v,mask)
        self.assertNotAlmostEqual(m['baseline_median_linear'],.5)
        self.assertAlmostEqual(np.median(b[mask]),0.)
        self.assertTrue(np.isnan(a[5]) and np.isnan(b[5]))
        self.assertEqual(a[-1],1.);self.assertEqual(b[-1],1.)
        self.assertTrue(np.all(np.diff(a[np.isfinite(a)])>=0))

    def test_reference_is_baseline_only_and_affine_invariant(self):
        v=np.r_[np.arange(11,dtype=float),1000.];mask=np.r_[np.ones(11,bool),False]
        a,b,m=trial_quantile_variants(v,mask)
        self.assertEqual(m['p10'],1.);self.assertEqual(m['p90'],9.)
        x,y,_=trial_quantile_variants(v*3+17,mask)
        np.testing.assert_allclose(a,x);np.testing.assert_allclose(b,y)

    def test_degenerate_and_small_baselines_remain_explicitly_missing(self):
        for v in [np.ones(5),np.array([1.,np.nan,np.nan,2.,3.])]:
            mask=np.array([True,True,True,False,False])
            a,b,m=trial_quantile_variants(v,mask)
            self.assertTrue(np.isnan(a).all() and np.isnan(b).all())
            self.assertNotEqual(m['status'],'ok')

    def test_even_skewed_baseline_preserves_sample_median_after_clipping(self):
        v=np.array([-3.,-1.,1.,20.,100.,np.nan]);mask=np.array([1,1,1,1,0,0],bool)
        _,b,m=trial_quantile_variants(v,mask)
        self.assertAlmostEqual(np.median(b[mask]),0.)
        self.assertEqual(m['symmetric_radius'],m['p90']-m['p50'])
        self.assertEqual(b[4],1.)
        self.assertTrue(np.isnan(b[5]))

    def test_two_baseline_bins_are_retained_without_inventing_missingness(self):
        v=np.array([1.,4.,9.,np.nan]);mask=np.array([1,1,0,0],bool)
        a,b,m=trial_quantile_variants(v,mask)
        self.assertTrue(np.isfinite(a[:3]).all() and np.isfinite(b[:3]).all())
        self.assertAlmostEqual(np.median(b[mask]),0.)
        self.assertEqual(m['n'],2)

if __name__=='__main__':unittest.main()
