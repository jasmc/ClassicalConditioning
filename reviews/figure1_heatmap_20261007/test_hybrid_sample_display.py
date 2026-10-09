import unittest
import numpy as np
from hybrid_sample_display import hybrid_sample_values


class HybridSampleTests(unittest.TestCase):
    def test_each_bin_gets_one_vote_despite_unequal_sample_counts(self):
        times=np.r_[np.linspace(-14.99,-14.60,100),-14.25,-13.75,.25]
        values=np.r_[np.zeros(100),1.,10.,4.]
        samples,bins,counts,meta=hybrid_sample_values(times,values)
        self.assertEqual(np.median(values[times<0]),0.)
        self.assertEqual(meta['reference'],1.)
        self.assertEqual(meta['baseline_count'],3)
        self.assertEqual(counts[10],100)
        self.assertTrue(np.all(samples['centred_log'][:100]==-1))
        self.assertAlmostEqual(np.nanmedian((bins-meta['reference'])[10:40]),0)
        self.assertLess(np.median(samples['C'][times<0]),0)

    def test_eligible_gaps_stay_missing_in_sample_display(self):
        times=np.array([-14.75,-14.6,-14.25,-13.75,.25])
        values=np.array([0.,np.nan,1.,10.,100.])
        samples,bins,counts,meta=hybrid_sample_values(times,values)
        self.assertTrue(np.isnan(samples['C'][1]) and np.isnan(samples['D'][1]))
        self.assertEqual(samples['C'][-1],1.)
        self.assertEqual(counts[10],1)
        self.assertTrue(np.isnan(bins[9]))

    def test_one_supported_baseline_bin_has_no_quantile_range(self):
        samples,bins,counts,meta=hybrid_sample_values([-14.75,.25],[2.,3.])
        self.assertEqual(meta['reference'],2.)
        self.assertTrue(np.isnan(samples['C']).all() and np.isnan(samples['D']).all())
        np.testing.assert_allclose(samples['centred_log'],[0,1])


if __name__=='__main__':
    unittest.main()
