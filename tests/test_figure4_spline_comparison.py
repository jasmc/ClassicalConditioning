import unittest
import pandas as pd
from scripts.paper.figure_04.exports.summarize_spline_comparison import fish_scores,adjacent_correlation

class ComparisonTests(unittest.TestCase):
    def test_scores_weight_fish_not_observation_counts(self):
        frame=pd.DataFrame({'fish_id':['a','b','b','b'],'display_group':['Control']*4,'fold':[1]*4,
            'signed':[2.,0.,0.,0.],'predicted':[0.,0.,0.,0.]})
        scores=fish_scores(frame)
        self.assertEqual(scores.mse.mean(),2.)
        self.assertEqual(scores.observed_bins.tolist(),[1,3])

    def test_lag_pairs_respect_missing_time_bins_and_fish_identity(self):
        frame=pd.DataFrame({'fish_id':['a']*4+['b']*2,'time_s':[.25,.75,1.75,2.25,.25,.75],
            'residual':[1.,3.,2.,4.,-1.,-3.]})
        expected=pd.Series([3.,4.,-3.]).corr(pd.Series([1.,2.,-1.]))
        self.assertAlmostEqual(adjacent_correlation(frame,'residual'),expected)

if __name__=='__main__':unittest.main()
