"""Check the distinct mean and historical LogMedian window definitions."""
import importlib.util
from pathlib import Path
import unittest
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
spec=importlib.util.spec_from_file_location('logmedian_review',Path(__file__).resolve().parents[1]/'scripts/render_figure2_delay_logmedian.py')
review=importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class WindowSummaryTests(unittest.TestCase):
    def test_large_value_affects_mean_but_not_logmedian(self):
        result=review.window_summary([1,1,10])
        self.assertEqual(result['mean'],4)
        self.assertEqual(result['median'],1)
        self.assertEqual(result['logmedian'],0)
        self.assertNotEqual(np.log(result['mean']),result['logmedian'])

    def test_logs_positive_frames_once_and_no_bout_is_missing(self):
        result=review.window_summary([0,.1,.1,10,np.nan])
        self.assertAlmostEqual(result['logmedian'],np.log(.1))
        self.assertEqual(result['positive_count'],3)
        self.assertEqual(result['nonpositive_count'],1)
        empty=review.window_summary([np.nan])
        self.assertTrue(np.isnan(empty['mean']))
        self.assertTrue(np.isnan(empty['logmedian']))

    def test_even_count_logmedian_is_not_log_of_interpolated_raw_median(self):
        result=review.window_summary([1,4])
        self.assertEqual(result['median'],2.5)
        self.assertAlmostEqual(np.exp(result['logmedian']),2)

    def test_mean_log_is_distinct_from_log_mean_and_ignores_nonpositive(self):
        result=review.window_summary([0,1,1,10,np.nan])
        self.assertAlmostEqual(result['logmean'],np.log(10)/3)
        self.assertNotEqual(result['logmean'],np.log(result['mean']))
        self.assertTrue(np.isnan(review.window_summary([0,np.nan])['logmean']))

    def test_mean_log_difference_cancels_unit_conversion(self):
        response=np.array([1.,2.,8.]); baseline=np.array([.5,1.,3.])
        effect=review.window_summary(response)['logmean']-review.window_summary(baseline)['logmean']
        converted=review.window_summary(response*1000)['logmean']-review.window_summary(baseline*1000)['logmean']
        self.assertAlmostEqual(effect,converted)


if __name__=='__main__': unittest.main()
