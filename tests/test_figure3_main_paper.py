"""Scientific boundaries of the descriptive Figure 3 acquisition contrast."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('figure3_main_builder',Path(__file__).parents[1]/'scripts/build_figure3_main_paper.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


class Figure3MainContrastTests(unittest.TestCase):
    def fixture(self):
        trials=[]
        for fish in ['observed','unestimable']:
            for trial in range(5,95):
                y=2. if trial<=14 else -1. if 40<=trial<=64 else 0.
                if fish=='unestimable':y=np.nan
                trials.append({'experiment_id':'allDelay','condition_id':'delay','fish_id':fish,
                    'trial_number':trial,'baseline_logmedian':-3.,'response_logmedian':-3.+y})
        inventory=pd.DataFrame({'experiment_id':['allDelay']*2,'condition_id':['delay']*2,'fish_id':['observed','unestimable']})
        return pd.DataFrame(trials),inventory

    def test_future_test_observations_do_not_enter_acquisition(self):
        trials,inventory=self.fixture();before=builder.acquisition_scores(trials,inventory)
        trials.loc[trials.trial_number.ge(65),'response_logmedian']=1e6
        after=builder.acquisition_scores(trials,inventory)
        np.testing.assert_allclose(before.acquisition_score,after.acquisition_score,equal_nan=True)
        self.assertEqual(float(after.iloc[0].acquisition_score),3.)

    def test_undefined_fish_is_retained_without_a_label(self):
        trials,inventory=self.fixture();result=builder.acquisition_scores(trials,inventory)
        self.assertEqual(len(result),2);row=result.loc[result.fish_id.eq('unestimable')].iloc[0]
        self.assertFalse(row.estimable);self.assertTrue(np.isnan(row.acquisition_score))
        self.assertNotIn('learner',result.columns)

    def test_score_is_robust_median_contrast_and_retains_coverage(self):
        trials,inventory=self.fixture()
        trials.loc[trials.fish_id.eq('observed')&trials.trial_number.eq(14),'response_logmedian']=10000
        trials.loc[trials.fish_id.eq('observed')&trials.trial_number.eq(64),'response_logmedian']=-10000
        row=builder.acquisition_scores(trials,inventory).iloc[0]
        self.assertEqual(row.acquisition_score,3.)
        self.assertEqual((row.pre_n,row.late_training_n),(10,25))


if __name__=='__main__':unittest.main()
