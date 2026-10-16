"""Scientific edge cases for the exploratory fish atlas."""
import unittest
import numpy as np
import pandas as pd
from classical_conditioning.analysis.review.individual_atlas import bin_trial,VARIANTS
from classical_conditioning.figures.individual_atlas import decision,select_examples,feature_value,g_trial_outcome


class IndividualAtlasTests(unittest.TestCase):
    def test_g_outcome_uses_logmedians_without_second_log_or_mean_ratio(self):
        data=pd.DataFrame({'baseline_logmedian':[-3.,-2.,-4.],
            'response_logmedian':[-2.,-2.,np.nan],
            'baseline_bout_mean':[100.,100.,100.],
            'response_bout_mean':[1.,2.,3.],'classifier_ratio':[.01,.02,.03]})
        np.testing.assert_allclose(g_trial_outcome(data),[1.,0.,np.nan],equal_nan=True)

    def test_immobility_is_observed_but_intensity_is_undefined(self):
        heat,movement,coverage,counts,_=bin_trial(
            np.array([-15000,0,500,1000]),np.array([1.,0.,0.,7.]),
            np.array([True,True,True,False]),np.array([True,False,False,True]),
            np.array([1000.,500.,500.,1000.]),0)
        self.assertTrue(np.isnan(heat[15]))
        self.assertEqual(movement[15],0)
        self.assertEqual(coverage[15],1)
        self.assertEqual(counts[15],0)
        self.assertTrue(np.isnan(movement[16]))
        self.assertEqual(coverage[16],0)

    def test_time_weighted_movement_and_bout_only_log_intensity(self):
        heat,movement,coverage,counts,centre=bin_trial(
            np.array([-15000,0,250]),np.array([2.,999.,8.]),
            np.ones(3,bool),np.array([True,False,True]),np.array([1000.,250.,750.]),0)
        self.assertAlmostEqual(centre,np.log(2))
        self.assertAlmostEqual(heat[15],np.log(4))
        self.assertEqual(movement[15],.75)
        self.assertEqual(coverage[15],1)
        self.assertEqual(counts[15],1)

    def test_unclassified_fish_is_not_a_nonlearner(self):
        table=pd.DataFrame({'Condition':['delay'],'Fish_ID':['one'],'learner_primary':[False]}).set_index(['Condition','Fish_ID'])
        self.assertEqual(decision({'legacy-wip':table},'legacy-wip','delay','missing')[0],'Unclassified')
        self.assertEqual(decision({},'legacy-wip','delay','one')[0],'Failed')

    def test_selection_does_not_invent_agreement_and_keeps_distinct_fish(self):
        inventory=pd.DataFrame({'condition_id':['delay']*5,'fish_id':list('abcde'),
                                'protocol_review_flag':[False]*4+[True]})
        results={}
        for variant in VARIANTS:
            frame=pd.DataFrame({'Condition':['delay']*5,'Fish_ID':list('abcde'),
                'learner_primary':[False]*5,'T_joint':[1,2,3,4,100],
                'decision_threshold':[2.5]*5})
            results[variant]=frame.set_index(['Condition','Fish_ID'])
        selected=select_examples(inventory,results)
        self.assertEqual(len({r['fish_id'] for r in selected}),4)
        self.assertNotIn('e',{r['fish_id'] for r in selected})
        self.assertFalse(any('all four rules flag' in r['reason'] for r in selected))
        self.assertEqual(feature_value(pd.Series({'Feature_acquisition':-.2}),'acquisition'),-.2)


if __name__=='__main__':unittest.main()
