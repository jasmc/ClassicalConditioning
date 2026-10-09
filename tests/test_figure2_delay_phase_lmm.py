"""Verify phase separation and the scheduled pre-reference contrast."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from patsy import dmatrix

spec=importlib.util.spec_from_file_location('phase_review',Path(__file__).resolve().parents[1]/'scripts/render_figure2_delay_phase_lmm.py')
review=importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class PhaseDefinitionTests(unittest.TestCase):
    def test_basis_does_not_cross_phase_boundaries(self):
        features=review.phase_features()
        self.assertEqual(features.loc[features.fit_phase.eq('Pre'),'trial_number'].tolist(),list(range(5,15)))
        self.assertEqual(features.loc[features.fit_phase.eq('Train'),'trial_number'].tolist(),list(range(15,65)))
        self.assertEqual(features.loc[features.fit_phase.eq('Test'),'trial_number'].tolist(),list(range(65,95)))
        for phase,prefix in [('Train','train'),('Test','test')]:
            self.assertTrue((features.loc[~features.fit_phase.eq(phase),[f'{prefix}_b{i}' for i in range(4)]]==0).all().all())
        self.assertTrue((features.loc[~features.fit_phase.eq('Pre'),'pre_trial']==0).all())
        self.assertAlmostEqual(features.loc[features.fit_phase.eq('Pre'),'pre_trial'].mean(),0)

    def test_training_effect_cannot_create_pretrend_and_reference_is_average(self):
        features=review.phase_features()
        cases=pd.concat([features.assign(condition_id=c) for c in ['control','delay']],ignore_index=True)
        cases['log_baseline']=0.2
        formula=review.PHASE.split('~',1)[1]
        x=dmatrix(formula,cases,return_type='dataframe')
        beta=pd.Series(0.,index=x.columns)
        # A pure Delay training shift must be zero throughout Pre, Test.
        term=next(c for c in x.columns if 'condition_id' in c and '[T.Train]' in c)
        beta[term]=-.4
        fake=SimpleNamespace(model=SimpleNamespace(data=SimpleNamespace(design_info=x.design_info)),fe_params=beta)
        data=pd.DataFrame({'log_baseline':[.2],'trial_center':[49.5],'trial_scale':[26.]})
        with patch.object(review,'_fixed_covariance',return_value=np.eye(len(beta))*.01):
            result,_,matrix=review.contrasts(fake,data,features)
        np.testing.assert_allclose(result.change_estimate.iloc[:10],0,atol=1e-12)
        np.testing.assert_allclose(result.loc[result.fit_phase.eq('Train'),'change_estimate'],.4,atol=1e-12)
        np.testing.assert_allclose(result.loc[result.fit_phase.eq('Test'),'change_estimate'],0,atol=1e-12)
        np.testing.assert_allclose(matrix[:10].mean(axis=0),0,atol=1e-12)


if __name__=='__main__': unittest.main()
