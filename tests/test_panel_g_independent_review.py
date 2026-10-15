"""Scientific invariants of the isolated G analysis, including missing trials."""
import unittest
import numpy as np
import pandas as pd

from classical_conditioning.analysis.inference.panel_g_review import (
    first_run, timing_bounds, prepare_input, simultaneous_band, phase_features,
)
from classical_conditioning.analysis.inference.panel_g_review_extensions import simulate
from classical_conditioning.analysis.inference.panel_g_individual import fit_individual, individual_curve
from classical_conditioning.analysis.inference.panel_g_missingness import scheduled_individual_timing


class TimingTests(unittest.TestCase):
    def test_missing_schedule_trial_does_not_join_a_run(self):
        self.assertEqual(first_run([20,21,23],[.2,.2,.2],.1),float("inf"))
        self.assertEqual(first_run([20,21,22],[.2,.2,.2],.1),20)
        self.assertEqual(first_run([20,21,22],[.2,.2,.2],.1,eligible=[True,False,True]),float("inf"))

    def test_run_cannot_cross_training_testing_boundary(self):
        self.assertEqual(first_run([63,64,65],[.2,.2,.2],.1),float("inf"))
        self.assertEqual(first_run([62,63,64],[.2,.2,.2],.1),62)
        self.assertEqual(first_run([12,13,14],[.2,.2,.2],.1),float("inf"))

    def test_crossing_and_evidence_are_distinct(self):
        trials=np.arange(15,65)
        mean=np.linspace(0,.3,50);lower=mean-.04;upper=mean+.04
        result=timing_bounds(trials,mean,lower,upper,.1)
        self.assertLess(result["earliest_compatible_trial"],result["estimate_trial"])
        self.assertLess(result["estimate_trial"],result["first_supported_trial"])
        self.assertEqual(first_run([20,21,22],[.1,.1,.1],.1),float("inf"))

    def test_failed_refits_are_not_zero_effect_draws(self):
        values=np.array([.2,.1]);se=np.array([.04,.02])
        draws=[{"success":True,"estimate":values+se,"se":se},
               {"success":True,"estimate":values-se,"se":se},
               {"success":False,"diagnostic":{"error":"singular"}}]
        lower,upper,critical,maxima=simultaneous_band(values,se,draws)
        self.assertEqual(len(maxima),2)
        np.testing.assert_allclose(lower,values-se);np.testing.assert_allclose(upper,values+se)

    def test_unobserved_trial_remains_a_possible_individual_onset(self):
        trials=np.arange(15,25)
        mean=np.array([0,0,0,.2,.2,.2,.2,.2,.2,.2])
        observed=np.ones(10,bool);observed[3]=False
        result=scheduled_individual_timing(trials,mean,mean-.02,mean+.02,observed,.1)
        self.assertEqual(result["estimate_trial"],18)
        self.assertEqual(result["earliest_compatible_trial"],18)
        self.assertEqual(result["first_observed_supported_trial"],19)
        self.assertTrue(result["prediction_in_estimated_run"])


class InputTests(unittest.TestCase):
    def test_single_log_difference_and_unavailable_outcomes(self):
        windows=pd.DataFrame({"fish_id":["a"]*3,"condition_id":["delay"]*3,
            "trial_number":[15,17,18],"baseline_logmedian":[-5,-5,np.nan],
            "response_logmedian":[-5.2,np.nan,-5.1]})
        result=prepare_input(windows)
        self.assertEqual(result.trial_number.tolist(),[15])
        self.assertAlmostEqual(result.z.iloc[0],-.2)
        self.assertRaises(ValueError,prepare_input,pd.concat([windows,windows.iloc[:1]]))

    def test_row_order_invariance_and_phase_isolation(self):
        data,_=simulate(11,"gradual",counts=(4,4))
        windows=data.drop(columns=[c for c in phase_features() if c!="trial_number"])
        a=prepare_input(windows);b=prepare_input(windows.sample(frac=1,random_state=2))
        pd.testing.assert_frame_equal(a,b)
        f=phase_features()
        self.assertTrue((f.loc[f.trial_number<15,[f"train_b{i}" for i in range(4)]]==0).all().all())
        self.assertTrue((f.loc[f.trial_number<65,[f"test_b{i}" for i in range(4)]]==0).all().all())


class IndividualTests(unittest.TestCase):
    def test_heldout_response_cannot_change_training_fit(self):
        from threadpoolctl import threadpool_limits
        data,_=simulate(19,"gradual",counts=(8,8))
        altered=data.copy();altered.loc[altered.trial_number>=65,"z"]+=100
        with threadpool_limits(limits=1):
            original=fit_individual(data,"state_space")
            repeated=fit_individual(altered,"state_space",start=original.theta)
        np.testing.assert_allclose(original.beta,repeated.beta,atol=1e-5,rtol=1e-4)
        mean,cov=individual_curve(original,0)
        self.assertEqual(len(mean),50)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(),-1e-8)
        self.assertLessEqual(original.data.trial_number.max(),64)


class RenderingTests(unittest.TestCase):
    def test_inline_svg_preserves_uncertainty_and_observation_markers(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from classical_conditioning.analysis.inference.panel_g_review_visuals import svg_text
        fig,axis=plt.subplots()
        axis.fill_between([0,1],[0,0],[1,1],alpha=.2)
        axis.scatter([0,1],[.5,.5])
        text=svg_text(fig,"test-fish")
        plt.close(fig)
        self.assertIn('xlink:href="#test-fish-',text)
        self.assertNotIn('ns4:href',text)
        self.assertIn('<svg ',text)


if __name__=="__main__":unittest.main()
