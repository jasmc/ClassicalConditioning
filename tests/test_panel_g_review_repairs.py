"""Regression checks for audited numerical failure, publication and cohort rules."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from classical_conditioning.analysis.inference.panel_g_review import ReviewConfig
from classical_conditioning.analysis.inference.panel_g_missingness import scheduled_individual_timing
from classical_conditioning.analysis.inference.panel_g_review_extensions import run_optimizer_stability,simulate
from classical_conditioning.analysis.inference.panel_g_run import make_manifest,verify_stage,stage_transaction,run_lock
from classical_conditioning.analysis.inference.panel_g_legacy import bout_in_window,coverage_rules
from classical_conditioning.analysis.inference.panel_g_marginal import fish_matrix,marginal_band,summarize_marginal_calibration
from classical_conditioning.analysis.inference.panel_g_review import fit,design_for,estimate,resample,bootstrap_worker,fit_with_retry


class RepairTests(unittest.TestCase):
    def test_calibration_counts_accepted_fits_with_unlocalized_timing(self):
        with tempfile.TemporaryDirectory() as name:
            out=Path(name);rows=[]
            for status in ['bounded','not_localized','band_failed']:
                rows.append({'scenario':'null','status':status,'simultaneous_trial_coverage':True,
                    'simultaneous_phase_coverage':True,'false_supported_run':False,'onset_interval_contains_truth':True})
            pd.DataFrame(rows).to_csv(out/'unsmoothed-calibration.csv',index=False)
            summarize_marginal_calibration(out,3)
            f=pd.read_csv(out/'unsmoothed-calibration-summary.csv',keep_default_na=False)
            self.assertTrue(f.scenario.eq('null').all())
            self.assertTrue(f.independent_experiments.eq(2).all())
            self.assertTrue(f.failures_withheld.eq(1).all())
            self.assertTrue(f.requested.eq(3).all())
    def test_integer_cluster_weights_match_duplicated_fish_with_unique_ids(self):
        from threadpoolctl import threadpool_limits
        data,_=simulate(57,'tracking_gaps',counts=(12,13))
        with threadpool_limits(1):
            for target in ['difference','adjusted_response']:
                cached=bootstrap_worker((data,target,502,float(data.baseline_logmedian.mean())))
                duplicated=resample(data,np.random.default_rng(502))
                reference,diag=fit_with_retry(duplicated,target,group='resampled_key',extended=False)
                self.assertTrue(cached['success']);self.assertIsNotNone(reference,str(diag))
                _,matrix,_=design_for(reference,target,data.baseline_logmedian.mean())
                values,se=estimate(reference,matrix)
                np.testing.assert_allclose(cached['estimate'],values,atol=2e-6,rtol=2e-4)
                np.testing.assert_allclose(cached['se'],se,atol=2e-6,rtol=2e-4)
    def test_sufficient_statistics_match_reference_likelihood_and_contrasts(self):
        from threadpoolctl import threadpool_limits
        data,_=simulate(82,'tracking_gaps',counts=(12,13))
        with threadpool_limits(1):
            for target in ['difference','adjusted_response']:
                accelerated,da=fit(data,target)
                reference,dr=fit(data,target,backend='statsmodels',optimizer='powell',extended=False)
                self.assertIsNotNone(accelerated,str(da));self.assertIsNotNone(reference,str(dr))
                _,matrix,_=design_for(reference,target,data.baseline_logmedian.mean())
                a,se_a=estimate(accelerated,matrix);b,se_b=estimate(reference,matrix)
                self.assertLess(abs(accelerated.llf-reference.llf),.002)
                np.testing.assert_allclose(a,b,atol=2e-4,rtol=1e-3)
                np.testing.assert_allclose(se_a,se_b,atol=2e-4,rtol=1e-2)

    def test_failed_full_target_does_not_require_missing_bootstrap_files(self):
        with tempfile.TemporaryDirectory() as name:
            out=Path(name);pd.DataFrame({'fish_key':[]}).to_parquet(out/'model-input.parquet')
            for target in ['difference','adjusted_response']:
                (out/target).mkdir();(out/target/'result.json').write_text('{"status":"failed"}')
            run_optimizer_stability(out,ReviewConfig(bootstrap=19),workers=1)
            for target in ['difference','adjusted_response']:
                self.assertTrue(json.loads((out/target/'optimizer-stability.json').read_text())['inference_withheld'])

    def test_same_information_rule_does_not_reject_a_missing_onset_trial(self):
        tr=np.arange(15,65);m=np.zeros(50);m[5:]=.3
        seen=np.ones(50,bool);seen[5]=False
        result=scheduled_individual_timing(tr,m,m-.01,m+.01,seen,.1,pre_count=3)
        self.assertEqual(result['estimate_trial'],20)
        self.assertEqual(result['first_observed_supported_trial'],21)
        self.assertEqual(scheduled_individual_timing(tr,m,m-.01,m+.01,seen,.1,pre_count=2)['status'],
                         'insufficient_information')

    def test_manifest_detects_content_and_inventory_changes(self):
        with tempfile.TemporaryDirectory() as name:
            out=Path(name);(out/'complete.json').write_text('{}');(out/'values.csv').write_text('x\n1\n')
            digest=make_manifest(out);verify_stage(out,digest)
            (out/'values.csv').write_text('x\n2\n')
            with self.assertRaisesRegex(ValueError,'artifact changed'):verify_stage(out,digest)
            (out/'values.csv').write_text('x\n1\n');(out/'unbound.csv').write_text('x\n3\n')
            with self.assertRaisesRegex(ValueError,'inventory changed'):verify_stage(out,digest)

    def test_failed_stage_preserves_previous_pointer_and_values(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'source.py';source.write_text('x=1\n')
            def initial(d):(d/'values.csv').write_text('x\n1\n')
            state=stage_transaction(root,{},'prepare',[source],initial,lambda:None)
            previous=root/state['latest']['directory'];pointer=(root/'analysis-state.json').read_bytes()
            def failing(d):
                (d/'values.csv').write_text('x\n99\n');raise ValueError('simulated failure')
            with self.assertRaisesRegex(ValueError,'simulated failure'):
                stage_transaction(root,state,'population',[source],failing,lambda:None)
            self.assertEqual((root/'analysis-state.json').read_bytes(),pointer)
            self.assertEqual((previous/'values.csv').read_text(),'x\n1\n')
            verify_stage(previous,state['latest']['manifest_sha256'])

    def test_source_change_cannot_publish_a_stage(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'source.py';source.write_text('x=1\n')
            def changing(d):source.write_text('x=2\n')
            with self.assertRaisesRegex(ValueError,'Source changed'):
                stage_transaction(root,{},'prepare',[source],changing,lambda:None)
            self.assertFalse((root/'analysis-state.json').exists())

    def test_second_writer_cannot_take_run_lock(self):
        with tempfile.TemporaryDirectory() as name:
            with run_lock(Path(name)):
                with self.assertRaisesRegex(ValueError,'Another process'):
                    with run_lock(Path(name)):pass

    def test_historical_windows_are_inclusive(self):
        self.assertTrue(bout_in_window(np.array([5000]),0,0,5))
        self.assertFalse(bout_in_window(np.array([5001]),0,0,5))
        self.assertTrue(bout_in_window(np.array([-15000]),0,-15,0))

    def test_any_block_rule_is_not_all_block_completeness(self):
        status=coverage_rules(range(5,11),[3,3,3],[3,2,3])
        self.assertTrue(status['any_block_6']);self.assertFalse(status['all_blocks_6'])
        self.assertTrue(status['baseline_3']);self.assertFalse(status['cr_3'])

    def test_unsmoothed_comparator_does_not_impute_unavailable_vigor(self):
        data,_=simulate(28,'tracking_gaps',counts=(8,8));index,x=fish_matrix(data)
        self.assertTrue(np.isnan(x[:,:90]).any())
        values,lo,hi,status,_=marginal_band(data,44,99)
        self.assertEqual(len(values),93)
        self.assertEqual(status['status'],'exploratory')
        self.assertTrue(np.all(lo<=values));self.assertTrue(np.all(values<=hi))


if __name__=='__main__':unittest.main()
