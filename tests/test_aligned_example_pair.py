"""Shared identity/layout guards and the analytical boundary used by draft E."""
import copy
import unittest
import numpy as np
from classical_conditioning.figures.aligned_example_pair import scale_bins, validate_pair, TRIALS

class AlignedPairTests(unittest.TestCase):
    def pair(self):
        r={p:{} for p in ('D','E')}
        for t in TRIALS:
            mapping=dict(fish='same',trial=t,frame_ids_sha256='frames',time_values_sha256='times',time_coordinate='CS-relative',vigor_input_sha256='vigor')
            for p in ('D','E'):
                r[p][f'trial-{t}-trace']=dict(data_mapping=copy.deepcopy(mapping),layout=dict(x=1,y=t,width=4,height=2),row_label=dict(block='stage',trial=t))
            r['E'][f'trial-{t}-heatmap']=dict(data_mapping=copy.deepcopy(mapping))
        return r,dict(D_body_size=[7,5],E_body_size=[7,5])
    def test_mismatch_rejected_before_export(self):
        for property in ('fish','trial','frame_ids_sha256','time_values_sha256','time_coordinate','vigor_input_sha256'):
            r,c=self.pair();r['E'][f'trial-{TRIALS[0]}-heatmap']['data_mapping'][property]='different'
            with self.subTest(property=property),self.assertRaises(ValueError):validate_pair(r,c)
    def test_equal_outer_frames_do_not_hide_data_axis_mismatch(self):
        r,c=self.pair();r['E'][f'trial-{TRIALS[0]}-trace']['layout']['height']=3
        with self.assertRaises(ValueError):validate_pair(r,c)
    def test_bin_boundaries_missingness_and_scale_are_preserved(self):
        edges=np.linspace(-20,20,81);time=edges[:-1].copy();raw=np.exp(np.arange(80)*.02);eligible=np.ones(80,bool);eligible[45]=False
        z,med,counts,good,params=scale_bins(time,raw,eligible,edges)
        self.assertEqual(params['baseline_bins'],30)
        self.assertAlmostEqual(params['baseline_log'],.49)
        self.assertAlmostEqual(params['denominator_log'],.232)
        self.assertAlmostEqual(z[25],.7*.01/.232)
        self.assertTrue(np.isnan(z[45]));self.assertEqual(counts[45],0)
        self.assertGreater(abs(z[0]),1) # scientific values remain unclipped
    def test_insufficient_baseline_remains_missing(self):
        edges=np.linspace(-20,20,81);eligible=np.zeros(80,bool);eligible[10:19]=True
        z,_,_,_,_=scale_bins(edges[:-1],np.ones(80),eligible,edges)
        self.assertTrue(np.isnan(z).all())

if __name__=='__main__':unittest.main()
