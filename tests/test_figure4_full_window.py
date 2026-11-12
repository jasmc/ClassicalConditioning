"""Scientific boundary checks for the author-requested full-window refit."""
import unittest
import numpy as np
import pandas as pd
from classical_conditioning.analysis.review.figure4_assay_profiles import ProfileSpec, model_data, GROUPS, fit
from classical_conditioning.analysis.review.figure4_profiles import prediction_design


class FullWindowTests(unittest.TestCase):
    def test_full_grid_preserves_missingness_and_both_tails(self):
        times=np.arange(-20,20,.5)+.25
        values=np.ones(80);values[10]=np.nan
        frame=pd.DataFrame(dict(fish_id=['a']*80,display_group=[GROUPS[0]]*80,time_s=times,signed=values))
        data=model_data(frame,ProfileSpec(-20,20))
        self.assertEqual(len(data),79)
        self.assertEqual(data.time_s.min(),-19.75)
        self.assertEqual(data.time_s.max(),19.75)
        self.assertNotIn(times[10],data.time_s.tolist())
        np.testing.assert_array_equal(data.trial_scaled,data.time_s/20)

    def test_full_window_design_reproduces_known_group_cubic_means(self):
        rng=np.random.default_rng(29);rows=[]
        times=np.arange(-20,20,.5)+.25
        # Balanced fish offsets/noise let a cubic fixed mean remain recoverable
        # across both tails; the old cropped design cannot predict these times.
        for g,group in enumerate(GROUPS):
            for fish in range(8):
                offset=(fish-3.5)*.015
                for t in times:
                    mean=.02*g + .003*t - .00001*t**3
                    rows.append(dict(fish_id=f'{g}-{fish}',condition_id='control' if g==0 else 'trace',display_group=group,time_s=t,signed=mean+offset+rng.normal(0,.015)))
        frame=pd.DataFrame(rows)
        target=frame.display_group.map({g:.02*k for k,g in enumerate(GROUPS)}).to_numpy()+.003*frame.time_s-.00001*frame.time_s**3
        frame['noise']=frame.signed-target
        frame['signed']-=frame.groupby(['display_group','time_s']).noise.transform('mean')
        spec=ProfileSpec(-20,20);data=model_data(frame,spec)
        result,history=fit(data,random='1',spec=spec)
        self.assertIsNotNone(result,history)
        _,matrix=prediction_design(result,times)
        predicted=(matrix@result.fe_params.to_numpy()).reshape(3,-1)
        expected=np.asarray([.02*g+.003*times-.00001*times**3 for g in range(3)])
        np.testing.assert_allclose(predicted,expected,atol=1e-9)


if __name__=='__main__': unittest.main()
