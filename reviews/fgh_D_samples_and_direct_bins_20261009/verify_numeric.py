from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
current=REPO/'reviews/fgh_v1_trace20230310_08_20261009/frozen-version1'
prior=REPO/'reviews/fgh_c_bout_vs_bins_20261008/outputs'
checks=[]
for panel in 'FGH':
    c=pd.read_csv(current/f'Panel{panel}_display_sample_runs.csv')
    d=pd.read_csv(HERE/f'Panel{panel}_D_sample_runs.csv')
    pd.testing.assert_frame_equal(c.drop(columns='C'),d.drop(columns='D'),check_exact=True)
    assert np.all(np.abs(d.D)<=np.abs(c.C)+1e-12)
    checks.append({'panel':panel,'D_sample_support_identical_to_C':True,
                   'D_absolute_values_no_larger_than_C':True})
    if panel in 'FH':
        old=pd.read_csv(prior/f'Panel{panel}_direct_halfsecond_bins.csv')
        new=pd.read_csv(HERE/f'Panel{panel}_direct_bins.csv')
        np.testing.assert_allclose(new.mean_log_vigor,old.direct_log_bin,atol=1e-12,rtol=0,equal_nan=True)
        np.testing.assert_allclose(new.C,old.C_bin,atol=1e-12,rtol=0,equal_nan=True)
        checks[-1]['Version2_matches_original_direct_bin_values']=True
statistics=pd.read_csv(HERE/'baseline_statistics.csv')
assert len(statistics)==810
report={'checks':checks,'undefined_trials':statistics[~statistics.defined][['panel','trial','variant']].to_dict('records')}
(HERE/'numeric_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
