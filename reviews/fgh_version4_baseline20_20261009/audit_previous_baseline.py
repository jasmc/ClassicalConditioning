"""Audit the existing fourth version without changing its values or figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
SOURCE=REPO/'reviews/fgh_version4_bin_medians_20261009'
TOL=1e-12
rows=[]
for panel in 'FGH':
    table=pd.read_csv(SOURCE/f'Panel{panel}_median_bins.csv')
    for trial,p in table.groupby('trial'):
        baseline=p[p.start_s.ge(-15)&p.start_s.lt(0)].delta_log_vigor.dropna().to_numpy()
        whole=p.delta_log_vigor.dropna().to_numpy()
        assert abs(np.median(baseline))<TOL
        above=int((baseline>TOL).sum());below=int((baseline<-TOL).sum());zero=int((abs(baseline)<=TOL).sum())
        assert above<=len(baseline)/2 and below<=len(baseline)/2
        rows.append({'panel':panel,'trial':int(trial),'finite_baseline_bins':len(baseline),
            'baseline_above_zero':above,'baseline_below_zero':below,'baseline_tied_at_zero':zero,
            'absolute_above_below_difference':abs(above-below),'baseline_median':float(np.median(baseline)),
            'finite_display_bins':len(whole),'whole_window_above_zero':int((whole>TOL).sum()),
            'whole_window_above_fraction':float(np.mean(whole>TOL)),
            'baseline_above_fraction':above/len(baseline)})
result=pd.DataFrame(rows)
result.to_csv(HERE/'previous_v4_per_trial_balance.csv',index=False)
report={'audit':'previous Version 4; [-15,0) baseline; NaNs ignored','tolerance_log_units':TOL,'trials':len(result),
    'median_zero_all_trials':True,'neither_strict_sign_exceeds_half_in_any_trial':True,
    'equal_above_below_counts':int(result.absolute_above_below_difference.eq(0).sum()),
    'above_below_differ_by_at_most_one':int(result.absolute_above_below_difference.le(1).sum()),
    'maximum_above_below_count_difference':int(result.absolute_above_below_difference.max()),
    'unequal_counts_explained_by_ties':bool((result.absolute_above_below_difference<=result.baseline_tied_at_zero).all()),
    'maximum_absolute_baseline_median':float(result.baseline_median.abs().max()),
    'largest_imbalance_examples':result.nlargest(6,'absolute_above_below_difference').to_dict('records')}
(HERE/'previous_v4_balance_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
