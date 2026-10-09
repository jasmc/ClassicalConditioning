"""Compare the direct-bin display with its three possible reference populations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
stats=pd.read_csv(HERE/'baseline_statistics.csv')
rows=[]
for panel,fish in [('F','20221115_07'),('G','20230310_08'),('H','20221115_09')]:
    frames=pd.read_parquet(HERE/f'Panel{panel}_complete_bout_sample_data.parquet',columns=['trial','time_s','eligible','log_vigor'])
    bins=pd.read_csv(HERE/f'Panel{panel}_direct_bins.csv')
    for trial,p in frames.groupby('trial',sort=True):
        base=p.loc[p.eligible&p.time_s.ge(-15)&p.time_s.lt(0),'log_vigor']
        b=bins[bins.trial.eq(trial)&bins.bin_index.between(10,39)]
        reference=stats[(stats.panel==panel)&(stats.trial==trial)].iloc[0]
        raw_lo,raw_m,raw_hi=np.quantile(base,[.1,.5,.9])
        means=b.mean_log_vigor.dropna().to_numpy()
        bin_lo,bin_m,bin_hi=np.quantile(means,[.1,.5,.9])
        raw_centered=np.clip((means-raw_m)/((raw_hi-raw_lo)/2),-1,1) if raw_hi>raw_lo else np.full(len(means),np.nan)
        bin_centered=np.clip((means-bin_m)/((bin_hi-bin_lo)/2),-1,1) if bin_hi>bin_lo else np.full(len(means),np.nan)
        current=b.C.dropna().to_numpy()
        rows.append({'panel':panel,'fish':fish,'trial':int(trial),
            'bout_reference_median':reference.p50,'direct_timepoint_median':float(raw_m),'direct_bin_median':float(bin_m),
            'direct_bin_minus_bout_median':float(bin_m-reference.p50),
            'current_baseline_cell_median':float(np.median(current)) if len(current) else None,
            'current_baseline_cells_clipped_minus_one_fraction':float(np.mean(current<=-1)) if len(current) else None,
            'direct_timepoint_reference_baseline_cell_median':float(np.nanmedian(raw_centered)),
            'direct_bin_reference_baseline_cell_median':float(np.nanmedian(bin_centered))})
    del frames
table=pd.DataFrame(rows);table.to_csv(HERE/'direct_bin_offset_audit.csv',index=False)
summary=[]
for panel,p in table.groupby('panel',sort=True):
    summary.append({'panel':panel,'fish':p.fish.iloc[0],
        'median_of_current_trial_baseline_cell_medians':float(p.current_baseline_cell_median.median()),
        'median_baseline_cells_clipped_minus_one_fraction':float(p.current_baseline_cells_clipped_minus_one_fraction.median()),
        'median_direct_bin_minus_bout_median_log':float(p.direct_bin_minus_bout_median.median()),
        'median_baseline_cell_median_with_direct_timepoint_reference':float(p.direct_timepoint_reference_baseline_cell_median.median()),
        'max_absolute_baseline_cell_median_with_direct_bin_reference':float(p.direct_bin_reference_baseline_cell_median.abs().max())})
(HERE/'direct_bin_offset_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
