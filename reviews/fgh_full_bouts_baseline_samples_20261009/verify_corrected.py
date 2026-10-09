"""Readback of complete-bout summaries and shared timepoint baseline."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/'data_manifest.json').read_text())
for item in manifest['data_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
stats=pd.read_csv(HERE/'baseline_statistics.csv');assert len(stats)==270
checks=[]
for panel in 'FGH':
    frame=pd.read_parquet(HERE/f'Panel{panel}_complete_bout_sample_data.parquet')
    support=pd.read_parquet(HERE/f'Panel{panel}_complete_bout_support.parquet')
    bouts=pd.read_csv(HERE/f'Panel{panel}_complete_bouts.csv').set_index('bout_id')
    summaries=support.groupby('bout_id').log_vigor.agg(['median','count'])
    np.testing.assert_allclose(summaries['median'],bouts.loc[summaries.index,'median_log_vigor'],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(summaries['count'],bouts.loc[summaries.index,'eligible_sample_count'])
    eligible=frame[frame.eligible]
    mapped=summaries['median'].reindex(eligible.bout_id).to_numpy()
    np.testing.assert_allclose(mapped,eligible.bout_median_log,atol=1e-12,rtol=0)
    cells=pd.read_csv(HERE/f'Panel{panel}_direct_bins.csv')
    zero_residuals=[];cell_medians=[]
    for trial,p in frame.groupby('trial',sort=True):
        base=p.eligible&p.time_s.ge(-15)&p.time_s.lt(0)
        values=p.loc[base,'bout_median_log']
        row=stats[(stats.panel==panel)&(stats.trial==trial)].iloc[0]
        assert len(values)==row.baseline_timepoint_count
        lo,m,hi=np.quantile(values,[.1,.5,.9],method='linear')
        np.testing.assert_allclose([lo,m,hi],[row.p10,row.p50,row.p90],atol=1e-12,rtol=0)
        for mode,width in [('C',(hi-lo)/2),('D',max(m-lo,hi-m))]:
            target=np.clip((p.bout_median_log-m)/width,-1,1) if width>0 else np.full(len(p),np.nan)
            np.testing.assert_allclose(target,p[mode+'_sample'],atol=1e-12,rtol=0,equal_nan=True)
            if width>0:zero_residuals.append(float(abs(p.loc[base,mode+'_sample'].median())))
        bins=cells[cells.trial==trial].sort_values('bin_index')
        good=p[p.eligible].copy();good['bin_index']=np.floor((good.time_s+20)/.5).astype(int)
        means=good.groupby('bin_index').log_vigor.mean().reindex(range(80))
        np.testing.assert_allclose(means,bins.mean_log_vigor,atol=1e-12,rtol=0,equal_nan=True)
        target=np.clip((means-m)/((hi-lo)/2),-1,1) if hi>lo else np.full(80,np.nan)
        np.testing.assert_allclose(target,bins.C,atol=1e-12,rtol=0,equal_nan=True)
        median=bins.iloc[10:40].C.median()
        if np.isfinite(median):cell_medians.append(float(median))
    checks.append({'panel':panel,'full_bout_support_and_medians_verified':True,
        'shared_baseline_timepoint_quantiles_verified':True,'Version2_direct_means_and_shared_reference_scaling_verified':True,
        'maximum_C_D_sample_baseline_median_residual':max(zero_residuals),
        'Version2_baseline_cell_median_range':[min(cell_medians),max(cell_medians)]})
    del frame,support
g=next(r for r in manifest['panels'] if r['panel']=='G')
assert g['fish']=='20230310_08' and all('20230310_08' in r['path'] for r in g['inputs'])
report={'fish':manifest['fish'],'checks':checks,
        'undefined_trials':stats[stats.C_scale.le(0)|stats.C_scale.isna()][['panel','trial']].to_dict('records'),
        'baseline_zero':'C/D sample baseline medians are zero; Version2 shares that reference but its displayed baseline-bin median need not be zero'}
(HERE/'numeric_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
