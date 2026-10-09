"""Independent readback of G's current figures against the selected frame table."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
manifest=json.loads((HERE/'data_manifest.json').read_text())
source=next(r for r in manifest['source_tables'] if r['panel']=='G')
path=Path(source['path'])
assert 'fgh_v1_trace20230310_08_20261009' in str(path)
assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
freeze=json.loads((REPO/'reviews/fgh_v1_trace20230310_08_20261009/frozen-version1/freeze.json').read_text())
assert freeze['fish']['G']=='20230310_08'
assert all('/20230310_08/' in r['path'].replace('\\','/') for r in freeze['replacement_inputs']['inputs'])
frames=pd.read_parquet(path)
export=pd.read_csv(HERE/'PanelG_direct_bins.csv').sort_values(['trial','bin_index'])
eligible=frames.loc[frames.eligible].copy()
eligible['bin_index']=np.floor((eligible.time_s+20)/.5).astype(int)
means=eligible.groupby(['trial','bin_index']).log_vigor.mean()
idx=pd.MultiIndex.from_frame(export[['trial','bin_index']])
recomputed=means.reindex(idx).to_numpy()
np.testing.assert_allclose(recomputed,export.mean_log_vigor,atol=1e-12,rtol=0,equal_nan=True)
recomputed_C=[];sample_D={}
for trial,p in frames.groupby('trial',sort=True):
    base=p.loc[p.time_s.ge(-15)&p.time_s.lt(0),'bout_median_log'].dropna()
    lo,mid,hi=base.quantile([.1,.5,.9]).to_numpy()
    denominator=max(mid-lo,hi-mid)
    sample_D[trial]=pd.Series(np.clip((p.bout_median_log-mid)/denominator,-1,1).to_numpy(),index=p.FrameID)
    b=export.loc[export.trial.eq(trial),'mean_log_vigor']
    lo,mid,hi=b.iloc[10:40].dropna().quantile([.1,.5,.9]).to_numpy()
    recomputed_C.extend(np.clip((b-mid)/((hi-lo)/2),-1,1))
np.testing.assert_allclose(recomputed_C,export.C,atol=1e-12,rtol=0,equal_nan=True)
runs=pd.read_csv(HERE/'PanelG_D_sample_runs.csv')
for row in runs.itertuples():
    values=sample_D[row.trial].loc[row.first_frame_id:row.last_frame_id]
    assert len(values)==row.sample_count
    np.testing.assert_allclose(values,row.D,atol=1e-12,rtol=0)
old=pd.read_parquet(REPO/'reviews/fgh_c_bout_vs_bins_20261008/outputs/PanelG_sample_data.parquet',columns=['FrameID'])
report={'confirmed_fish':'20230310_08','source_frame_table':str(path),'sha256':source['sha256'],
        'original_inputs':freeze['replacement_inputs']['inputs'],
        'Version2_all_7200_means_and_C_values_recomputed':True,
        'D_all_13813_runs_checked_against_frame_samples':True,
        'previous_fish_frame_table_is_different':len(old)!=len(frames) or not np.array_equal(old.FrameID,frames.FrameID)}
assert report['previous_fish_frame_table_is_different']
(HERE/'trace_source_confirmation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='original_inputs'},indent=2))
