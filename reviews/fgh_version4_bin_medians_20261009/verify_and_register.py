"""Read exported median-bin data back and register the additional fourth row."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((HERE/'data_manifest.json').read_text())
validation=json.loads((HERE/'validation.json').read_text())
assert digest(HERE/'data_manifest.json')==validation['data_manifest_sha256']
for r in manifest['data_files']+validation['outputs']:
    assert digest(Path(r['path']))==r['sha256']
stats=pd.read_csv(HERE/'baseline_statistics.csv')
report={'version':manifest['version'],'fish':manifest['fish'],'panels':[],'defined_trials':0}
for letter in ['F','G','H']:
    table=pd.read_csv(HERE/f'Panel{letter}_median_bins.csv')
    assert len(table)==7200
    assert np.array_equal(table.median_log_vigor.isna(),table.eligible_frame_count.eq(0))
    assert (table.eligible_frame_count+table.nan_sample_count==table.total_frame_count).all()
    np.testing.assert_allclose(table.delta_log_vigor,table.median_log_vigor-table.baseline_median_log_vigor,atol=1e-14,rtol=1e-14,equal_nan=True)
    residuals=[]
    for trial,p in table.groupby('trial'):
        assert p.bin_index.tolist()==list(range(80))
        base=p[p.start_s.ge(-15)&p.start_s.lt(0)]
        finite=base.median_log_vigor.dropna()
        reference=np.median(finite)
        np.testing.assert_allclose(p.baseline_median_log_vigor,reference,atol=1e-14,rtol=1e-14)
        record=stats[stats.panel.eq(letter)&stats.trial.eq(trial)].iloc[0]
        assert record.finite_baseline_bin_count==len(finite) and bool(record.defined)
        residual=abs(np.nanmedian(base.delta_log_vigor));assert residual<1e-14
        residuals.append(float(residual));report['defined_trials']+=1
    singleton=table.eligible_frame_count.eq(1)
    assert table.loc[singleton,'delta_log_vigor'].notna().all()
    report['panels'].append({'panel':letter,'trial_count':90,'single_sample_bins_retained':int(singleton.sum()),
        'all_nan_bins_missing':int(table.median_log_vigor.isna().sum()),
        'maximum_absolute_exported_baseline_median':max(residuals),
        'unscaled_subtraction_verified':True,'unclipped_values_outside_colour_range':int(table.delta_log_vigor.abs().gt(.25).sum())})
assert report['defined_trials']==270
report['control_trial16_defined']=bool(stats[stats.panel.eq('H')&stats.trial.eq(16)].defined.iloc[0])
(HERE/'numeric_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
config=REPO/'configs/paper-figures/figure1-fgh-full-bout-correction-20261009.json'
registration=json.loads(config.read_text())
registration['status']='Complete-bout sample versions and own-reference direct means retained; unscaled direct bin-median Version 4 added'
entry={'variant':manifest['version'],'validation':str(HERE/'validation.json'),'validation_sha256':digest(HERE/'validation.json')}
registration['variants']=[v for v in registration['variants'] if v['variant']!=manifest['version']]+[entry]
registration['version4_data_manifest']=str(HERE/'data_manifest.json')
registration['version4_data_manifest_sha256']=digest(HERE/'data_manifest.json')
registration['version4_numeric_verification']=str(HERE/'numeric_verification.json')
registration['version4_recipe']=manifest['formula']
registration['version4_colour_limits']=[-.25,.25]
registration['baseline']='C/D samples: unbinned timepoints carrying complete-bout medians; Version 2: finite direct baseline-bin means; Version 4: finite direct baseline-bin medians'
config.write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
