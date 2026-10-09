"""V8: V7 scaling with P10/P90 at colour endpoints; colours saturate."""
from pathlib import Path
import json,sys,hashlib,copy
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
SOURCE=REPO/'reviews/fgh_version7_unpooled_recipe_20261009'
TOL=1e-12
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def build():
    source=json.loads((SOURCE/'data_manifest.json').read_text())
    hashes={Path(r['path']).name:r['sha256'] for r in source['data_files']}
    protected={str(p):digest(p) for p in SOURCE.iterdir() if p.is_file()}
    parameters=pd.read_csv(SOURCE/'scaling_parameters.csv')
    assert digest(SOURCE/'scaling_parameters.csv')==hashes['scaling_parameters.csv']
    numeric=['centre_half_width_scaled','centre_slope','negative_outer_slope','positive_outer_slope',
             'scaled_baseline_p10','scaled_baseline_median','scaled_baseline_p90']
    parameters[numeric]=parameters[numeric]/.7
    parameters.to_csv(HERE/'scaling_parameters.csv',index=False)
    stats=[];inputs=[];panels=[]
    for panel in 'FGH':
        path=SOURCE/f'Panel{panel}_mean_bins.csv'; assert digest(path)==hashes[path.name]
        table=pd.read_csv(path); old=table.copy(deep=True)
        table['version7_scaled_baseline_vigor']=table.scaled_baseline_vigor
        table['scaled_baseline_vigor']=table.scaled_baseline_vigor/.7
        pd.testing.assert_frame_equal(table[[c for c in old.columns if c!='scaled_baseline_vigor']],old.drop(columns='scaled_baseline_vigor'))
        for trial,t in table.groupby('trial'):
            p=parameters[(parameters.panel==panel)&parameters.trial.eq(trial)].iloc[0]
            if p.scaled_trial_defined:
                b=t.loc[t.start_s.ge(-15)&t.start_s.lt(0),'scaled_baseline_vigor'].dropna()
                q=np.quantile(b,[.1,.5,.9]); np.testing.assert_allclose(q,[-1,0,1],atol=TOL,rtol=0)
                # Independently reconstruct the piecewise map from physical values.
                x=t.delta_log_vigor.to_numpy();h=p.centre_half_width_log;c=p.centre_half_width_scaled
                expected=np.where(x<-h,-c+p.negative_outer_slope*(x+h),np.where(x>h,c+p.positive_outer_slope*(x-h),p.centre_slope*x))
                np.testing.assert_allclose(t.scaled_baseline_vigor,expected,atol=TOL,rtol=TOL,equal_nan=True)
                np.testing.assert_allclose(np.clip(expected[np.isfinite(x)&(x<=p.p10_log)],-1,1),-1,atol=TOL,rtol=0)
                np.testing.assert_allclose(np.clip(expected[np.isfinite(x)&(x>=p.p90_log)],-1,1),1,atol=TOL,rtol=0)
                stats.append({'panel':panel,'trial':int(trial),'p10':q[0],'median':q[1],'p90':q[2]})
            else: assert t.scaled_baseline_vigor.isna().all()
        table.to_csv(HERE/path.name,index=False)
        readback=pd.read_csv(HERE/path.name)
        np.testing.assert_allclose(readback.delta_log_vigor,old.delta_log_vigor,atol=TOL,rtol=0,equal_nan=True)
        np.testing.assert_allclose(readback.scaled_baseline_vigor,old.scaled_baseline_vigor/.7,atol=TOL,rtol=TOL,equal_nan=True)
        panels.append({'panel':panel,'defined_trials':int(table.groupby('trial').scaled_trial_defined.first().sum()),
                       'finite_scaled_bins':int(table.scaled_baseline_vigor.notna().sum()),'total_bins':len(table),
                       'cells_at_or_beyond_colour_endpoints':int(table.scaled_baseline_vigor.abs().ge(1-TOL).sum())})
        inputs.append({'path':str(path),'sha256':digest(path)})
    (HERE/'baseline_statistics.csv').write_bytes((SOURCE/'baseline_statistics.csv').read_bytes())
    manifest=copy.deepcopy(source)
    manifest['version']='Version8_PercentileEndpoints';manifest['revision']='P10/P90 at -1/+1 with endpoint colour saturation'
    manifest['scaling']['anchors']={'P10':-1,'P50':0,'P90':1}
    manifest['scaling']['centre_rule']=manifest['scaling']['centre_rule'].replace('0.7','1.0')
    manifest['scaling']['outer_slopes']=manifest['scaling']['outer_slopes'].replace('0.7','1.0')
    manifest['scaling']['extrapolation']='outer linear slopes continue numerically; colour norm clips at -1/+1, so values beyond P10/P90 have saturated endpoint colours'
    manifest.update(source_version7_manifest=str(SOURCE/'data_manifest.json'),source_version7_manifest_sha256=digest(SOURCE/'data_manifest.json'),
                    source_version7_numeric_files=inputs,panels=panels,data_files=[{'path':str(p),'sha256':digest(p)} for p in HERE.glob('*.csv')])
    (HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    audit=pd.DataFrame(stats)
    report={'defined_scaled_trials':len(audit),'undefined_scaled_trials':270-len(audit),'minimum_finite_baseline_bins':10,
            'baseline_anchors_scaled':[-1,0,1],'max_absolute_baseline_median':float(audit['median'].abs().max()),
            'max_absolute_anchor_error':float(max((audit.p10+1).abs().max(),(audit.p90-1).abs().max())),
            'physical_values_and_defined_trial_mask_unchanged_from_version7':True,
            'independent_piecewise_map_and_endpoint_saturation_verified':True,'csv_readback_verified':True,
            'version8_equals_version7_divided_by_0_7':True,'numeric_scaled_values_unclipped':True,'panels':panels}
    (HERE/'numeric_verification.json').write_text(json.dumps(report,indent=2))
    code=(SOURCE/'render_version7.py').read_text().replace('Version7_','Version8_').replace('[-1,-.7,0,.7,1]','[-1,0,1]')
    (HERE/'render_version8.py').write_text(code); sys.path.insert(0,str(HERE))
    import render_version8
    render_version8.render('Version8_PercentileEndpoints',HERE,'scaled_baseline_vigor',True)
    assert all(digest(Path(p))==h for p,h in protected.items())
    (HERE/'version7_preservation.json').write_text(json.dumps(protected,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': build()
