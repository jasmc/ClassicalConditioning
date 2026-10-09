"""Scale retained V7 physical values; fit each trial using baseline only."""
from pathlib import Path
import json,sys,hashlib,copy
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
SOURCE=HERE/'history/physical_bin_centred'
TOL=1e-12; MIN_BINS=10; A=.7
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def transform(x,p):
    x=np.asarray(x,dtype=float); h=p['centre_half_width_log']; c=p['centre_half_width_scaled']
    return np.where(x<-h,-c+p['negative_outer_slope']*(x+h),np.where(x>h,c+p['positive_outer_slope']*(x-h),p['centre_slope']*x))
def fit(b):
    x=np.sort(b[np.isfinite(b)]); p={'finite_baseline_bins':len(x),'scaled_trial_defined':False,'undefined_reason':''}
    if len(x)<MIN_BINS: return p|{'undefined_reason':'fewer than 10 finite baseline bins'}
    lo,med,hi=np.quantile(x,[.1,.5,.9],method='linear'); L=-lo; R=hi
    p.update(p10_log=lo,p50_log=med,p90_log=hi)
    assert abs(med)<TOL
    if min(L,R)<=TOL: return p|{'undefined_reason':'collapsed percentile side'}
    radius=max(abs(x[(len(x)-1)//2]),abs(x[len(x)//2])); h=max(radius,.05*min(L,R)); k=A/max(L,R); c=k*h
    if min(L,R)-h<=TOL: return p|{'undefined_reason':'percentile overlaps median centre'}
    p.update(centre_half_width_log=h,centre_half_width_scaled=c,centre_slope=k,negative_outer_slope=(A-c)/(L-h),positive_outer_slope=(A-c)/(R-h))
    z=transform(x,p); q=np.quantile(z,[.1,.5,.9])
    if not np.allclose(q,[-A,0,A],atol=TOL,rtol=0): return p|{'undefined_reason':'interpolated quantiles incompatible with centre'}
    p.update(scaled_trial_defined=True,scaled_baseline_p10=q[0],scaled_baseline_median=q[1],scaled_baseline_p90=q[2])
    assert np.all(np.diff(z)>=-TOL)
    assert all(p[s]>0 for s in ['centre_slope','negative_outer_slope','positive_outer_slope'])
    np.testing.assert_allclose(transform([lo,0,hi],p),[-A,0,A],atol=TOL,rtol=0)
    np.testing.assert_allclose(transform([-h,0,h],p),[-c,0,c],atol=TOL,rtol=0)
    # Left and right expressions agree with the central expression at joins.
    assert abs((-c+p['negative_outer_slope']*0)-k*(-h))<TOL
    assert abs((c+p['positive_outer_slope']*0)-k*h)<TOL
    return p
def build():
    old=json.loads((SOURCE/'data_manifest.json').read_text()); refs=[]; inputs=[]; panels=[]
    for panel in 'FGH':
        src=SOURCE/f'Panel{panel}_mean_bins.csv'
        expected=next(r['sha256'] for r in old['data_files'] if Path(r['path']).name==src.name)
        assert digest(src)==expected
        table=pd.read_csv(src); original=table.copy(deep=True)
        table['scaled_baseline_vigor']=np.nan; table['scaled_trial_defined']=False
        for trial,t in table.groupby('trial'):
            baseline=t.loc[t.start_s.ge(-15)&t.start_s.lt(0),'delta_log_vigor'].to_numpy(); p=fit(baseline)
            refs.append({'panel':panel,'fish':old['fish'][panel],'trial':int(trial),**p})
            if p['scaled_trial_defined']:
                z=transform(t.delta_log_vigor.to_numpy(),p)
                assert np.array_equal(np.isnan(z),t.delta_log_vigor.isna())
                assert np.all(np.sign(z[np.isfinite(z)])==np.sign(t.delta_log_vigor.to_numpy()[np.isfinite(z)]))
                table.loc[t.index,'scaled_baseline_vigor']=z; table.loc[t.index,'scaled_trial_defined']=True
        pd.testing.assert_frame_equal(table[original.columns],original)
        out=HERE/src.name; table.to_csv(out,index=False); readback=pd.read_csv(out)
        np.testing.assert_allclose(readback.delta_log_vigor,original.delta_log_vigor,atol=TOL,rtol=0,equal_nan=True)
        for trial,t in readback[readback.scaled_trial_defined].groupby('trial'):
            b=t.loc[t.start_s.ge(-15)&t.start_s.lt(0),'scaled_baseline_vigor'].dropna()
            np.testing.assert_allclose(np.quantile(b,[.1,.5,.9]),[-A,0,A],atol=TOL,rtol=0)
        inputs.append({'path':str(src),'sha256':expected})
        panels.append({'panel':panel,'fish':old['fish'][panel],'defined_trials':int(table.groupby('trial').scaled_trial_defined.first().sum()),'finite_physical_bins':int(table.delta_log_vigor.notna().sum()),'finite_scaled_bins':int(table.scaled_baseline_vigor.notna().sum()),'total_bins':len(table)})
    ref=pd.DataFrame(refs); ref.to_csv(HERE/'scaling_parameters.csv',index=False)
    (HERE/'baseline_statistics.csv').write_bytes((SOURCE/'baseline_statistics.csv').read_bytes())
    scaling={'baseline_interval_s':[-15,0],'minimum_finite_baseline_bins':MIN_BINS,'quantile_method':'numpy linear','anchors':{'P10':-A,'P50':0,'P90':A},
        'centre_rule':'h=max(abs(two middle order statistics), 0.05*min(-P10,P90)); k=0.7/max(-P10,P90); c=k*h',
        'formula':'x<-h: -c+s_negative*(x+h); abs(x)<=h: k*x; x>h: c+s_positive*(x-h)',
        'outer_slopes':'s_negative=(0.7-c)/(-P10-h); s_positive=(0.7-c)/(P90-h)',
        'extrapolation':'continue outer linear slopes; scaled exports not clipped; colours saturate beyond +/-1',
        'undefined':'fewer than 10 finite baseline bins, collapsed/overlapping sides, or failure to preserve interpolated P10/P50/P90',
        'independence_caveat':'bins can share a bout; count is a screening rule, not independent sample size'}
    manifest=copy.deepcopy(old)
    manifest.update(revision='separate P10/P90 scaling with a median-preserving centre',physical_value_column='delta_log_vigor',display_value_column='scaled_baseline_vigor',physical_colour_limits_previous_revision=[-.25,.25],colour_limits=[-1,1],colourbar_label='Scaled deviation from baseline',data_scaling=True,data_clipping=False,scaling=scaling,
        undefined_rule='scaled trial entirely missing for sparse or degenerate baseline; physical values retained',
        source_physical_data_manifest=str(SOURCE/'data_manifest.json'),source_physical_data_manifest_sha256=digest(SOURCE/'data_manifest.json'),source_physical_files=inputs,panels=panels,
        data_files=[{'path':str(p),'sha256':digest(p)} for p in HERE.glob('*.csv')])
    (HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    defined=ref[ref.scaled_trial_defined]
    report={'defined_scaled_trials':len(defined),'undefined_scaled_trials':len(ref)-len(defined),'undefined_trials':ref.loc[~ref.scaled_trial_defined,['panel','trial','finite_baseline_bins','undefined_reason']].to_dict('records'),
        'minimum_finite_baseline_bins':MIN_BINS,'baseline_anchors_scaled':[-A,0,A],'max_absolute_baseline_median':float(defined.scaled_baseline_median.abs().max()),'max_absolute_anchor_error':float(max((defined.scaled_baseline_p10+A).abs().max(),(defined.scaled_baseline_p90-A).abs().max())),
        'physical_values_retained_unchanged':True,'monotonicity_sign_continuity_and_csv_readback_verified':True,'baseline_only_fit':True,'unclipped_scaled_exports':True,'panels':panels}
    (HERE/'numeric_verification.json').write_text(json.dumps(report,indent=2))
    code=(REPO/'reviews/fgh_colour_binning_candidates_20261009/render_candidates.py').read_text()
    code=code.replace("('Version4_','Version5_','Version6_')","('Version4_','Version5_','Version6_','Version7_')")
    code=code.replace('norm=Normalize(-.25,.25,clip=True) if is_version4 else Normalize(-1,1,clip=True)',"norm=Normalize(-1,1,clip=True) if kind.startswith('Version7_') else (Normalize(-.25,.25,clip=True) if is_version4 else Normalize(-1,1,clip=True))")
    code=code.replace('ticks=[-.25,0,.25] if is_version4 else [-1,-.5,0,.5,1]',"ticks=[-1,-.7,0,.7,1] if kind.startswith('Version7_') else ([-.25,0,.25] if is_version4 else [-1,-.5,0,.5,1])\n    if kind.startswith('Version7_'): label='Scaled deviation from baseline'")
    (HERE/'render_version7.py').write_text(code); sys.path.insert(0,str(HERE))
    import render_version7
    render_version7.render('Version7_UnpooledRecipe',HERE,'scaled_baseline_vigor',True)
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': build()
