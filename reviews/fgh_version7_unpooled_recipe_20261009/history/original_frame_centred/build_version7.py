"""Additional V7: raw-frame baseline, window bout medians, 0.5 s means."""
from pathlib import Path
import sys, json, gc, hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
SOURCE=REPO/'reviews/fgh_full_bouts_baseline_samples_20261009'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def build():
    previous=json.loads((SOURCE/'data_manifest.json').read_text())
    refs=[];panels=[];sources=[]
    edges=np.arange(-20,20.5,.5)
    for panel in 'FGH':
        path=SOURCE/f'Panel{panel}_complete_bout_sample_data.parquet'
        expected=next(x['sha256'] for x in previous['data_files'] if Path(x['path'])==path)
        assert digest(path)==expected
        frames=pd.read_parquet(path,columns=['trial','time_s','eligible','bout_id','log_vigor'])
        np.testing.assert_array_equal(frames.log_vigor.notna(),frames.eligible)
        tables=[]
        for trial,p in frames.groupby('trial',sort=True):
            assert p.time_s.ge(-20).all() and p.time_s.lt(20).all()
            good=p.eligible & p.bout_id.gt(0) & np.isfinite(p.log_vigor)
            assert good.equals(p.eligible)
            baseline_values=p.loc[good & p.time_s.ge(-15) & p.time_s.lt(0),'log_vigor']
            baseline=float(baseline_values.median()) if len(baseline_values) else np.nan
            centred=p.log_vigor-baseline
            # Group only eligible frames inside this trial's extracted window.
            repeated=centred.where(good).groupby(p.bout_id).transform('median').where(good)
            assert repeated[~good].isna().all()
            medians=p.loc[good].groupby('bout_id').log_vigor.median()-baseline
            np.testing.assert_allclose(repeated[good],p.loc[good,'bout_id'].map(medians),rtol=0,atol=1e-12,equal_nan=True)
            idx=np.searchsorted(edges,p.time_s.to_numpy(),side='right')-1
            vals=repeated.to_numpy();finite=np.isfinite(vals)
            counts=np.bincount(idx[finite],minlength=80)
            sums=np.bincount(idx[finite],weights=vals[finite],minlength=80)
            means=np.divide(sums,counts,out=np.full(80,np.nan),where=counts>0)
            independent=pd.DataFrame({'bin':idx,'value':vals}).groupby('bin').value.mean().reindex(range(80)).to_numpy()
            np.testing.assert_allclose(means,independent,atol=1e-12,rtol=0,equal_nan=True)
            residual=float(np.median(baseline_values-baseline)) if len(baseline_values) else np.nan
            if len(baseline_values): assert abs(residual)<1e-12
            displayed_base=means[(edges[:-1]>=-15)&(edges[:-1]<0)]
            refs.append({'panel':panel,'trial':int(trial),'baseline_median_log_vigor':baseline,
                         'eligible_baseline_frames':len(baseline_values),'defined':bool(np.isfinite(baseline)),
                         'centred_raw_frame_baseline_median':residual,
                         'displayed_baseline_bin_median':float(np.nanmedian(displayed_base)) if np.isfinite(displayed_base).any() else np.nan})
            tables.append(pd.DataFrame({'trial':trial,'bin_index':range(80),'start_s':edges[:-1],'end_s':edges[1:],
                'finite_repeated_bout_frame_count':counts,'baseline_median_log_vigor':baseline,'delta_log_vigor':means}))
        table=pd.concat(tables,ignore_index=True)
        assert len(table)==7200
        table.to_csv(HERE/f'Panel{panel}_mean_bins.csv',index=False)
        panels.append({'panel':panel,'fish':previous['fish'][panel],'finite_bins':int(table.delta_log_vigor.notna().sum()),'total_bins':7200})
        sources.append({'path':str(path),'sha256':expected})
        del frames;gc.collect()
    ref=pd.DataFrame(refs);ref.to_csv(HERE/'baseline_statistics.csv',index=False)
    manifest={'version':'Version7_UnpooledRecipe','fish':previous['fish'],'trials':[5,94],
              'display_interval_s':[-20,20],'baseline_interval_s':[-15,0],'bin_width_s':.5,
              'baseline':'median of original eligible log frames before any bout summarization',
              'bout_summary':'median baseline-centred log value of eligible frames within each extracted trial window; repeat on eligible frames only',
              'bin_value':'mean of finite repeated bout values per 0.5 s bin; frame-count weighting; no pooling',
              'palette':'managua_r','colour_limits':[-.25,.25],'data_scaling':False,'data_clipping':False,
              'undefined_rule':'missing eligible baseline makes whole trial missing',
              'source_data_manifest':str(SOURCE/'data_manifest.json'),'source_data_manifest_sha256':digest(SOURCE/'data_manifest.json'),
              'source_tables':sources,'panels':panels,
              'source_caveat':'Uses the same reconstructed raw vigor and eligibility masks as current F/G/H candidates, not a re-import of processed legacy movement masks.',
              'data_files':[{'path':str(p),'sha256':digest(p)} for p in HERE.glob('*.csv')]}
    (HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    report={'defined_trials':int(ref.defined.sum()),'undefined_trials':int((~ref.defined).sum()),
            'raw_frame_baseline_medians_zero':True,'independent_bout_and_bin_calculations_verified':True,
            'displayed_baseline_bin_medians_need_not_be_zero':True,
            'max_absolute_displayed_baseline_bin_median':float(ref.displayed_baseline_bin_median.abs().max()),
            'panels':panels}
    (HERE/'numeric_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    # Separate renderer copy: retain historical renderer and all prior exports.
    original=REPO/'reviews/fgh_colour_binning_candidates_20261009/render_candidates.py'
    code=original.read_text().replace("('Version4_','Version5_','Version6_')", "('Version4_','Version5_','Version6_','Version7_')")
    (HERE/'render_version7.py').write_text(code)
    sys.path.insert(0,str(HERE))
    import render_version7
    render_version7.render('Version7_UnpooledRecipe',HERE,'delta_log_vigor',True)

if __name__=='__main__': build()
