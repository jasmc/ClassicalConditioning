"""Version 4 with arithmetic means, -inf non-bout frames and a 20 s baseline."""
from pathlib import Path
import sys,json,gc
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
CLASSIFIED=REPO/'reviews/fgh_version4_baseline20_20261009'
sys.path.insert(0,str(REPO/'reviews/fgh_legacy_layout_20261009'))
import render_layout as plotting
from render_layout import digest
EDGES=np.arange(-20,20.5,.5)

def build():
    inputs=json.loads((CLASSIFIED/'frame_classification_manifest.json').read_text())
    refs=[];panels=[];audit=[]
    for info in inputs['panels']:
        path=Path(info['path']);assert digest(path)==info['sha256']
        frames=pd.read_parquet(path)
        values=frames.log_vigor.to_numpy().copy();values[frames.valid_nonbout.to_numpy()]=-np.inf
        assert np.isnan(values[~frames.eligible.to_numpy()&~frames.valid_nonbout.to_numpy()]).all()
        frames['input_log_vigor']=values
        tables=[];defined_count=0
        for trial,p in frames.groupby('trial',sort=True):
            idx=np.searchsorted(EDGES,p.time_s.to_numpy(),side='right')-1
            v=p.input_log_vigor.to_numpy()
            good=~np.isnan(v);floor=np.isneginf(v)
            count=np.bincount(idx[good],minlength=80)
            floor_count=np.bincount(idx[floor],minlength=80)
            means=np.full(80,np.nan)
            # The arithmetic mean is -inf whenever at least one lower-bound frame contributes.
            for b in range(80):
                samples=v[(idx==b)&good]
                if len(samples):means[b]=float(np.mean(samples))
            assert np.array_equal(np.isneginf(means),floor_count>0)
            assert np.array_equal(np.isnan(means),count==0)
            finite_only=p[p.eligible].assign(bin_index=idx[p.eligible.to_numpy()]).groupby('bin_index').log_vigor.mean().reindex(range(80)).to_numpy()
            check=(floor_count==0)&(count>0)
            np.testing.assert_allclose(means[check],finite_only[check],atol=1e-12,rtol=0)
            base=means[:40][~np.isnan(means[:40])]
            baseline=float(np.mean(base)) if len(base) else np.nan
            defined=bool(np.isfinite(baseline));defined_count+=defined
            delta=means-baseline if defined else np.full(80,np.nan)
            if defined:assert abs(np.mean(delta[:40][~np.isnan(delta[:40])]))<1e-12
            reason='' if defined else ('baseline mean is -inf' if np.isneginf(baseline) else 'no baseline values')
            refs.append({'panel':info['panel'],'fish':info['fish'],'trial':int(trial),'baseline_mean_log_vigor':baseline,
                'nonmissing_baseline_bins':len(base),'baseline_negative_infinite_bins':int(np.isneginf(base).sum()),
                'defined':defined,'undefined_reason':reason})
            audit.append({'panel':info['panel'],'trial':int(trial),'defined':defined,
                'baseline_above_zero':int((delta[:40]>1e-12).sum()),'baseline_below_zero':int((delta[:40]<-1e-12).sum()),
                'baseline_tied_at_zero':int((~np.isnan(delta[:40])&(abs(delta[:40])<=1e-12)).sum()),
                'mean_centring_does_not_guarantee_equal_sign_counts':True})
            total=np.bincount(idx,minlength=80)
            tables.append(pd.DataFrame({'trial':int(trial),'bin_index':range(80),'start_s':EDGES[:-1],'end_s':EDGES[1:],
                'total_frame_count':total,'eligible_frame_count':count-floor_count,'valid_nonbout_frame_count':floor_count,
                'nan_sample_count':total-count,'mean_log_vigor':means,'baseline_mean_log_vigor':baseline,
                'trial_defined':defined,'delta_log_vigor':delta}))
        table=pd.concat(tables,ignore_index=True);assert len(table)==7200
        table.to_csv(HERE/f"Panel{info['panel']}_mean_bins.csv",index=False)
        panels.append({'panel':info['panel'],'fish':info['fish'],'defined_trials':int(defined_count),'undefined_trials':90-int(defined_count),
            'negative_infinite_bin_means':int(np.isneginf(table.mean_log_vigor).sum()),'frame_means_and_lower_bound_propagation_verified':True})
        del frames;gc.collect()
    statistics=pd.DataFrame(refs);statistics.to_csv(HERE/'baseline_statistics.csv',index=False)
    pd.DataFrame(audit).to_csv(HERE/'per_trial_baseline_balance.csv',index=False)
    report={'version':'Version4_DirectBinMeans','panels':panels,'defined_trials':int(statistics.defined.sum()),
        'undefined_trials':int((~statistics.defined).sum()),'mean_of_bins_baseline_verified':True,
        'invalid_frames_remain_NaN':True,'other_versions_unchanged':True,
        'balance':'an arithmetic mean does not guarantee equal above/below counts; undefined trials cannot be assessed after centring'}
    (HERE/'numeric_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    manifest={'version':'Version4_DirectBinMeans','fish':{p['panel']:p['fish'] for p in panels},
        'bin_width_s':.5,'display_interval_s':[-20,20],'baseline_interval_s':[-20,0],'trials':[5,94],
        'frame_value':'eligible bout natural-log vigor; -inf for valid non-bout frames; NaN for remaining invalid/ineligible frames',
        'bin_value':'arithmetic mean of non-NaN frame log values, including -inf, per 0.5 s bin',
        'baseline':'arithmetic mean of non-NaN baseline-bin means in [-20,0), including -inf; equal weight per bin',
        'undefined_rule':'Leave the whole trial undefined whenever the baseline mean is not finite, as with the selected -inf baseline rule',
        'formula':'bin_mean - baseline_mean when baseline_mean is finite; otherwise whole trial undefined',
        'data_scaling':False,'data_clipping':False,'percentile_scaling':False,'palette':'managua_r','colour_limits':[-.25,.25],
        'source_classification_manifest':str(CLASSIFIED/'frame_classification_manifest.json'),
        'source_classification_sha256':digest(CLASSIFIED/'frame_classification_manifest.json'),
        'source_tables':[{'path':r['path'],'sha256':r['sha256']} for r in inputs['panels']],
        'panels':panels,'data_files':[{'path':str(p),'sha256':digest(p)} for p in HERE.glob('*.csv')]}
    (HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

def register():
    config_path=REPO/'configs/paper-figures/figure1-fgh-full-bout-correction-20261009.json'
    config=json.loads(config_path.read_text())
    for v in config['variants']:
        if v['variant'].startswith('Version4_'):
            v['previous_layout_validation']=v['validation'];v['variant']='Version4_DirectBinMeans'
            path=HERE/'Version4_DirectBinMeans_validation.json';v['validation']=str(path);v['validation_sha256']=digest(path)
    config['version4_data_manifest']=str(HERE/'data_manifest.json');config['version4_data_manifest_sha256']=digest(HERE/'data_manifest.json')
    config['version4_numeric_verification']=str(HERE/'numeric_verification.json')
    config['version4_recipe']='Arithmetic mean per 0.5 s bin, including valid non-bout -inf frames; mean of [-20,0) baseline bins; finite baseline subtraction only'
    config['baseline']='C/D samples and Version 2 retain [-15,0); Version 4 uses mean of bin means in [-20,0), including -inf'
    config['status']='Only Version 4 revised to arithmetic means; -inf non-bout frames and undefined-baseline rule retained'
    config_path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    build();plotting.HERE=HERE;plotting.render('Version4_DirectBinMeans',HERE,'delta_log_vigor',True);register()
