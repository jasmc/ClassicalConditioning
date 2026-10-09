"""Version 4: valid non-bout frames at -inf, [-20,0) baseline, no scaling."""
from pathlib import Path
import sys,json,gc
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'reviews/fgh_legacy_layout_20261009'))
import render_layout as plotting
from render_layout import digest
EDGES=np.arange(-20,20.5,.5);TOL=1e-12

def sign_counts(values):
    values=np.asarray(values,float);good=~np.isnan(values)
    return {'nonmissing':int(good.sum()),'above_zero':int((values>TOL).sum()),
            'below_zero':int((values<-TOL).sum()),'tied_at_zero':int((good&(abs(values)<=TOL)).sum())}

def build():
    classification=json.loads((HERE/'frame_classification_manifest.json').read_text())
    references=[];reports=[];balance=[]
    for info in classification['panels']:
        panel=info['panel'];path=Path(info['path']);assert digest(path)==info['sha256']
        frames=pd.read_parquet(path)
        np.testing.assert_array_equal(frames.log_vigor.notna(),frames.eligible)
        assert (frames.loc[frames.valid_nonbout,'bout_id']==0).all()
        # Only valid non-bout frames get the lower bound. Other missing values stay NaN.
        frames['bin_input_log_vigor']=frames.log_vigor
        frames.loc[frames.valid_nonbout,'bin_input_log_vigor']=-np.inf
        assert frames.loc[~frames.eligible&~frames.valid_nonbout,'bin_input_log_vigor'].isna().all()
        tables=[];residuals=[]
        for trial,p in frames.groupby('trial',sort=True):
            idx=np.searchsorted(EDGES,p.time_s.to_numpy(),side='right')-1
            assert ((idx>=0)&(idx<80)).all()
            values=p.bin_input_log_vigor.to_numpy()
            total=np.bincount(idx,minlength=80)
            bout_count=np.bincount(idx[p.eligible.to_numpy()],minlength=80)
            nonbout_count=np.bincount(idx[p.valid_nonbout.to_numpy()],minlength=80)
            medians=p.assign(bin_index=idx).groupby('bin_index').bin_input_log_vigor.median().reindex(range(80)).to_numpy()
            checked=np.array([np.median(values[(idx==i)&~np.isnan(values)]) if np.any((idx==i)&~np.isnan(values)) else np.nan for i in range(80)])
            np.testing.assert_array_equal(medians,checked)
            assert np.array_equal(np.isnan(medians),bout_count+nonbout_count==0)
            base=medians[:40][~np.isnan(medians[:40])]
            reference=float(np.median(base)) if len(base) else np.nan
            defined=bool(np.isfinite(reference))
            delta=medians-reference if defined else np.full(80,np.nan)
            counts=sign_counts(delta[:40]);whole=sign_counts(delta)
            if defined:
                residual=float(abs(np.median(delta[:40][~np.isnan(delta[:40])])))
                assert residual<TOL;residuals.append(residual)
                assert counts['above_zero']<=counts['nonmissing']/2 and counts['below_zero']<=counts['nonmissing']/2
            references.append({'panel':panel,'fish':info['fish'],'trial':int(trial),
                'nonmissing_baseline_bin_count':len(base),'baseline_lower_bound_bins':int(np.isneginf(base).sum()),
                'baseline_median_log_vigor':reference,'defined':defined,
                'undefined_reason':'' if defined else ('baseline median is -inf' if np.isneginf(reference) else 'no baseline values')})
            balance.append({'panel':panel,'trial':int(trial),'defined':defined,
                **{'baseline_'+k:v for k,v in counts.items()},**{'whole_window_'+k:v for k,v in whole.items()},
                'absolute_above_below_difference':abs(counts['above_zero']-counts['below_zero']) if defined else np.nan,
                'raw_baseline_bins_above_reference':int((base>reference).sum()) if len(base) else 0,
                'raw_baseline_bins_below_reference':int((base<reference).sum()) if len(base) else 0,
                'raw_baseline_bins_equal_reference':int((base==reference).sum()) if len(base) else 0})
            tables.append(pd.DataFrame({'trial':int(trial),'bin_index':range(80),'start_s':EDGES[:-1],'end_s':EDGES[1:],
                'total_frame_count':total,'eligible_frame_count':bout_count,'valid_nonbout_frame_count':nonbout_count,
                'nan_sample_count':total-bout_count-nonbout_count,'median_log_vigor':medians,
                'baseline_median_log_vigor':reference,'trial_defined':defined,'delta_log_vigor':delta}))
        table=pd.concat(tables,ignore_index=True);assert len(table)==7200
        table.to_csv(HERE/f'Panel{panel}_median_bins.csv',index=False)
        panel_refs=pd.DataFrame(references);panel_refs=panel_refs[panel_refs.panel.eq(panel)]
        reports.append({'panel':panel,'fish':info['fish'],'defined_trials':int(panel_refs.defined.sum()),
            'undefined_trials':int((~panel_refs.defined).sum()),'lower_bound_bin_medians':int(np.isneginf(table.median_log_vigor).sum()),
            'displayed_lower_endpoint_bins':int(np.isneginf(table.delta_log_vigor).sum()),
            'maximum_absolute_defined_baseline_median':max(residuals) if residuals else None})
        del frames;gc.collect()
    stats=pd.DataFrame(references);stats.to_csv(HERE/'baseline_statistics.csv',index=False)
    audit=pd.DataFrame(balance);audit.to_csv(HERE/'per_trial_baseline_balance.csv',index=False)
    finite=audit[audit.defined]
    report={'panels':reports,'defined_trials':int(stats.defined.sum()),'undefined_trials':int((~stats.defined).sum()),
        'defined_trial_baseline_medians_zero':True,'tolerance_log_units':TOL,
        'defined_trials_with_equal_above_below_counts':int(finite.absolute_above_below_difference.eq(0).sum()),
        'maximum_defined_above_below_count_difference':float(finite.absolute_above_below_difference.max()) if len(finite) else None,
        'whole_trial_balance_not_required':True,'each_bin_median_independently_verified':True,
        'invalid_or_ineligible_inside_bout_frames_remain_NaN':True,'remaining_versions_unchanged':True}
    (HERE/'numeric_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    source_files=[{'path':r['path'],'sha256':r['sha256']} for r in classification['panels']]
    manifest={'version':'Version4_DirectBinMedians','fish':{r['panel']:r['fish'] for r in classification['panels']},
        'baseline_interval_s':[-20,0],'display_interval_s':[-20,20],'bin_width_s':.5,'trials':[5,94],
        'frame_value':'direct eligible natural-log vigor in bouts; -inf for valid non-bout frames; NaN for remaining invalid/ineligible frames',
        'bin_value':'median of all non-NaN frame values, including -inf non-bout contributions, within each CS-aligned 0.5 s bin',
        'baseline':'median of non-NaN bin medians in [-20,0); include -inf bin medians; one scalar per bin',
        'formula':'delta_log_vigor = bin_median - baseline_median, only when baseline_median is finite',
        'undefined_rule':'User selected: leave the entire trial undefined when baseline median is -inf; no alternate zero or finite reference',
        'data_scaling':False,'data_clipping':False,'percentile_scaling':False,'colour_limits':[-.25,.25],'palette':'managua_r',
        'negative_infinite_display_values':'use the negative endpoint colour when the baseline is finite',
        'source_frame_classification':str(HERE/'frame_classification_manifest.json'),
        'source_frame_classification_sha256':digest(HERE/'frame_classification_manifest.json'),'source_tables':source_files,
        'data_files':[{'path':str(HERE/name),'sha256':digest(HERE/name)} for name in
            ['PanelF_median_bins.csv','PanelG_median_bins.csv','PanelH_median_bins.csv','baseline_statistics.csv','per_trial_baseline_balance.csv']],
        'panels':reports}
    (HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

def render_and_register():
    plotting.HERE=HERE
    plotting.render('Version4_DirectBinMedians',HERE,'delta_log_vigor',True)
    validation=HERE/'Version4_DirectBinMedians_validation.json'
    config_path=REPO/'configs/paper-figures/figure1-fgh-full-bout-correction-20261009.json'
    config=json.loads(config_path.read_text())
    for variant in config['variants']:
        if variant['variant']=='Version4_DirectBinMedians':
            variant['previous_layout_validation']=variant['validation'];variant['validation']=str(validation)
            variant['validation_sha256']=digest(validation)
    config['version4_data_manifest']=str(HERE/'data_manifest.json')
    config['version4_data_manifest_sha256']=digest(HERE/'data_manifest.json')
    config['version4_numeric_verification']=str(HERE/'numeric_verification.json')
    config['version4_recipe']='Valid non-bout frames contribute -inf before 0.5 s medians; [-20,0) bin-median baseline; finite reference subtraction only'
    config['baseline']='C/D samples: [-15,0) complete-bout timepoints; Version 2: [-15,0) direct bin means; Version 4: [-20,0) bin medians including non-bout lower bounds'
    config['status']='Only Version 4 revised: 20 s baseline and valid non-bout -inf frame contributions; other versions retained'
    config_path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':build();render_and_register()
