"""Matched historical LogMedian outcome; unchanged corrected frames and cohort."""
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
import gc
import hashlib
import json
import os
import sys
import argparse

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from classical_conditioning.analysis.inference.learning_onset import LearningOnsetConfig, _fit_mixed_model, _fixed_covariance, model_coefficients, block_global_interaction_test
from render_figure2_delay_legacy_metric_lme import bootstrap_trajectories, adjust_family, BLOCK_FORMULA, LOCAL_FORMULA, BLOCK_ORDER, stars
from render_figure2_delay_phase_lmm import phase_features, contrasts, adequacy, PHASE, GLOBAL

ROOT=Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review')
SOURCE=ROOT/'20261009T141045627289Z-delay-boutonly-lme'
PHASE_SOURCE=ROOT/'20261009T144413666038Z-delay-phase-lmm'
PANEL_SOURCE=ROOT.parent/'sources/20261008T200908002798Z/Fig2_PanelG_allDelay_pre15.figure.json'
METRIC='legacy_distal_angular_speed_rad_per_ms'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')


def window_summary(values):
    """Mean keeps all finite bout values; LogMedian logs positive values once."""
    finite=np.asarray(values,dtype=float)
    finite=finite[np.isfinite(finite)]
    positive=finite[finite>0]
    logged=np.log(positive)
    return {'mean':float(finite.mean()) if len(finite) else np.nan,
            'median':float(np.median(finite)) if len(finite) else np.nan,
            'logmedian':float(np.median(logged)) if len(positive) else np.nan,
            'logmean':float(np.mean(logged)) if len(positive) else np.nan,
            'finite_count':len(finite),'positive_count':len(positive),
            'nonpositive_count':int((finite<=0).sum())}


def extract(out):
    panel=json.loads(PANEL_SOURCE.read_text())
    inputs=[x for x in panel['inputs'] if 'trial-outcomes' in x['path'] and x['path'].endswith('.parquet')]
    rows=[]; lineage=[]
    for i,item in enumerate(inputs):
        path=Path(item['path']); assert sha(path)==item['sha256']
        trials=pd.read_parquet(path)
        trials=trials.loc[trials.metric_id.eq('legacy_distal_angular_speed')&trials.alignment.eq('CS')&trials.trial_number.between(5,94)].sort_values('trial_number')
        recording=str(trials.recording_id.iloc[0]); project=path.parents[2]
        summarypath=project/'Quality checks'/recording/'candidate-trial-outcomes-corrected-v1_summary.json'
        metadata=json.loads(summarypath.read_text())
        assert metadata['artifacts']['outcomes']['sha256']==item['sha256']
        fpath=path.parent/'frame_activity_candidates-corrected-v1.parquet'
        mpath=path.parent/'movement_state_candidates-corrected-v2.parquet'
        for p,key in [(fpath,'candidate_frames'),(mpath,'movement_state')]:
            expected=metadata['inputs'][key]
            assert sha(p)==expected,p
            lineage.append({'path':str(p),'sha256':expected})
        frames=pd.read_parquet(fpath,columns=['FrameID','AbsoluteTime','FrameStep',METRIC])
        movement=pd.read_parquet(mpath,columns=['FrameID','AbsoluteTime','valid','moving','bout_id'])
        np.testing.assert_array_equal(frames.FrameID,movement.FrameID)
        np.testing.assert_array_equal(frames.AbsoluteTime,movement.AbsoluteTime)
        np.testing.assert_array_equal(movement.moving.to_numpy(bool),movement.bout_id.to_numpy()>0)
        time=frames.AbsoluteTime.to_numpy(np.int64); assert np.all(np.diff(time)>=0)
        eligible=movement.valid.to_numpy(bool)&movement.moving.to_numpy(bool)&frames.FrameStep.eq(1).to_numpy()
        values=frames[METRIC].to_numpy(float)
        for _,trial in trials.iterrows():
            start=int(trial.event_start_absolute_time_ms)
            a,b,c=np.searchsorted(time,[start-15000,start,start+9000],side='left')
            baseline=window_summary(values[a:b][eligible[a:b]])
            response=window_summary(values[b:c][eligible[b:c]])
            # Independently establish exactly the same masks/windows as means.
            np.testing.assert_allclose(baseline['mean'],trial.baseline_conditional_intensity,rtol=1e-10,atol=1e-12,equal_nan=True)
            np.testing.assert_allclose(response['mean'],trial.conditional_intensity,rtol=1e-10,atol=1e-12,equal_nan=True)
            row=trial.to_dict()
            row.update(baseline_logmedian=baseline['logmedian'],response_logmedian=response['logmedian'],
                baseline_logmean=baseline['logmean'],response_logmean=response['logmean'],
                baseline_window_median=baseline['median'],response_window_median=response['median'],
                baseline_positive_frames=baseline['positive_count'],response_positive_frames=response['positive_count'],
                baseline_nonpositive_frames=baseline['nonpositive_count'],response_nonpositive_frames=response['nonpositive_count'])
            rows.append(row)
        del frames,movement,time,eligible,values
        gc.collect()
        if (i+1)%5==0 or i+1==len(inputs):
            pd.DataFrame(rows).to_parquet(out/'window-logmedians-all-trials.parquet',index=False)
            dump(out/'frame-source-manifest.json',lineage)
            print(f'Authenticated LogMedian extraction {i+1}/{len(inputs)} fish',flush=True)
    return pd.DataFrame(rows),lineage


def fit(data,formula,cfg,name,out,allow_failed=False):
    result,diag=_fit_mixed_model(data,formula=formula,config=cfg)
    dump(out/(name+'-diagnostics.json'),diag)
    if result is None:
        if not allow_failed: raise RuntimeError(name+' failed: '+str(diag))
        model_coefficients(None,model_name=name,confidence_level=.95).to_csv(out/(name+'-coefficients.csv'),index=False)
        print(name+': unavailable; '+str(diag.get('error')),flush=True)
        return None,diag
    cov=_fixed_covariance(result)
    if diag['hessian_max_eigenvalue']>=0 or np.linalg.eigvalsh(cov).min()<=0:
        raise RuntimeError(name+' failed curvature/fixed covariance')
    model_coefficients(result,model_name=name,confidence_level=.95).to_csv(out/(name+'-coefficients.csv'),index=False)
    names=result.fe_params.index
    pd.DataFrame(cov,index=names,columns=names).to_csv(out/(name+'-fixed-covariance.csv'))
    dump(out/(name+'-random-covariance.json'),np.asarray(result.cov_re).tolist())
    print(name+': numerical checks passed',flush=True)
    return result,diag


def render(summary,tables,dtests,local,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    fig=plt.figure(figsize=(10.5,8.1),layout='constrained')
    grid=fig.add_gridspec(3,1,height_ratios=[2.1,2.5,.65])
    marks=fig.add_subplot(grid[0]); ax=fig.add_subplot(grid[1],sharex=marks); note=fig.add_subplot(grid[2])
    marks.set(ylim=(-.5,4.7),yticks=[4,3,2,1,0],yticklabels=['D: block interaction · Holm','M: block difference · FDR','R: block slope · raw','R: block slope · FDR','Trial change: phase LMM · FDR'])
    marks.tick_params(length=0,labelbottom=False,labelsize=9); marks.spines[['top','right','left','bottom']].set_visible(False)
    for _,r in dtests.iterrows():
        if r.p_holm<.05: marks.text(r.center,4,'D'+stars(r.p_holm),ha='center',va='center',color='#a97900',fontsize=10)
    for _,r in local.iterrows():
        for column,y,label,color in [('mean_p_fdr',3,'M','.35'),('slope_p_raw',2,'R','#a43c24'),('slope_p_fdr',1,'R','#a43c24')]:
            if not np.isfinite(r[column]):
                marks.text(r.center,y,'n/a',ha='center',va='center',color='.55',fontsize=8)
            if r[column]<.05: marks.text(r.center,y,label+stars(r[column]),ha='center',va='center',color=color,fontsize=10)
    trial=tables['phase']; significant=trial.loc[trial.change_p_fdr<.05]
    marks.scatter(significant.trial_number,np.zeros(len(significant)),color='black',marker='*',s=24,lw=0)
    counts=[len(significant),int((local.slope_p_fdr<.05).sum()),int((local.slope_p_raw<.05).sum()),int((local.mean_p_fdr<.05).sum()),int((dtests.p_holm<.05).sum())]
    for y,count in enumerate(counts):
        if count==0: marks.text(49.5,y,'none below p=.05' if y==2 else 'none below adjusted p=.05',color='.5',ha='center',fontsize=8)
    for i,name in enumerate(['Pre','Tr1','Tr2','Tr3','Tr4','Tr5','Te1','Te2','Te3']):
        marks.text(9.5+10*i,4.55,name,ha='center',fontsize=9,color='.35')
    for condition,color,label in [('control','#27aae1','Control (cohort n=28)'),('delay','#f00098','Delay (cohort n=29)')]:
        d=summary.loc[summary.condition_id.eq(condition)].sort_values('trial_number')
        ax.plot(d.trial_number,d['median'],color=color,lw=1.3,label=label)
        ax.fill_between(d.trial_number,d.ci_lower,d.ci_upper,color=color,alpha=.22,lw=0)
    for axis in [marks,ax]:
        for boundary in [14.5,64.5]: axis.axvline(boundary,color='.5',ls=':',lw=.8)
    ax.axhline(0,color='.35',lw=.7)
    ax.set(xlim=(4,95),xlabel='Global CS trial',ylabel='Response − baseline window LogMedian\nmedian across fish [95% fish-bootstrap CI]')
    ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,fontsize=9,loc='lower right')
    note.set_axis_off()
    note.text(0,.95,'Historical LogMedian summary on identical corrected inputs; no extra rolling median or downsampling.',fontsize=9,va='top')
    note.text(0,.59,'One log only. D/M/R: fresh block models. Black stars: fresh phase-aware contrasts relative to Pre; BH90.',fontsize=9,va='top')
    failed_note=' Test3 local M/R unavailable: singular fit.' if (local.status!='ok').any() else ''
    note.text(0,.22,'Exploratory; no onset claim. 5,000 whole-fish resamples, seed 10; NaNs retained.'+failed_note,fontsize=8,va='top',color='#a43c24')
    fig.suptitle('Delay | Historical LogMedian summary — matched-input candidate\nFrozen legacy metric; bout frames only',fontsize=13)
    for ext in ['png','svg']: fig.savefig(out/('Fig2_Delay_LogMedian_stats.'+ext),dpi=220)
    plt.close(fig)
    fig,(obs,fitted)=plt.subplots(2,1,figsize=(10.5,7.8),sharex=True,layout='constrained')
    for condition,color,label in [('control','#27aae1','Control'),('delay','#f00098','Delay')]:
        d=summary.loc[summary.condition_id.eq(condition)].sort_values('trial_number')
        obs.plot(d.trial_number,d['median'],color=color,lw=1.3,label=label)
        obs.fill_between(d.trial_number,d.ci_lower,d.ci_upper,color=color,alpha=.22,lw=0)
    obs.axhline(0,color='.35',lw=.7); obs.legend(frameon=False)
    obs.set(ylabel='Within-window LogMedian difference\ncondition median [95% fish-bootstrap CI]',title='Observed LogMedian — same fish, windows and bout frames')
    mean=pd.read_csv(PHASE_SOURCE/'phase-contrasts.csv')
    for table,color,label in [(mean,'#777777','Arithmetic means: phase-aware LMM'),(tables['phase'],'#6645a2','Within-window LogMedian: phase-aware LMM')]:
        for i,(_,g) in enumerate(table.groupby('fit_phase',observed=True)):
            fitted.plot(g.trial_number,g.change_estimate,color=color,label=label if i==0 else None,lw=1.5)
            fitted.fill_between(g.trial_number,g.change_lower,g.change_upper,color=color,alpha=.15,lw=0)
    fitted.axhline(0,color='.35',lw=.7)
    fitted.set(ylabel='Adjusted control − Delay change from Pre\n(log-vigor units; pointwise model CIs)',xlabel='Global CS trial',title='Same phase-aware model form; different window summaries')
    fitted.legend(frameon=False,fontsize=9)
    for axis in [obs,fitted]:
        axis.set_xlim(4,95); axis.spines[['top','right']].set_visible(False)
        for boundary in [14.5,64.5]: axis.axvline(boundary,color='.5',ls=':',lw=.8)
    fig.suptitle('Delay | typical-intensity and average-intensity candidates\nLMM contrasts are separate from observed medians; one log, no log(x+1)',fontsize=12)
    for ext in ['png','svg']: fig.savefig(out/('Fig2_Delay_LogMedian_observed_fitted.'+ext),dpi=220)
    plt.close(fig)


def main():
    global ROOT, SOURCE, PHASE_SOURCE, PANEL_SOURCE
    parser=argparse.ArgumentParser()
    parser.add_argument('--review-root',type=Path,default=ROOT)
    parser.add_argument('--analysis-dir',type=Path,default=SOURCE)
    parser.add_argument('--phase-dir',type=Path,default=PHASE_SOURCE)
    parser.add_argument('--source-panel',type=Path,default=PANEL_SOURCE)
    parser.add_argument('--extraction-dir',type=Path)
    args=parser.parse_args()
    ROOT, SOURCE, PHASE_SOURCE, PANEL_SOURCE=args.review_root,args.analysis_dir,args.phase_dir,args.source_panel
    out=ROOT/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-delay-logmedian')
    out.mkdir(exist_ok=False); print('OUTPUT_DIRECTORY='+str(out),flush=True)
    if args.extraction_dir:
        record=json.loads((args.extraction_dir/'extraction-complete.json').read_text())
        assert record['source_panel_sha256']==sha(PANEL_SOURCE)
        for name,key in [('window-logmedians-all-trials.parquet','table_sha256'),('frame-source-manifest.json','manifest_sha256')]:
            assert sha(args.extraction_dir/name)==record[key]
        allrows=pd.read_parquet(args.extraction_dir/'window-logmedians-all-trials.parquet')
        lineage=json.loads((args.extraction_dir/'frame-source-manifest.json').read_text())
        assert len(allrows)==5130 and allrows.fish_id.nunique()==57 and len(lineage)==114
        allrows.to_parquet(out/'window-logmedians-all-trials.parquet',index=False)
        dump(out/'frame-source-manifest.json',lineage)
        print('Authenticated completed extraction reused; no frame data copied',flush=True)
    else: allrows,lineage=extract(out)
    dump(out/'extraction-complete.json',{'source_panel_sha256':sha(PANEL_SOURCE),
        'table_sha256':sha(out/'window-logmedians-all-trials.parquet'),
        'manifest_sha256':sha(out/'frame-source-manifest.json')})
    data=allrows.loc[np.isfinite(allrows.baseline_logmedian)&np.isfinite(allrows.response_logmedian)].copy()
    old=pd.read_parquet(SOURCE/'model-input.parquet')
    assert set(zip(data.fish_id,data.trial_number))==set(zip(old.fish_id,old.trial_number)), 'Window positivity changes eligibility; review required'
    data=data.sort_values(['condition_id','fish_id','trial_number']).reset_index(drop=True)
    data['log_baseline']=data.baseline_logmedian
    data['log_response']=data.response_logmedian
    data['logmedian_difference']=data.log_response-data.log_baseline
    data['fish_key']=data.fish_id.astype(str)
    data['trial_center']=old.trial_center.iloc[0]; data['trial_scale']=old.trial_scale.iloc[0]
    data['trial_scaled']=(data.trial_number-data.trial_center)/data.trial_scale
    data['condition_id']=pd.Categorical(data.condition_id,categories=['control','delay'],ordered=True)
    data['block_10_name']=pd.Categorical(data.block_10_name,categories=BLOCK_ORDER,ordered=True)
    features=phase_features(); data=data.merge(features,on='trial_number',validate='many_to_one')
    data.to_parquet(out/'model-input.parquet',index=False)
    fish=data[['fish_id','condition_id','trial_number','logmedian_difference']].rename(columns={'logmedian_difference':'ratio'})
    fish.to_parquet(out/'fish-logmedian-differences.parquet',index=False)
    summary,draws=bootstrap_trajectories(fish,n_boot=5000,seed=10)
    summary.to_parquet(out/'bootstrap-summary.parquet',index=False)
    for condition,item in draws.items(): np.savez_compressed(out/(condition+'-bootstrap-draws.npz'),**item)
    spec={'metric_id':'legacy_distal_angular_speed','outcome':'window-logmedian-bout-intensity',
        'definition':'median(log positive bout response frames) minus median(log positive bout baseline frames); then median across fish',
        'baseline_s':[-15,0],'response_s':[0,9],'preprocessing':'identical corrected frames and shared bout detector; no added historical smoothing/downsampling',
        'historical_relation':'historical LogMedian summary only; not an exact rerun of historical preprocessing or historical inference',
        'model_response':'response window median(log vigor)','model_baseline':'baseline window median(log vigor); covariate; no second log',
        'units':'dimensionless difference of log(rad/ms) summaries; model contrasts log-vigor units',
        'baseline_nonpositive_frames_excluded':int(allrows.baseline_nonpositive_frames.sum()),
        'response_nonpositive_frames_excluded':int(allrows.response_nonpositive_frames.sum()),
        'scheduled_rows':len(allrows),'usable_rows':len(data),'cohort_fish':data.fish_id.nunique(),
        'trial_formula':PHASE,'global_sensitivity_formula':GLOBAL,'block_formula':BLOCK_FORMULA,'local_formula':LOCAL_FORMULA,
        'bootstrap':{'resamples':5000,'seed':10,'unit':'whole fish trajectories within condition; NaNs retained'},
        'D':'Holm8 block interactions','M':'BH9 centered local condition differences','R':'raw plus BH9 local slopes',
        'trial':'BH90 two-sided control-minus-Delay change relative to average Pre5-14, phase-aware model',
        'status':'exploratory candidate, no outcome or paper-model selection inferred'}
    dump(out/'candidate-specification.json',spec)
    config=LearningOnsetConfig(metric_id='legacy_distal_angular_speed',outcome_id='conditional-intensity',n_bootstrap=0,allow_random_intercept_fallback=False)
    fits={}; tables={}; reviews=[]; diagnostics=[]
    with threadpool_limits(limits=1):
        for name,formula,cfg in [('block',BLOCK_FORMULA,config),('global',GLOBAL,config),('phase',PHASE,config),('phase-powell',PHASE,replace(config,optimizer='powell')),('phase-intercept',PHASE,replace(config,optimizer='powell',random_effects_formula='1'))]:
            result,diag=fit(data,formula,cfg,name,out); fits[name]=result; diagnostics.append({'name':name,**diag})
            residual,review=adequacy(result,data,name,diag); review['residual_sd_by_condition_phase']={str(k):v for k,v in review['residual_sd_by_condition_phase'].items()}
            reviews.append(review); residual.to_parquet(out/(name+'-residuals.parquet'),index=False)
            if name!='block':
                table,condition,change=contrasts(result,data,features); tables[name]=table
                table.to_csv(out/(name+'-contrasts.csv'),index=False); np.savez_compressed(out/(name+'-contrast-matrices.npz'),condition=condition,change=change)
        block=fits['block']; dtests=[]
        for term in block.fe_params.index:
            if 'condition_id' in term and ':' in term:
                name=next(n for n in BLOCK_ORDER if '[T.'+n+']' in term)
                dtests.append({'block_10_name':name,'center':data.loc[data.block_10_name.eq(name),'trial_number'].mean(),
                    'estimate':block.fe_params[term],'p_raw':block.pvalues[term]})
        dtests=pd.DataFrame(dtests); adjust_family(dtests,'p_raw','p_holm','holm'); dtests.to_csv(out/'D-interaction-tests.csv',index=False)
        joint=block_global_interaction_test(block); joint.to_csv(out/'joint-interaction-test.csv',index=False)
        local=[]
        for name in BLOCK_ORDER:
            sub=data.loc[data.block_10_name.eq(name)].copy(); sub['within_block_trial']=sub.trial_number-sub.trial_number.mean()
            result,diag=fit(sub,LOCAL_FORMULA,replace(config,random_effects_formula='1',optimizer='powell'),name.replace(' ','-')+'-local',out,allow_failed=True)
            diagnostics.append({'name':name+'-local',**diag})
            row={'block_10_name':name,'center':sub.trial_number.mean(),'status':diag['diagnostic_status'],
                'mean_estimate':np.nan,'mean_p_raw':np.nan,'slope_estimate':np.nan,'slope_p_raw':np.nan}
            for term in result.fe_params.index if result is not None else []:
                if 'condition_id' in term:
                    label='slope' if ':' in term else 'mean'
                    row[label+'_estimate']=result.fe_params[term]; row[label+'_p_raw']=result.pvalues[term]
            local.append(row)
        local=pd.DataFrame(local)
        for label in ['mean','slope']: adjust_family(local,label+'_p_raw',label+'_p_fdr','fdr_bh')
        local.to_csv(out/'MR-local-block-tests.csv',index=False)
        influence=[]
        for i,fish_id in enumerate(sorted(data.fish_key.unique())):
            sub=data.loc[~data.fish_key.eq(fish_id)]
            result,diag=_fit_mixed_model(sub,formula=PHASE,config=config,collect_extended_diagnostics=False)
            row={'omitted_fish':fish_id,'status':diag['diagnostic_status']}
            if result is not None:
                table,_,_=contrasts(result,sub,features)
                row['max_change_shift']=float(abs(table.change_estimate-tables['phase'].change_estimate).max())
            influence.append(row)
            if (i+1)%10==0 or i==56:
                pd.DataFrame(influence).to_csv(out/'phase-leave-one-fish-out.csv',index=False)
                print(f'LogMedian influence refits {i+1}/57',flush=True)
    render(summary,tables,dtests,local,out)
    report={'models':len(diagnostics),'failed_model_names':[d['name'] for d in diagnostics if d['diagnostic_status']!='ok'],
        'D':int((dtests.p_holm<.05).sum()),'M':int((local.mean_p_fdr<.05).sum()),
        'R_raw':int((local.slope_p_raw<.05).sum()),'R_fdr':int((local.slope_p_fdr<.05).sum()),
        'trial_significant':{name:table.loc[table.change_p_fdr<.05,'trial_number'].tolist() for name,table in tables.items()},
        'joint_interaction':joint.to_dict('records'),'adequacy':reviews,
        'influence_failed':sum(r['status']!='ok' for r in influence),'max_influence_shift':max(r.get('max_change_shift',0) for r in influence)}
    dump(out/'result-summary.json',report); pd.DataFrame(diagnostics).to_csv(out/'all-fit-diagnostics.csv',index=False)
    (out/'analysis-script.py').write_bytes(Path(__file__).read_bytes())
    for script in ['render_figure2_delay_phase_lmm.py','render_figure2_delay_legacy_metric_lme.py']:
        (out/script).write_bytes((Path(__file__).parent/script).read_bytes())
    dump(out/'Fig2_Delay_LogMedian.figure.json',{'specification':spec,'results':report,'source_inputs':lineage,
        'source_panel':{'path':str(PANEL_SOURCE),'sha256':sha(PANEL_SOURCE)},
        'outputs':[{'path':str(p),'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({'directory':str(out),'summary':report},indent=2),flush=True)


if __name__=='__main__': main()
