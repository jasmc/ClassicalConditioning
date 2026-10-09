"""Observed bout-only medians and separate global/phase-aware LMM contrasts."""
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os
import sys
import argparse

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices
from scipy.stats import norm, skew, kurtosis, probplot
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from classical_conditioning.analysis.inference.learning_onset import LearningOnsetConfig, _fit_mixed_model, _fixed_covariance, model_coefficients

ROOT = Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review')
SOURCE = ROOT / '20261009T141045627289Z-delay-boutonly-lme'
GLOBAL = "log_response ~ log_baseline + C(condition_id, Treatment(reference='control')) * bs(trial_scaled, df=5, degree=3, include_intercept=False)"
PHASE = "log_response ~ log_baseline + C(condition_id, Treatment(reference='control')) * (C(fit_phase, Treatment(reference='Pre')) + pre_trial + train_b0 + train_b1 + train_b2 + train_b3 + test_b0 + test_b1 + test_b2 + test_b3)"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str)+'\n', encoding='utf-8')


def phase_features():
    """Schedule-defined bases: no cross-phase basis values or learned knots."""
    trials = np.arange(5,95)
    frame = pd.DataFrame({'trial_number':trials})
    frame['fit_phase'] = pd.Categorical(np.where(trials<15,'Pre',np.where(trials<65,'Train','Test')),categories=['Pre','Train','Test'])
    frame['pre_trial'] = np.where(trials<15,(trials-9.5)/9,0)
    for phase,start,end,prefix in [('Train',15,64,'train'),('Test',65,94,'test')]:
        selected=(trials>=start)&(trials<=end)
        u=(trials[selected]-start)/(end-start)
        basis=np.asarray(dmatrix('bs(u, df=4, degree=3, include_intercept=False, lower_bound=0, upper_bound=1)-1',{'u':u}))
        for i in range(4):
            frame[f'{prefix}_b{i}']=0.0
            frame.loc[selected,f'{prefix}_b{i}']=basis[:,i]
    return frame


def contrasts(result, data, features):
    """Common-baseline fixed-effect contrast; zero random effects for population."""
    rows=[]
    for condition in ['control','delay']:
        cases=features.copy()
        cases['condition_id']=condition
        cases['log_baseline']=data.log_baseline.mean()
        cases['trial_scaled']=(cases.trial_number-data.trial_center.iloc[0])/data.trial_scale.iloc[0]
        rows.append(np.asarray(build_design_matrices([result.model.data.design_info],cases)[0]))
    condition=rows[0]-rows[1]
    change=condition-condition[:10].mean(axis=0)
    covariance=_fixed_covariance(result)
    beta=np.asarray(result.fe_params)
    output=features[['trial_number','fit_phase']].copy()
    for label,matrix in [('condition',condition),('change',change)]:
        estimate=matrix@beta
        se=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',matrix,covariance,matrix),0))
        output[label+'_estimate']=estimate
        output[label+'_se']=se
        output[label+'_lower']=estimate-norm.ppf(.975)*se
        output[label+'_upper']=estimate+norm.ppf(.975)*se
        output[label+'_p_raw']=2*norm.sf(np.divide(abs(estimate),se,out=np.full_like(se,np.nan),where=se>0))
        output[label+'_p_fdr']=multipletests(output[label+'_p_raw'].fillna(1),method='fdr_bh')[1]
    return output,condition,change


def adequacy(result,data,name,diag):
    residual=np.asarray(result.resid)
    pairs=data[['fish_key','condition_id','trial_number','fit_phase']].copy()
    pairs['residual']=residual
    pairs['fitted']=np.asarray(result.fittedvalues)
    shifted=pairs.copy(); shifted.trial_number-=1
    adjacent=pairs.merge(shifted,on=['fish_key','trial_number'],suffixes=('_now','_next'))
    lag=[g.residual_now.corr(g.residual_next) for _,g in adjacent.groupby('fish_key')]
    hessian_ok=diag['hessian_max_eigenvalue']<0
    cov=_fixed_covariance(result)
    return pairs,{'model':name,'numerical_status':diag['diagnostic_status'],
        'hessian_ok':bool(hessian_ok),'fixed_covariance_positive':bool(np.linalg.eigvalsh(cov).min()>0),
        'aic':float(result.aic),'bic':float(result.bic),'log_likelihood':float(result.llf),
        'fixed_coefficients':len(result.fe_params),'residual_skewness':float(skew(residual)),
        'residual_excess_kurtosis':float(kurtosis(residual)),
        'median_true_adjacent_lag1':float(np.nanmedian(lag)),
        'residual_sd_by_condition_phase':pairs.groupby(['condition_id','fit_phase'],observed=True).residual.std().to_dict()}


def render(summary,tables,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    fig=plt.figure(figsize=(10.5,8.3),layout='constrained')
    grid=fig.add_gridspec(3,1,height_ratios=[3,3,.65])
    obs=fig.add_subplot(grid[0]); fitted=fig.add_subplot(grid[1],sharex=obs); note=fig.add_subplot(grid[2])
    for condition,color,label in [('control','#27aae1','Control (cohort n=28)'),('delay','#f00098','Delay (cohort n=29)')]:
        d=summary.loc[summary.condition_id.eq(condition)].sort_values('trial_number')
        obs.plot(d.trial_number,d['median'],color=color,lw=1.25,label=label)
        obs.fill_between(d.trial_number,d.ci_lower,d.ci_upper,color=color,alpha=.22,lw=0)
    obs.axhline(1,color='.35',lw=.7)
    obs.set(ylabel='Bout-only response mean / baseline mean\ncondition median and pointwise 95% bootstrap CI',ylim=(.72,1.19),xlim=(4,95),title='Observed trials — unsmoothed; 5,000 whole-fish resamples, seed 10')
    obs.legend(frameon=False,fontsize=9,loc='lower right')
    for name,color,style,label in [('global','#777777','--','Global cubic spline (previous model)'),('phase','#6645a2','-','Phase-aware LMM (new candidate)')]:
        d=tables[name]
        # Segment the phase-aware curve: do not draw artificial continuity.
        groups=[d] if name=='global' else [g for _,g in d.groupby('fit_phase',observed=True)]
        for i,g in enumerate(groups):
            fitted.plot(g.trial_number,g.change_estimate,color=color,linestyle=style,lw=1.6,label=label if i==0 else None)
            fitted.fill_between(g.trial_number,g.change_lower,g.change_upper,color=color,alpha=.16 if name=='phase' else .10,lw=0)
    fitted.axhline(0,color='.35',lw=.7)
    fitted.set(ylabel='Adjusted change in control − Delay\nrelative to average Pre5–14 (log-response units)',xlabel='Global CS trial',title='Fitted population contrasts — pointwise 95% model CIs')
    fitted.legend(frameon=False,fontsize=9,loc='upper right')
    for ax in [obs,fitted]:
        for boundary in [14.5,64.5]: ax.axvline(boundary,color='.5',linestyle=':',lw=.8)
        ax.spines[['top','right']].set_visible(False)
    for center,label in [(9.5,'Pre'),(39.5,'Training'),(79.5,'Test')]:
        obs.text(center,1.18,label,ha='center',va='top',fontsize=9,color='.35')
    note.set_axis_off()
    note.text(0,.92,'Positive fitted contrast: greater response suppression in Delay relative to control and Pre.',fontsize=9,va='top')
    note.text(0,.60,'Observed bands resample fish; fitted bands use the LMM covariance. These are different uncertainties.',fontsize=9,va='top')
    note.text(0,.28,'Exploratory: heavy residual tails remain. No onset claim, simultaneous band or new D/M/R strip.',fontsize=9,va='top',color='#a43c24')
    fig.suptitle('Delay G | frozen legacy metric; bout frames only\nObserved medians and model contrasts shown separately',fontsize=13)
    for ext in ['png','svg','pdf']: fig.savefig(out/('Fig2_Delay_observed_and_fitted_contrasts.'+ext),dpi=220)
    plt.close(fig)


def main():
    global ROOT, SOURCE
    parser=argparse.ArgumentParser()
    parser.add_argument('--review-root',type=Path,default=ROOT)
    parser.add_argument('--analysis-dir',type=Path,default=SOURCE)
    args=parser.parse_args()
    ROOT, SOURCE=args.review_root,args.analysis_dir
    out=ROOT/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-delay-phase-lmm')
    out.mkdir(exist_ok=False)
    print('OUTPUT_DIRECTORY='+str(out),flush=True)
    side=json.loads((SOURCE/'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json').read_text())
    for item in side['outputs']: assert sha(item['path'])==item['sha256'],item['path']
    data=pd.read_parquet(SOURCE/'model-input.parquet')
    summary=pd.read_parquet(SOURCE/'bootstrap-summary.parquet')
    features=phase_features()
    data=data.merge(features,on='trial_number',validate='many_to_one')
    data.to_parquet(out/'model-input.parquet',index=False)
    features.to_csv(out/'phase-features.csv',index=False)
    spec={'metric_id':'legacy_distal_angular_speed','outcome_id':'conditional-intensity',
        'observed_summary':'equal-fish median of bout-only arithmetic response/baseline means',
        'windows_s':{'baseline':[-15,0],'response':[0,9]},'phase_trials':{'Pre':[5,14],'Train':[15,64],'Test':[65,94]},
        'phase_shape':'Pre linear; Train/Test separate cubic splines, df4, phase-specific intercepts, no continuity constraints',
        'phase_basis':'schedule-defined; normalized phase trial; one interior knot at phase midpoint',
        'global_formula':GLOBAL,'phase_formula':PHASE,'random_effects':'fish intercept + scaled-trial slope',
        'transform':'ln positive arithmetic bout-window response; ln baseline covariate; no offset',
        'estimator':'ML','primary_optimizer':'lbfgs','sensitivity':'Powell and random-intercept-only phase fits; 57 leave-one-fish-out fits',
        'trial_tests':'two-sided change relative to average scheduled Pre5-14; BH90 retained for comparison; no trial significance strip',
        'uncertainty':'pointwise Gaussian model CI, distinct from 5000/seed10 whole-fish descriptive CI',
        'onset':'not estimated; no simultaneous onset band','status':'exploratory candidate; not selected from p-values'}
    dump(out/'candidate-specification.json',spec)
    config=LearningOnsetConfig(metric_id=spec['metric_id'],outcome_id=spec['outcome_id'],n_bootstrap=0,allow_random_intercept_fallback=False)
    fits={}; tables={}; reviews=[]; diagnostics=[]
    with threadpool_limits(limits=1):
        for name,formula,cfg in [('global',GLOBAL,config),('phase',PHASE,config),('phase-powell',PHASE,replace(config,optimizer='powell')),('phase-intercept',PHASE,replace(config,optimizer='powell',random_effects_formula='1'))]:
            result,diag=_fit_mixed_model(data,formula=formula,config=cfg)
            diagnostics.append({'name':name,**diag}); dump(out/(name+'-diagnostics.json'),diag)
            if result is None: raise RuntimeError(name+' failed: '+str(diag))
            residual,review=adequacy(result,data,name,diag)
            review['residual_sd_by_condition_phase']={str(k):v for k,v in review['residual_sd_by_condition_phase'].items()}
            if not review['hessian_ok'] or not review['fixed_covariance_positive']: raise RuntimeError(name+' failed covariance/curvature gate')
            fits[name]=result; reviews.append(review)
            residual.to_parquet(out/(name+'-residuals.parquet'),index=False)
            model_coefficients(result,model_name=name,confidence_level=.95).to_csv(out/(name+'-coefficients.csv'),index=False)
            names=result.fe_params.index
            pd.DataFrame(_fixed_covariance(result),index=names,columns=names).to_csv(out/(name+'-fixed-covariance.csv'))
            dump(out/(name+'-random-covariance.json'),np.asarray(result.cov_re).tolist())
            table,condition,change=contrasts(result,data,features)
            tables[name]=table; table.to_csv(out/(name+'-contrasts.csv'),index=False)
            np.savez_compressed(out/(name+'-contrast-matrices.npz'),condition=condition,change=change)
            print(name+': numerical checks passed; kurtosis='+str(review['residual_excess_kurtosis']),flush=True)
        old=pd.read_csv(SOURCE/'trial-tests-FDR.csv')
        np.testing.assert_allclose(tables['global'].change_estimate,old.learning_contrast,atol=1e-10)
        influence=[]
        for i,fish in enumerate(sorted(data.fish_key.unique())):
            sub=data.loc[~data.fish_key.eq(fish)]
            result,diag=_fit_mixed_model(sub,formula=PHASE,config=config,collect_extended_diagnostics=False)
            row={'omitted_fish':fish,'status':diag['diagnostic_status']}
            if result is not None:
                table,_,_=contrasts(result,sub,features)
                row['max_change_shift']=float(abs(table.change_estimate-tables['phase'].change_estimate).max())
                row['mean_training_change']=float(table.loc[table.trial_number.between(15,64),'change_estimate'].mean())
            influence.append(row)
            if (i+1)%10==0 or i==56:
                pd.DataFrame(influence).to_csv(out/'phase-leave-one-fish-out.csv',index=False)
                print(f'Phase influence refits {i+1}/57',flush=True)
    dump(out/'model-adequacy.json',reviews)
    pd.DataFrame(diagnostics).to_csv(out/'all-fit-diagnostics.csv',index=False)
    results={name:{'significant_trials':t.loc[t.change_p_fdr<.05,'trial_number'].tolist(),
                   'pre_significant':int((t.change_p_fdr.iloc[:10]<.05).sum()),
                   'mean_training_change':float(t.loc[t.trial_number.between(15,64),'change_estimate'].mean()),
                   'mean_test_change':float(t.loc[t.trial_number.between(65,94),'change_estimate'].mean())} for name,t in tables.items()}
    results['influence']={'failed':sum(r['status']!='ok' for r in influence),'max_change_shift':max(r.get('max_change_shift',0) for r in influence)}
    dump(out/'comparison-summary.json',results)
    render(summary,tables,out)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    r=fits['phase'].resid
    axes[0].scatter(fits['phase'].fittedvalues,r,s=3,alpha=.2)
    axes[0].axhline(0,color='black',lw=.7); axes[0].set(xlabel='Fitted log bout response',ylabel='Conditional residual')
    probplot(r,dist='norm',plot=axes[1]); fig.savefig(out/'phase-residual-diagnostics.png',dpi=170); plt.close(fig)
    (out/'renderer-script.py').write_bytes(Path(__file__).read_bytes())
    helper=Path('src/classical_conditioning/analysis/inference/learning_onset.py')
    (out/'learning-onset-script.py').write_bytes(helper.read_bytes())
    dump(out/'Fig2_Delay_observed_and_fitted_contrasts.figure.json',{'scientific_status':'exploratory; adequacy unresolved; no freeze',
        'specification':spec,'results':results,'adequacy':reviews,
        'source_sidecar':{'path':str(SOURCE/'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json'),'sha256':sha(SOURCE/'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json')},
        'outputs':[{'path':str(p),'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({'directory':str(out),'results':results,'adequacy':reviews},indent=2),flush=True)


if __name__=='__main__': main()
