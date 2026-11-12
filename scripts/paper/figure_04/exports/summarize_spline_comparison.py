"""Score whole-fish held-out predictions and inspect a bounded spline comparison.

This is an exploratory comparison conditional on saved retrospective labels.
No best-model declaration, selected-figure publication or confidence calibration.
"""
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import json
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from classical_conditioning.figure_workspace import Snapshot,sha
from classical_conditioning.analysis.review.figure4_profiles import GROUPS
CANDIDATES=['LMM6','LMM12','LMM20','GAMM20','GAMM30','GAMM20_AR1']

def fish_scores(frame):
    d=frame.copy()
    d['squared_error']=(d.signed-d.predicted)**2
    d['absolute_error']=(d.signed-d.predicted).abs()
    return d.groupby(['fish_id','display_group','fold'],observed=True).agg(
        mse=('squared_error','mean'),mae=('absolute_error','mean'),observed_bins=('signed','size')).reset_index()

def adjacent_correlation(frame,column):
    left=frame[['fish_id','time_s',column]].copy()
    right=left.copy();right['time_s']+=.5
    pair=left.merge(right,on=['fish_id','time_s'],suffixes=('','_previous'),validate='one_to_one')
    return float(pair[column].corr(pair[column+'_previous'])) if len(pair)>2 else None

def summarize(directory):
    d=Path(directory);spec=json.loads((d/'comparison-specification.json').read_text())
    guard=Snapshot([d/'execution-start.json',Path(__file__)])
    guard.bind_hashes(json.loads((d/'execution-start.json').read_text())['hashes']);guard.check()
    guard.include(sorted(p for p in (d/'fits').iterdir()if p.is_file()))
    status=pd.concat([pd.read_csv(p)for p in sorted((d/'fits').glob('*-status.csv'))],ignore_index=True)
    assert len(status)==2*len(CANDIDATES)*6
    scores=[];metrics=[];checks={};stability=[]
    for assay in ['delay','3strace']:
        reference=np.load(Path(spec['sources'][assay]['summary']).parent/'model-arrays.npz')
        master=pd.read_csv(d/(assay+'-model-data.csv'))
        for candidate in CANDIDATES:
            rows=status.loc[status.assay.eq(assay)&status.candidate.eq(candidate)]
            if not rows.success.all():
                metric={'assay':assay,'candidate':candidate,'status':'incomplete; not ranked','failed_fits':int((~rows.success).sum())}
                full=d/'fits'/f'{assay}-{candidate}-0-residuals.csv'
                if full.exists():
                    residuals=pd.read_csv(full)
                    metric.update(full_fit_success=True,
                        raw_residual_excess_kurtosis=float(kurtosis(residuals.residual)),
                        raw_adjacent_residual_correlation=adjacent_correlation(residuals,'residual'),
                        normalized_adjacent_residual_correlation=adjacent_correlation(residuals,'normalized'))
                metrics.append(metric)
                continue
            parts=[pd.read_csv(d/'fits'/f'{assay}-{candidate}-{fold}-predictions.csv')for fold in range(1,6)]
            pred=pd.concat(parts,ignore_index=True)
            assert not pred.duplicated(['fish_id','time_s']).any()
            cols=['fish_id','time_s','fold','display_group','signed']
            pd.testing.assert_frame_equal(pred[cols].sort_values(cols[:2]).reset_index(drop=True),
                master[cols].sort_values(cols[:2]).reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-12)
            fs=fish_scores(pred);fs['assay']=assay;fs['candidate']=candidate;scores.append(fs)
            resid=pd.read_csv(d/'fits'/f'{assay}-{candidate}-0-residuals.csv')
            normalized=resid.normalized
            fullcurve=pd.read_csv(d/'fits'/f'{assay}-{candidate}-0-curves.csv')
            assert len(fullcurve)==240 and fullcurve.fitted.notna().all()
            if candidate=='LMM6':
                actual=np.asarray([fullcurve.loc[fullcurve.display_group.eq(g)].sort_values('time_s').fitted.to_numpy()for g in GROUPS])
                difference=float(np.max(np.abs(actual-reference['group_curves'])))
                checks[assay]={'independent_R_vs_Python_reference_max_abs_difference':difference,
                    'same_validation_rows_all_candidates':True,'whole_fish_folds':True}
                assert difference<.0002,checks[assay]
            metric={'assay':assay,'candidate':candidate,'status':'success','failed_fits':0,
                'cv_equal_fish_rmse':float(np.sqrt(fs.mse.mean())),
                'cv_equal_fish_mae':float(fs.mae.mean()),
                'cv_group_balanced_rmse':float(np.sqrt(fs.groupby('display_group').mse.mean().mean())),
                'raw_residual_excess_kurtosis':float(kurtosis(resid.residual)),
                'raw_adjacent_residual_correlation':adjacent_correlation(resid,'residual'),
                'normalized_adjacent_residual_correlation':adjacent_correlation(resid,'normalized'),
                'apvar_ok_all_fits':bool(rows.apvar_ok.all()),
                'warning_fits':int(rows.warning.notna().sum())}
            for g in GROUPS:
                metric['cv_rmse_'+g]=float(np.sqrt(fs.loc[fs.display_group.eq(g),'mse'].mean()))
            # The pre-US interval was fixed in the source assay, not selected
            # according to whichever candidate displays a favorable extremum.
            us=9 if assay=='delay' else 13
            for region,mask in [('baseline',pred.time_s.lt(0)),('pre_expected_us',pred.time_s.ge(0)&pred.time_s.lt(us)),('post_expected_us',pred.time_s.ge(us))]:
                region_scores=fish_scores(pred.loc[mask]);metric['cv_rmse_'+region]=float(np.sqrt(region_scores.mse.mean()))
                for g in GROUPS:
                    metric['cv_rmse_'+region+'_'+g]=float(np.sqrt(region_scores.loc[region_scores.display_group.eq(g),'mse'].mean()))
            metric['group_residual_sd']=resid.groupby('display_group').residual.std().to_dict()
            edf=d/'fits'/f'{assay}-{candidate}-0-edf.csv'
            if edf.exists():metric['group_smooth_edf']=pd.read_csv(edf).set_index('term').edf.to_dict()
            rho=d/'fits'/f'{assay}-{candidate}-0-rho.csv'
            if rho.exists():metric['estimated_ar1_rho']=float(pd.read_csv(rho).rho.iloc[0])
            metrics.append(metric)
        for g in GROUPS:
            base=pd.read_csv(d/'fits'/f'{assay}-GAMM20-0-curves.csv')
            other=pd.read_csv(d/'fits'/f'{assay}-GAMM30-0-curves.csv')
            joined=base.merge(other,on=['display_group','time_s'],suffixes=('_20','_30'))
            joined=joined.loc[joined.display_group.eq(g)]
            stability.append({'assay':assay,'group':g,'GAMM20_vs_30_max_absolute_curve_difference':float((joined.fitted_20-joined.fitted_30).abs().max())})
    cv=pd.concat(scores,ignore_index=True);cv.to_csv(d/'fish-cv-scores.csv',index=False)
    flat=[{k:v for k,v in m.items()if not isinstance(v,dict)}for m in metrics]
    pd.DataFrame(flat).to_csv(d/'comparison-metrics.csv',index=False)
    paired=[]
    for assay in ['delay','3strace']:
        a=cv.loc[cv.assay.eq(assay)];b=a.loc[a.candidate.eq('LMM6')]
        for candidate in CANDIDATES[1:]:
            if not a.candidate.eq(candidate).any():
                paired.append({'assay':assay,'candidate':candidate,'status':'Not ranked: incomplete successful CV fits.'})
                continue
            joined=a.loc[a.candidate.eq(candidate)].merge(b,on=['fish_id','display_group','fold'],suffixes=('_alternative','_reference'),validate='one_to_one')
            delta=joined.mse_alternative-joined.mse_reference
            paired.append({'assay':assay,'candidate':candidate,'paired_mean_MSE_change':float(delta.mean()),
                'fish_fraction_lower_MSE':float((delta<0).mean()),
                'fold_mean_MSE_changes':joined.assign(delta=delta).groupby('fold').delta.mean().tolist()})
    record={'status':'Exploratory comparison; no model adopted','metrics':metrics,'paired_prediction_losses':paired,
        'basis_stability':stability,'validation_checks':checks,
        'limitations':['Retrospective CV conditional on saved same-data group labels; no independent learner validation.',
            'All candidates use Gaussian conditional intensity; missing bout intensity is not reconstructed movement.',
            'One five-fold split; predictive loss differences have not been calibrated as significance tests.',
            'Fish holdouts assess this cohort conditional on its saved labels; experiment/day/tank transfer and dependence have not been validated.',
            'A non-positive variance-parameter covariance estimate is flagged separately from optimizer failure; these fits supply exploratory point predictions, not validated uncertainty.',
            'No full-refit alternative-model bootstrap, simultaneous curve coverage or peak-timing calibration completed.'],
        'source_hashes':guard.hashes}
    with (d/'comparison-summary.json').open('x')as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    status.to_csv(d/'fit-status.csv',index=False);guard.check()
    print(pd.DataFrame(flat)[['assay','candidate','cv_equal_fish_rmse','cv_group_balanced_rmse','normalized_adjacent_residual_correlation','apvar_ok_all_fits']].to_string(index=False))
    print(json.dumps({'validation':checks,'basis_stability':stability}))

if __name__=='__main__':summarize(sys.argv[1])
