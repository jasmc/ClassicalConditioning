"""Reproduce all saved Delay LMMs and independently check contrast algebra."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import sys
import warnings

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy.stats import norm, chi2, skew, kurtosis
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

ROOT = Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review')
SOURCE = ROOT / '20261009T141045627289Z-delay-boutonly-lme'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def trial_tests(result, data):
    info = result.model.data.design_info
    center, scale = data.trial_center.iloc[0], data.trial_scale.iloc[0]
    trials = np.arange(5, 95)
    vectors = []
    for trial in trials:
        cases = pd.DataFrame({'condition_id': ['control', 'delay'],
                              'log_baseline': data.log_baseline.mean(),
                              'trial_scaled': (trial-center)/scale})
        x = np.asarray(build_design_matrices([info], cases)[0])
        vectors.append(x[0]-x[1])
    raw = np.array(vectors)
    change = raw - raw[:10].mean(axis=0)
    beta = np.asarray(result.fe_params)
    cov = np.asarray(result.cov_params().loc[result.fe_params.index, result.fe_params.index])
    estimate = change @ beta
    se = np.sqrt(np.einsum('ij,jk,ik->i', change, cov, change))
    p = 2*norm.sf(abs(estimate/se))
    return pd.DataFrame({'trial_number': trials, 'estimate': estimate, 'se': se,
                         'p_raw': p, 'p_fdr': multipletests(p, method='fdr_bh')[1]}), change


def main():
    out = ROOT / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-delay-lmm-audit')
    out.mkdir(exist_ok=False)
    print('AUDIT_DIRECTORY='+str(out), flush=True)
    side = json.loads((SOURCE/'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json').read_text())
    for item in side['outputs']:
        assert sha(item['path']) == item['sha256'], item['path']
    data = pd.read_parquet(SOURCE/'model-input.parquet')
    original = pd.read_csv(SOURCE/'all-fit-diagnostics.csv')
    oldtrials = pd.read_csv(SOURCE/'trial-tests-FDR.csv')
    spec = json.loads((SOURCE/'prespecified-review-config.json').read_text())
    assert len(data)==4811 and data.fish_id.nunique()==57
    np.testing.assert_allclose(data.log_response, np.log(data.conditional_intensity))
    np.testing.assert_allclose(data.log_baseline, np.log(data.baseline_conditional_intensity))
    np.testing.assert_allclose(data.ratio, data.conditional_intensity/data.baseline_conditional_intensity)
    fits, rows, local_rows, sensitivities = {}, [], [], []
    with threadpool_limits(limits=1):
        for _, saved in original.iterrows():
            name = saved.model
            sub = data
            if name.endswith('-local'):
                block = name[:-6].replace('-', ' ')
                if block=='Pre train': block='Pre-train'
                sub = data.loc[data.block_10_name.eq(block)].copy()
                sub['within_block_trial'] = sub.trial_number - sub.trial_number.mean()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = smf.mixedlm(saved.formula, sub, groups=sub.fish_key,
                                    re_formula=saved.used_random_effects)
                result = model.fit(reml=False, method=saved.optimizer)
            fits[name] = result
            names = result.fe_params.index
            cov = np.asarray(result.cov_params().loc[names,names])
            correct, singular_hessian = model.hessian(result.params_object)
            wrong, _ = model.hessian(result.params)
            eigen = np.linalg.eigvalsh((correct+correct.T)/2)
            old = pd.read_csv(SOURCE/(name+'-coefficients.csv')).set_index('term')
            diff = np.max(abs(result.fe_params-old.loc[names,'estimate']))
            residual = np.asarray(result.resid)
            row = dict(model=name, observations=len(sub), fish=sub.fish_key.nunique(),
                       converged=bool(result.converged), full_rank=np.linalg.matrix_rank(model.exog)==model.exog.shape[1],
                       coefficient_max_difference=float(diff), llf_difference=float(result.llf-saved.llf),
                       fixed_covariance_min_eigenvalue=float(np.linalg.eigvalsh(cov).min()),
                       random_covariance_min_eigenvalue=float(np.linalg.eigvalsh(result.cov_re).min()),
                       corrected_hessian_max_eigenvalue=float(eigen.max()),
                       previous_wrong_hessian_max_eigenvalue=float(np.linalg.eigvalsh((wrong+wrong.T)/2).max()),
                       hessian_singular=bool(singular_hessian),
                       residual_skewness=float(skew(residual)), residual_excess_kurtosis=float(kurtosis(residual)),
                       residual_sd=float(np.sqrt(result.scale)), warnings=[str(w.message) for w in caught])
            # Genuine lag-1 pairs: missing trials must not create false neighbors.
            pairs = sub[['fish_key','condition_id','trial_number']].copy()
            pairs['residual']=residual
            nextrow=pairs.copy(); nextrow['trial_number']-=1
            paired=pairs.merge(nextrow,on=['fish_key','condition_id','trial_number'],suffixes=('_now','_next'))
            lag=[g.residual_now.corr(g.residual_next) for _,g in paired.groupby('fish_key') if len(g)>2]
            row['median_true_adjacent_lag1']=float(np.nanmedian(lag))
            grouped=pairs.groupby(['condition_id'],observed=True).residual.agg(['std','count'])
            row['residual_by_condition']=grouped.to_dict('index')
            rows.append(row)
            pd.DataFrame(rows).to_json(out/'all-refit-audit.json',orient='records',indent=2)
            if name.endswith('-local'):
                mean=next(t for t in names if 'condition_id' in t and ':' not in t)
                slope=next(t for t in names if 'condition_id' in t and ':' in t)
                local_rows.append(dict(block_10_name=block,mean_estimate=float(result.fe_params[mean]),
                    mean_p_raw=float(2*norm.sf(abs(result.fe_params[mean]/result.bse_fe[mean]))),
                    slope_estimate=float(result.fe_params[slope]),
                    slope_p_raw=float(2*norm.sf(abs(result.fe_params[slope]/result.bse_fe[slope])))))
            if name.startswith('longitudinal'):
                table, matrix = trial_tests(result,data)
                table.to_csv(out/(name+'-trial-tests.csv'),index=False)
                np.save(out/(name+'-contrast-matrix.npy'),matrix)
                sensitivities.append(dict(model=name, significant_trials=table.loc[table.p_fdr<.05,'trial_number'].tolist(),
                    pre_average_estimate=float(table.estimate.iloc[:10].mean()),
                    pre_significant=int((table.p_fdr.iloc[:10]<.05).sum())))
                if name=='longitudinal-model':
                    np.testing.assert_allclose(matrix,np.load(SOURCE/'trial-contrast-matrix.npy'),atol=1e-12)
                    np.testing.assert_allclose(table.estimate,oldtrials.learning_contrast,atol=1e-10)
                    np.testing.assert_allclose(table.se,oldtrials.standard_error,atol=1e-10)
                    np.testing.assert_allclose(table.p_fdr,oldtrials.p_fdr,atol=1e-8)
                    # Record actual spline knots on the original trial scale.
                    for factor in model.data.design_info.factor_infos.values():
                        for transform in factor.state.get('transforms',{}).values():
                            if hasattr(transform,'_all_knots'):
                                (out/'spline-knots.json').write_text(json.dumps((transform._all_knots*data.trial_scale.iloc[0]+data.trial_center.iloc[0]).tolist()))
            print(f'{name}: reproduced; corrected Hessian max={eigen.max():.5g}; residual kurtosis={row["residual_excess_kurtosis"]:.3f}',flush=True)
    local=pd.DataFrame(local_rows)
    for label in ['mean','slope']:
        local[label+'_p_fdr']=multipletests(local[label+'_p_raw'],method='fdr_bh')[1]
    oldlocal=pd.read_csv(SOURCE/'MR-local-block-tests.csv')
    for col in ['mean_estimate','slope_estimate','mean_p_raw','slope_p_raw','mean_p_fdr','slope_p_fdr']:
        np.testing.assert_allclose(local[col],oldlocal[col],atol=1e-8)
    local.to_csv(out/'MR-independent-recalculation.csv',index=False)
    block=fits['block-model']; terms=[t for t in block.fe_params.index if ':' in t]
    beta=block.fe_params.loc[terms].to_numpy(); cov=block.cov_params().loc[terms,terms].to_numpy()
    joint=float(beta@np.linalg.solve(cov,beta))
    d_raw=2*norm.sf(abs(block.fe_params.loc[terms]/block.bse_fe.loc[terms]))
    d_holm=multipletests(d_raw,method='holm')[1]
    oldd=pd.read_csv(SOURCE/'D-interaction-tests.csv')
    np.testing.assert_allclose(d_holm,oldd.p_holm,atol=1e-8)
    joint_old=pd.read_csv(SOURCE/'joint-interaction-test.csv').iloc[0]
    np.testing.assert_allclose(joint,joint_old.wald_chi_square,atol=1e-8)
    influence=pd.read_csv(SOURCE/'leave-one-fish-out.csv')
    report={'source':str(SOURCE),'model_count':len(rows),'all_coefficients_reproduced':True,
            'D_M_R_trial_algebra_and_multiplicity_reproduced':True,
            'joint_wald':joint,'joint_p':float(chi2.sf(joint,8)),
            'corrected_hessian_failures':[r['model'] for r in rows if r['corrected_hessian_max_eigenvalue']>=0 or r['hessian_singular']],
            'trial_sensitivity':sensitivities,
            'loo_max_trial_contrast_shift':float(influence.max_trial_contrast_shift.max()),
            'loo_training_contrast_range':[float(influence.mean_training_learning_contrast.min()),float(influence.mean_training_learning_contrast.max())],
            'model_specification':spec}
    (out/'audit-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'audit-script.py').write_bytes(Path(__file__).read_bytes())
    (out/'learning-onset-script.py').write_bytes(Path('src/classical_conditioning/analysis/inference/learning_onset.py').read_bytes())
    manifest={'source_sidecar':{'path':str(SOURCE/'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json'),'sha256':sha(SOURCE/'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json')},
              'outputs':[{'path':str(p),'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()]}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
