"""Explicit empirical-Bayes covariance sensitivity, not wider approved intervals."""
import numpy as np
import pandas as pd
from .panel_g_individual import fit_individual,individual_curve
from .panel_g_review_extensions import credible_envelope
from .panel_g_missingness import scheduled_individual_timing
from .panel_g_review import save_json


def run_hyperparameters(out,config):
    data=pd.read_parquet(out/'model-input.parquet');rows=[];diagnostics=[]
    for method in ['spline','state_space']:
        try:nominal=fit_individual(data,method)
        except ValueError as e:
            diagnostics.append({'method':method,'status':'failed','error':str(e)});continue
        variants=[('nominal',nominal.theta.copy())]
        for i,name in enumerate(['intercept','curve','noise']):
            for factor in [.5,2.]:
                theta=nominal.theta.copy();theta[i]+=np.log(factor)
                variants.append((f'{name}_variance_x{factor:g}',theta))
        for change in [-.1,.1]:
            theta=nominal.theta.copy();theta[3]=np.arctanh(np.clip(np.tanh(theta[3])+change,-.95,.95))
            variants.append((f'AR1_shift_{change:g}',theta))
        baseline={g['key']:individual_curve(nominal,i)[0] for i,g in enumerate(nominal.groups)}
        for label,theta in variants:
            fitted=nominal if label=='nominal' else fit_individual(data,method,fixed_theta=theta)
            rng=np.random.default_rng(config.seed+998)
            for index,group in enumerate(fitted.groups):
                mean,cov=individual_curve(fitted,index);lo,hi,_=credible_envelope(mean,cov,rng)
                observed=np.isin(np.arange(15,65),group['trials'])
                for fraction in config.threshold_fractions:
                    timing=scheduled_individual_timing(np.arange(15,65),mean,lo,hi,observed,
                        -np.log1p(-fraction),config.persistence,pre_count=int((group['trials']<15).sum()))
                    rows.append({'method':method,'variant':label,'condition_id':group['condition'],
                        'fish_id':group['fish_id'],'threshold_fraction':fraction,
                        'maximum_mean_shift':float(abs(mean-baseline[group['key']]).max()),**timing})
            print(f'Individual covariance sensitivity: {method} / {label}',flush=True)
    pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan).to_csv(out/'individual-covariance-sensitivity.csv',index=False)
    save_json(out/'individual-covariance-sensitivity-status.json',{'status':'sensitivity_only',
        'diagnostics':diagnostics,'covariance_uncertainty_integrated':False,
        'intervals_calibrated':False,'interpretation':'Sensitivity does not substitute for calibrated posterior uncertainty.'})
