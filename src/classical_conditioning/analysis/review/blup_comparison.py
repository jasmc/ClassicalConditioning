"""Auditable two-epoch Gaussian EBLUPs, REML and refitted prediction errors.

Epoch-specific random means are an invertible reparameterisation of a random
intercept and epoch slope. Fixed condition-by-epoch means are fitted jointly.
AR1 residuals reset between epochs and use original scheduled trial gaps.
This estimates conditional bout-vigor expression, not biological learning.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class EpochData:
    fish: tuple[str, ...]
    conditions: np.ndarray
    y: np.ndarray
    valid: np.ndarray
    times: np.ndarray
    epochs: tuple[tuple[int, int], tuple[int, int]]

    @property
    def counts(self):
        return self.valid.sum(axis=2)

    def subset(self, keep):
        keep=np.asarray(keep)
        return replace(self, fish=tuple(np.asarray(self.fish)[keep]),
            conditions=self.conditions[keep],y=self.y[keep],valid=self.valid[keep],times=self.times[keep])


def epoch_data(trials, inventory, epochs=((5,14),(40,64)), outcome='g'):
    keys=['condition_id','fish_id','trial_number']
    if trials.duplicated(keys).any():raise ValueError('Duplicate scheduled fish/trial')
    if inventory.duplicated(['condition_id','fish_id']).any():raise ValueError('Duplicate fish')
    n=len(inventory);width=max(b-a+1 for a,b in epochs)
    y=np.zeros((n,2,width));valid=np.zeros_like(y,dtype=bool);times=np.zeros_like(y,dtype=int)
    reasons=[]
    for i,row in enumerate(inventory.itertuples(index=False)):
        d=trials.loc[trials.fish_id.eq(row.fish_id)&trials.condition_id.eq(row.condition_id)].sort_values('trial_number')
        if len(d)!=90 or set(d.trial_number)!=set(range(5,95)):raise ValueError('Scheduled inventory not reconciled')
        if outcome=='g':values=d.response_logmedian-d.baseline_logmedian
        elif outcome=='legacy':
            values=np.log(d.response_bout_mean.where(d.response_bout_mean.gt(0))/d.baseline_bout_mean.where(d.baseline_bout_mean.gt(0)))
        else:raise ValueError(outcome)
        for j,(a,b) in enumerate(epochs):
            mask=d.trial_number.between(a,b)&np.isfinite(values)
            count=int(mask.sum());y[i,j,:count]=values.loc[mask];times[i,j,:count]=d.loc[mask,'trial_number'];valid[i,j,:count]=True
        if not valid[i,0].any() or not valid[i,1].any():reasons.append(row.fish_id)
    if reasons:raise ValueError('Retain inventory but mark unestimable before fitting: '+','.join(reasons))
    return EpochData(tuple(inventory.fish_id),inventory.condition_id.eq('delay').to_numpy(int),y,valid,times,tuple(epochs))


def parameters(theta, ar1=True):
    sd=np.exp(theta[:2]);r=np.tanh(theta[2]);G=np.outer(sd,sd)*np.array([[1.,r],[r,1.]])
    sigma=np.exp(theta[3:5]);phi=float(np.tanh(theta[5])) if ar1 else 0.
    return G,sigma,phi


def residual_summaries(data, sigma, phi):
    """AR1 precision sufficient statistics; exact for irregular observed gaps."""
    pair=data.valid[:,:,1:];gaps=data.times[:,:,1:]-data.times[:,:,:-1]
    a=np.where(pair,np.power(phi,np.maximum(gaps,1)),0.)
    denom=1-a*a;innov=data.y[:,:,1:]-a*data.y[:,:,:-1]
    scale=sigma[data.conditions,None]**2
    s=(1+np.sum(np.where(pair,(1-a)**2/denom,0),axis=2))/scale
    t=(data.y[:,:,0]+np.sum(np.where(pair,(1-a)*innov/denom,0),axis=2))/scale
    q=(data.y[:,:,0]**2+np.sum(np.where(pair,innov**2/denom,0),axis=2))/scale
    ld=np.sum(np.where(pair,np.log(denom),0),axis=(1,2))+2*data.counts.sum(axis=1)*np.log(sigma[data.conditions])
    return s,t,q.sum(axis=1),ld


def evaluate(data, theta, ar1=True, detail=False):
    G,sigma,phi=parameters(theta,ar1);s,t,q,ld=residual_summaries(data,sigma,phi)
    # H = (G^-1 + Z'R^-1 Z)^-1 without inverting nearly singular G.
    H=np.linalg.solve(np.eye(2)[None,:,:]+G[None,:,:]*s[:,None,:],np.broadcast_to(G,(len(data.fish),2,2)))
    T=np.array([np.diag(v) for v in s])-s[:,:,None]*H*s[:,None,:]
    u=t-s*np.einsum('nij,nj->ni',H,t)
    A=np.zeros((4,4));rhs=np.zeros(4)
    for c in (0,1):
        idx=data.conditions==c;A[c*2:c*2+2,c*2:c*2+2]=T[idx].sum(axis=0);rhs[c*2:c*2+2]=u[idx].sum(axis=0)
    beta=np.linalg.solve(A,rhs);covbeta=np.linalg.inv(A)
    logdet=np.linalg.slogdet(np.eye(2)[None,:,:]+G[None,:,:]*s[:,None,:])[1]
    quad=np.sum(q-np.einsum('ni,nij,nj->n',t,H,t))-rhs@beta
    nobs=int(data.valid.sum());nll=.5*(np.sum(ld+logdet)+np.linalg.slogdet(A)[1]+quad+(nobs-4)*np.log(2*np.pi))
    if not detail:return float(nll)
    means=beta.reshape(2,2)[data.conditions];random=np.einsum('nij,nj->ni',H,t-s*means)
    c=np.array([1.,-1.]);total=(means+random)@c;control=float(beta[:2]@c);excess=total-control
    se=[];excess_se=[];latent=means+random
    for i in range(len(data.fish)):
        D=np.eye(2)-H[i]*s[i][None,:];K=np.zeros((2,4));K[:,data.conditions[i]*2:data.conditions[i]*2+2]=np.eye(2)
        v=c@D@K;vref=v-np.r_[c,[0.,0.]]
        se.append(np.sqrt(max(0.,c@H[i]@c+v@covbeta@v)))
        excess_se.append(np.sqrt(max(0.,c@H[i]@c+vref@covbeta@vref)))
    return dict(nll=float(nll),theta=theta,G=G,sigma=sigma,phi=phi,beta=beta,covbeta=covbeta,
        latent=latent,random=random,total=total,control=control,excess=excess,se=np.array(se),
        excess_se=np.array(excess_se),H=H,nobs=nobs)


def fit(data, ar1=True, start=None, multistart=False):
    scale=max(.03,float(np.std(data.y[data.valid])))
    initial=np.array([np.log(scale*.6)]*2+[0.,np.log(scale*.7),np.log(scale*.7),.15])
    bounds=[(-9.,1.)]*2+[(-3.,3.)]+[(-8.,1.)]*2+[(-1.8,1.8)]
    if not ar1:bounds[-1]=(0.,0.);initial[-1]=0.
    seeds=[initial if start is None else np.array(start).copy()]
    if multistart:seeds += [initial,initial+np.array([.4,-.3,.5,0,0,.35])]
    runs=[]
    for seed in seeds:
        result=minimize(lambda th:evaluate(data,th,ar1),seed,method='L-BFGS-B',bounds=bounds,
            options={'maxiter':350,'ftol':1e-9,'gtol':2e-5,'maxls':30})
        if np.isfinite(result.fun):runs.append(result)
    good=[r for r in runs if r.success]
    if not good:raise RuntimeError('REML optimisation failed: '+str([r.message for r in runs]))
    best=min(good,key=lambda r:r.fun);out=evaluate(data,best.x,ar1,True)
    out.update(converged=True,optimizer_message=str(best.message),iterations=int(best.nit),
        boundary=bool(any(abs(best.x[i]-lo)<.02 or abs(best.x[i]-hi)<.02 for i,(lo,hi) in enumerate(bounds) if lo!=hi)),ar1=ar1)
    return out


def simulate(data, model, rng, scenario='matched'):
    G=model['G'];beta=model['beta'].reshape(2,2).copy();sigma=model['sigma'];phi=model['phi']
    b=rng.multivariate_normal(np.zeros(2),G,size=len(data.fish));means=beta[data.conditions]+b
    if scenario=='null':
        beta[1]=beta[0];beta[:,1]=beta[:,0];base=rng.normal(0,np.sqrt(np.mean(np.diag(G))),len(data.fish));b=np.column_stack([base,base]);means=beta[data.conditions]+b
    y=np.zeros_like(data.y)
    for i in range(len(data.fish)):
        for epoch in (0,1):
            count=int(data.counts[i,epoch]);times=data.times[i,epoch,:count]
            innovations=rng.standard_t(5,count)/np.sqrt(5/3) if scenario=='heavy_tail' else rng.normal(size=count)
            errors=np.empty(count);errors[0]=innovations[0]*sigma[data.conditions[i]]
            for j in range(1,count):
                a=phi**int(times[j]-times[j-1]);errors[j]=a*errors[j-1]+sigma[data.conditions[i]]*np.sqrt(1-a*a)*innovations[j]
            trend=np.zeros(count)
            if scenario=='within_epoch_trend':
                a,z=data.epochs[epoch];trend=.15*(times-(a+z)/2)/max(z-a,1)
            y[i,epoch,:count]=means[i,epoch]+errors+trend
    truth=means@np.array([1.,-1.]);excess=truth-float(beta[0]@np.array([1.,-1.]))
    return replace(data,y=y),truth,excess


def bootstrap(data, model, repetitions, seed, scenario='matched', progress=None):
    rng=np.random.default_rng(seed);draws=[];failures=[]
    for rep in range(repetitions):
        simulated,truth,excess=simulate(data,model,rng,scenario)
        try:
            fitted=fit(simulated,ar1=model['ar1'],start=model['theta'])
            draws.append({'total_error':(fitted['total']-truth)/fitted['se'],
                'excess_error':(fitted['excess']-excess)/fitted['excess_se'],
                'boundary':fitted['boundary']})
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as error:failures.append({'replicate':rep,'reason':str(error)})
        if progress and (rep+1)%25==0:progress(rep+1,repetitions)
    if not draws:raise RuntimeError('All refits failed')
    return {'total_error':np.array([d['total_error'] for d in draws]),'excess_error':np.array([d['excess_error'] for d in draws]),
        'boundary':np.array([d['boundary'] for d in draws]),'failures':failures,'attempted':repetitions,'scenario':scenario}


def interval_quantiles(draws):
    return {'total':np.quantile(draws['total_error'],[.025,.975],axis=0),
        'excess':np.quantile(draws['excess_error'],[.025,.975],axis=0),
        'family_lower':float(np.quantile(np.max(draws['excess_error'],axis=1),.95))}


def scored_fish(data, model, quantiles=None):
    rows=[]
    for i,fish in enumerate(data.fish):
        row={'fish_id':fish,'condition_id':['control','delay'][data.conditions[i]],'epoch0_n':int(data.counts[i,0]),
            'epoch1_n':int(data.counts[i,1]),'acquisition':float(model['total'][i]),'control_reference':model['control'],
            'excess_acquisition':float(model['excess'][i]),'fixed_acquisition':float(model['beta'].reshape(2,2)[data.conditions[i]]@np.array([1.,-1.])),
            'random_acquisition':float(model['random'][i]@np.array([1.,-1.])),
            'plugin_se':float(model['se'][i]),'excess_plugin_se':float(model['excess_se'][i]),
            'epoch0_fitted':float(model['latent'][i,0]),'epoch1_fitted':float(model['latent'][i,1]),
            'raw_mean_contrast':float(np.mean(data.y[i,0,data.valid[i,0]])-np.mean(data.y[i,1,data.valid[i,1]])),
            'raw_median_contrast':float(np.median(data.y[i,0,data.valid[i,0]])-np.median(data.y[i,1,data.valid[i,1]]))}
        if quantiles is not None:
            for name,estimate,se in [('total',model['total'][i],model['se'][i]),('excess',model['excess'][i],model['excess_se'][i])]:
                lo,hi=quantiles[name][:,i];row[name+'_lower']=float(estimate-hi*se);row[name+'_upper']=float(estimate-lo*se)
            row['excess_family_lower']=float(model['excess'][i]-quantiles['family_lower']*model['excess_se'][i])
            row['positive_model_evidence']=bool(row['excess_family_lower']>0 and row['acquisition']>0)
        rows.append(row)
    return pd.DataFrame(rows)


def coverage(draws, quantiles):
    rows={}
    for name in ['total','excess']:
        e=draws[name+'_error'];q=quantiles[name]
        covered=(e>=q[0])&(e<=q[1]);per_run=covered.mean(axis=1)
        rows[name+'_pointwise_coverage']=float(covered.mean())
        rows[name+'_coverage_mcse']=float(per_run.std(ddof=1)/np.sqrt(len(per_run)))
    rows['family_lower_coverage']=float((np.max(draws['excess_error'],axis=1)<=quantiles['family_lower']).mean())
    rows['successful_refits']=len(draws['excess_error']);rows['attempted']=draws['attempted'];rows['boundary_fraction']=float(draws['boundary'].mean())
    return rows
