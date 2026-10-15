"""Training-only empirical-Bayes individual-expression candidates.

Gaussian spline/random-walk covariance models are compared explicitly. These
are conditional-on-hyperparameters exploratory intervals, not approved fish
learning labels or individual causal effects. Testing never enters fitting.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .panel_g_review import phase_features

TRAIN_FORMULA = (
    "C(condition_id, Treatment(reference='control')) * "
    "(C(fit_phase, Treatment(reference='Pre')) + pre_trial + "
    "train_b0 + train_b1 + train_b2 + train_b3)"
)


def coordinates(trials, method):
    features=phase_features().set_index("trial_number").loc[trials]
    if method=="state_space":
        k=np.maximum(np.asarray(trials)-14,0)
        return np.minimum(k[:,None],k[None,:]).astype(float)
    if method=="spline":
        b=features[[f"train_b{i}" for i in range(4)]].to_numpy()
        return b@b.T
    raise ValueError(method)


def cross_coordinates(left,right,method):
    if method=="state_space":
        return np.minimum(np.maximum(np.asarray(left)-14,0)[:,None],
                          np.maximum(np.asarray(right)-14,0)[None,:]).astype(float)
    features=phase_features().set_index("trial_number")
    columns=[f"train_b{i}" for i in range(4)]
    return features.loc[left,columns].to_numpy()@features.loc[right,columns].to_numpy().T


@dataclass
class IndividualModel:
    method: str
    data: pd.DataFrame
    design_info: object
    groups: list
    theta: np.ndarray
    beta: np.ndarray
    beta_cov: np.ndarray
    optimizer: object
    factors: list

    def cases(self, condition):
        f=phase_features().iloc[:60].copy()
        f["fit_phase"]=f.fit_phase.cat.remove_unused_categories()
        f["condition_id"]=condition
        return np.asarray(build_design_matrices([self.design_info],f)[0])


def covariance(group,theta):
    intercept,wiggle,noise=np.exp(theta[:3])
    rho=np.tanh(theta[3])
    return intercept+wiggle*group["kernel"]+noise*rho**group["distance"]


def fit_individual(data,method,*,start=None,fixed_theta=None):
    # Explicit filtering before any response/hyperparameter computation.
    d=data.loc[data.trial_number.between(5,64)].copy().sort_values(["fish_key","trial_number"])
    d["fit_phase"]=d.fit_phase.cat.remove_unused_categories()
    x=dmatrix(TRAIN_FORMULA,d,return_type="dataframe")
    if np.linalg.matrix_rank(x)!=x.shape[1]:raise ValueError("Training-only fixed design is deficient")
    groups=[]
    for key,indices in d.groupby("fish_key",observed=True).indices.items():
        frame=d.iloc[indices];trials=frame.trial_number.to_numpy(int)
        groups.append({"key":key,"condition":str(frame.condition_id.iloc[0]),
            "fish_id":str(frame.fish_id.iloc[0]),"trials":trials,"x":x.iloc[indices].to_numpy(),
            "y":frame.z.to_numpy(),"kernel":coordinates(trials,method),
            "distance":abs(trials[:,None]-trials[None,:])})
    n,p=len(d),x.shape[1]
    def solve(theta,details=False):
        precision=np.zeros((p,p));rhs=np.zeros(p);quad=0.;logdet=0.;factors=[]
        for group in groups:
            factor=cho_factor(covariance(group,theta),lower=True,check_finite=False)
            vx=cho_solve(factor,group["x"],check_finite=False)
            vy=cho_solve(factor,group["y"],check_finite=False)
            precision+=group["x"].T@vx;rhs+=group["x"].T@vy
            quad+=group["y"]@vy;logdet+=2*np.log(np.diag(factor[0])).sum()
            factors.append(factor)
        factor=cho_factor(precision,lower=True,check_finite=False)
        beta=cho_solve(factor,rhs,check_finite=False)
        value=.5*(logdet+2*np.log(np.diag(factor[0])).sum()+quad-rhs@beta+(n-p)*np.log(2*np.pi))
        if details:return beta,cho_solve(factor,np.eye(p)),factors
        return value
    initial=np.array([np.log(.08**2),np.log(.01**2 if method=="state_space" else .08**2),np.log(.12**2),np.arctanh(.1)]) if start is None else start
    if fixed_theta is None:
        result=minimize(solve,initial,method="L-BFGS-B",bounds=[(-16,0)]*3+[(-1,1)],options={"maxiter":200,"ftol":1e-9})
    else:
        from scipy.optimize import OptimizeResult
        result=OptimizeResult(x=np.asarray(fixed_theta),success=True,fun=solve(fixed_theta),nit=0,
                              message="fixed covariance parameter sensitivity; not fitted")
    if not result.success:raise ValueError(f"{method} optimizer failed: {result.message}")
    beta,cov,factors=solve(result.x,details=True)
    return IndividualModel(method,d,x.design_info,groups,result.x,beta,cov,result,factors)


def individual_curve(model,index):
    group=model.groups[index];trials=np.arange(5,65)
    intercept,wiggle,noise=np.exp(model.theta[:3])
    own=model.cases(group["condition"]);control=model.cases("control")
    anchor=np.eye(60)-np.ones((60,1))@np.r_[np.repeat(.1,10),np.zeros(50)][None,:]
    g=anchor@(control-own)
    cross=anchor@(intercept+wiggle*cross_coordinates(trials,group["trials"],model.method))
    prior=anchor@(intercept+wiggle*coordinates(trials,model.method))@anchor.T
    solved=cho_solve(model.factors[index],cross.T,check_finite=False)
    residual=group["y"]-group["x"]@model.beta
    mean=g@model.beta-cross@cho_solve(model.factors[index],residual,check_finite=False)
    # Integrate fixed-effect uncertainty at estimated covariance parameters.
    coefficient=g+cross@cho_solve(model.factors[index],group["x"],check_finite=False)
    cov=prior-cross@solved+coefficient@model.beta_cov@coefficient.T
    cov=(cov+cov.T)/2
    return mean[10:],cov[10:,10:]


def persistence_forecast(model,index,heldout_trials):
    """Training-end persistence forecast, not an extinction model.

    Shared latent state/mean is held at trial 64. AR noise evolves over actual
    held-out trial distances. No held-out response is used in the forecast.
    """
    group=model.groups[index];left=np.repeat(64,len(heldout_trials))
    intercept,wiggle,noise=np.exp(model.theta[:3]);rho=np.tanh(model.theta[3])
    cross=intercept+wiggle*cross_coordinates(left,group["trials"],model.method)+noise*rho**abs(np.asarray(heldout_trials)[:,None]-group["trials"][None,:])
    own=model.cases(group["condition"])[-1:]
    x=np.repeat(own,len(left),axis=0)
    mean=x@model.beta+cross@cho_solve(model.factors[index],group["y"]-group["x"]@model.beta,check_finite=False)
    prior=intercept+wiggle*cross_coordinates(left,left,model.method)+noise*rho**abs(np.asarray(heldout_trials)[:,None]-np.asarray(heldout_trials)[None,:])
    coefficient=x-cross@cho_solve(model.factors[index],group["x"],check_finite=False)
    cov=prior-cross@cho_solve(model.factors[index],cross.T,check_finite=False)+coefficient@model.beta_cov@coefficient.T
    return float(mean.mean()),float(max(cov.sum()/len(left)**2,0))
