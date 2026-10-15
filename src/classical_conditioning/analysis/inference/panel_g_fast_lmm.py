"""The same Gaussian ML random-intercept/slope model using sufficient statistics.

Woodbury identities avoid repeated frame-sized covariance calculations. Fixed
effect covariance uses the full observed likelihood information, including
covariance nuisance parameters, through its Schur complement. This backend
must pass known-answer comparisons before use; it does not change the model.
"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize


def prepare_fast(data,formula,group='fish_key',random='1 + trial_scaled'):
    d=data.copy(deep=False);d.attrs={}
    response,design=dmatrices(formula,d,return_type='dataframe')
    x=design.to_numpy();y=response.to_numpy().ravel();n,p=x.shape
    if n!=len(d) or np.linalg.matrix_rank(x)!=p:raise ValueError('Missing data or deficient fixed design')
    if random not in ['1','1 + trial_scaled']:raise ValueError('Unsupported random structure')
    z=np.ones((n,1)) if random=='1' else np.column_stack([np.ones(n),d.trial_scaled.to_numpy()])
    mapping=d.groupby(group,observed=True,sort=False).indices
    keys=list(mapping);indices=list(mapping.values());groups=[]
    for positions in indices:
        a=x[positions];b=z[positions];c=y[positions]
        groups.append((len(c),a.T@a,a.T@b,b.T@b,a.T@c,b.T@c,c@c))
    stats=[np.array([g[i] for g in groups]) for i in range(7)]
    return SimpleNamespace(x=x,y=y,z=z,p=p,indices=indices,keys=keys,
        conditions=np.array([str(d.condition_id.iloc[i[0]]) for i in indices]),
        stats=stats,names=design.columns,design_info=design.design_info,formula=formula,random=random)


def fast_fit(data,formula,group='fish_key',random='1 + trial_scaled',optimizer='lbfgs',prepared=None,weights=None):
    attempts={'formula':formula,'requested_random_effects':random,'used_random_effects':random,
        'optimizer':optimizer,'backend':'Gaussian ML sufficient statistics',
        'diagnostic_status':'failed','converged':False,'singular':False,'fallback_used':False}
    try:
        cached=prepare_fast(data,formula,group,random) if prepared is None else prepared
        if cached.formula!=formula or cached.random!=random:raise ValueError('Prepared model mismatch')
        x,y,z,p,indices=cached.x,cached.y,cached.z,cached.p,cached.indices
        q=z.shape[1];ns,xx,xz,zz,xy,zy,yy=cached.stats
        w=np.ones(len(ns)) if weights is None else np.asarray(weights,float)
        if w.shape!=ns.shape or np.any(w<0) or not np.all(w==w.astype(int)):raise ValueError('Invalid cluster weights')
        n=int(w@ns)
        eye=np.eye(q)[None,:,:]
        def covariance(theta):
            if q==1:return np.array([[np.exp(theta[0])]]),np.exp(theta[1]),[np.array([[np.exp(theta[0])]])]
            a,b=np.exp(theta[:2]);rho=np.tanh(theta[2]);cross=rho*np.sqrt(a*b)
            sigma=np.array([[a,cross],[cross,b]])
            derivatives=[np.array([[a,cross/2],[cross/2,0]]),
                         np.array([[0,cross/2],[cross/2,b]]),
                         np.array([[0,np.sqrt(a*b)*(1-rho*rho)],[np.sqrt(a*b)*(1-rho*rho),0]])]
            return sigma,np.exp(theta[3]),derivatives
        def solve(theta):
            sigma,noise,derivatives=covariance(theta)
            system=noise*eye+zz@sigma
            k=np.broadcast_to(sigma,system.shape)@np.linalg.inv(system)
            k=(k+k.transpose(0,2,1))/2
            precision=np.einsum('i,ipq->pq',w,(xx-xz@k@xz.transpose(0,2,1))/noise)
            rhs=np.einsum('i,ip->p',w,(xy-np.einsum('ipq,iqr,ir->ip',xz,k,zy))/noise)
            factor=cho_factor(precision,lower=True,check_finite=False)
            beta=cho_solve(factor,rhs,check_finite=False)
            conditional=cho_solve(factor,np.eye(p),check_finite=False)
            sign,logdet=np.linalg.slogdet(eye+zz@sigma/noise)
            if np.any(sign<=0):raise ValueError('Invalid covariance determinant')
            quadratic=w@((yy-np.einsum('iq,iqr,ir->i',zy,k,zy))/noise)-rhs@beta
            value=.5*(n*np.log(2*np.pi)+w@(ns*np.log(noise)+logdet)+quadratic)
            zr=zy-np.einsum('ipq,p->iq',xz,beta)
            wr=(zr-np.einsum('iqr,irs,is->iq',zz,k,zr))/noise
            wz=(zz-zz@k@zz)/noise
            gradient=[.5*(np.einsum('i,iqr,rq->',w,wz,derivative)-np.einsum('i,iq,qr,ir->',w,wr,derivative,wr))
                      for derivative in derivatives]
            rr=yy-2*xy@beta+np.einsum('p,ipq,q->i',beta,xx,beta)
            kr=np.einsum('iqr,ir->iq',k,zr)
            residual_square=(rr-2*np.einsum('iq,iq->i',zr,kr)+np.einsum('iq,iqr,ir->i',kr,zz,kr))/noise**2
            trace=(ns-np.einsum('iqr,irq->i',k,zz))/noise
            gradient.append(.5*noise*(w@(trace-residual_square)))
            return value,np.array(gradient),beta,conditional,precision,rhs,k
        ols=cho_solve(cho_factor(np.einsum('i,ipq->pq',w,xx),lower=True),w@xy)
        variance=max(float(w@(yy-2*xy@ols+np.einsum('p,ipq,q->i',ols,xx,ols)))/n,1e-6)
        initial=np.log([variance/4,variance]) if q==1 else np.array([np.log(variance/4),np.log(variance/20),0.,np.log(variance)])
        bounds=[(-24,2),(-24,2)] if q==1 else [(-24,2),(-24,2),(-6,6),(-24,2)]
        if optimizer=='lbfgs':
            result=minimize(lambda t:solve(t)[:2],initial,jac=True,method='L-BFGS-B',bounds=bounds,
                options={'maxiter':400,'ftol':1e-12,'gtol':1e-6})
        else:
            result=minimize(lambda t:solve(t)[0],initial,method='Powell',bounds=bounds,
                options={'maxiter':400,'ftol':1e-10,'xtol':1e-7})
        value,gradient,beta,conditional,precision,rhs,k=solve(result.x)
        sigma,noise,_=covariance(result.x);eigen=np.linalg.eigvalsh(sigma)
        boundary=bool(q==2 and abs(result.x[2])>=5.99)
        singular=eigen.min()<=1e-8 or eigen.min()<=eigen.max()*1e-6 or boundary
        attempts.update(converged=bool(result.success),singular=bool(singular),
            random_effect_min_eigenvalue=float(eigen.min()),random_effect_max_eigenvalue=float(eigen.max()),
            random_effect_covariance_json=__import__('json').dumps(sigma.tolist()),
            residual_variance=float(noise),gradient_maximum=float(abs(gradient).max()),
            observation_count=n,fish_count=int(w.sum()),llf=float(-value),correlation_boundary=boundary,
            aic=float(2*(p+q*(q+1)//2+1)+2*value),bic=float(np.log(n)*(p+q*(q+1)//2+1)+2*value),
            fixed_design_rank=p,fixed_design_columns=p,rank_deficient=False)
        if not result.success or singular:raise ValueError('Nonconverged or singular covariance')
        step=1e-4;hessian=np.zeros((len(result.x),len(result.x)));cross=np.zeros((p,len(result.x)))
        for i in range(len(result.x)):
            plus=result.x.copy();minus=result.x.copy();plus[i]+=step;minus[i]-=step
            a=solve(plus);b=solve(minus)
            hessian[:,i]=(a[1]-b[1])/(2*step)
            cross[:,i]=((a[4]@beta-a[5])-(b[4]@beta-b[5]))/(2*step)
        hessian=(hessian+hessian.T)/2
        factor=cho_factor(hessian,lower=True,check_finite=False)
        cov=conditional+conditional@cross@cho_solve(factor,cross.T@conditional,check_finite=False)
        cov=(cov+cov.T)/2
        attempts['profile_information_min_eigenvalue']=float(np.linalg.eigvalsh(hessian).min())
        fitted=x@beta
        if weights is None:
            for i,positions in enumerate(indices):
                gamma=k[i]@z[positions].T@(y[positions]-x[positions]@beta)
                fitted[positions]+=z[positions]@gamma
        names=cached.names
        covariance_frame=pd.DataFrame(cov,index=names,columns=names)
        fitted_result=SimpleNamespace(fe_params=pd.Series(beta,index=names),bse_fe=np.sqrt(np.diag(cov)),
            resid=y-fitted if weights is None else None,fittedvalues=fitted if weights is None else None,cov_re=sigma,llf=-value,
            model=SimpleNamespace(data=SimpleNamespace(design_info=cached.design_info),exog=x),
            cov_params=lambda:covariance_frame)
        attempts.update(diagnostic_status='ok',error=None)
        return fitted_result,attempts
    except (ValueError,np.linalg.LinAlgError) as e:
        attempts['error']=str(e)
        return None,attempts
