"""Independent dense-matrix checks for the corrected EBLUP candidate."""
import unittest
import numpy as np
from scipy.linalg import block_diag
from classical_conditioning.analysis.review.blup_comparison import EpochData,evaluate,residual_summaries,parameters,simulate,fit

class BLUPTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(12);y=rng.normal(size=(8,2,4));valid=np.ones_like(y,bool)
        times=np.tile([[5,6,8,10],[40,41,43,45]],(8,1,1));conditions=np.repeat([0,1],4)
        data=EpochData(tuple(map(str,range(8))),conditions,y,valid,times,((5,10),(40,45)))
        theta=np.array([-.3,-.6,.4,-.5,-.2,.45]);return data,theta

    def dense(self,data,theta):
        G,sigma,phi=parameters(theta);blocks=[];Xs=[];Zs=[];ys=[];Rs=[]
        for i in range(len(data.fish)):
            Z=np.repeat(np.eye(2),4,axis=0);X=np.zeros((8,4));X[:,data.conditions[i]*2:data.conditions[i]*2+2]=Z
            R=block_diag(*[sigma[data.conditions[i]]**2*phi**np.abs(t[:,None]-t[None,:]) for t in data.times[i]])
            blocks.append(R+Z@G@Z.T);Xs.append(X);Zs.append(Z);ys.append(data.y[i].reshape(-1));Rs.append(R)
        V=block_diag(*blocks);X=np.vstack(Xs);y=np.concatenate(ys);Vi=np.linalg.inv(V);C=np.linalg.inv(X.T@Vi@X);beta=C@X.T@Vi@y
        nll=.5*(np.linalg.slogdet(V)[1]+np.linalg.slogdet(X.T@Vi@X)[1]+(y-X@beta)@Vi@(y-X@beta)+(len(y)-4)*np.log(2*np.pi))
        return G,blocks,Xs,Zs,ys,Rs,C,beta,nll

    def test_reml_and_fixed_effects_match_dense_likelihood(self):
        data,th=self.fixture();m=evaluate(data,th,detail=True);*_,C,beta,nll=self.dense(data,th)
        np.testing.assert_allclose(m['nll'],nll,rtol=1e-11);np.testing.assert_allclose(m['beta'],beta,rtol=1e-10)
        np.testing.assert_allclose(m['covbeta'],C,rtol=1e-10)

    def test_random_prediction_and_full_pev_include_fixed_random_dependence(self):
        data,th=self.fixture();m=evaluate(data,th,detail=True);G,Vs,Xs,Zs,ys,Rs,C,beta,_=self.dense(data,th);c=np.array([1.,-1.])
        for i,(V,X,Z,y) in enumerate(zip(Vs,Xs,Zs,ys)):
            B=G@Z.T@np.linalg.inv(V);b=B@(y-X@beta);np.testing.assert_allclose(m['random'][i],b,atol=1e-12)
            K=np.zeros((2,4));K[:,data.conditions[i]*2:data.conditions[i]*2+2]=np.eye(2)
            H=G-B@Z@G;v=c@(K-B@X);vref=v-np.r_[c,[0.,0.]]
            self.assertAlmostEqual(m['se'][i]**2,c@H@c+v@C@v,places=11)
            self.assertAlmostEqual(m['excess_se'][i]**2,c@H@c+vref@C@vref,places=11)

    def test_uses_scheduled_gaps_and_condition_variances(self):
        data,th=self.fixture();G,sigma,phi=parameters(th);s,t,q,ld=residual_summaries(data,sigma,phi)
        for i in range(len(data.fish)):
            denseq=0;denseld=0
            for e in (0,1):
                tt=data.times[i,e];R=sigma[data.conditions[i]]**2*phi**np.abs(tt[:,None]-tt[None,:]);Q=np.linalg.inv(R);yy=data.y[i,e]
                self.assertAlmostEqual(s[i,e],np.ones(4)@Q@np.ones(4),places=11)
                self.assertAlmostEqual(t[i,e],np.ones(4)@Q@yy,places=11)
                denseq+=yy@Q@yy;denseld+=np.linalg.slogdet(R)[1]
            self.assertAlmostEqual(q[i],denseq,places=11);self.assertAlmostEqual(ld[i],denseld,places=11)

    def test_fixed_plus_random_is_reported_instead_of_random_deviation_alone(self):
        data,th=self.fixture();m=evaluate(data,th,detail=True)
        expected=(m['beta'].reshape(2,2)[data.conditions]+m['random'])@np.array([1.,-1.])
        np.testing.assert_allclose(m['total'],expected)
        np.testing.assert_allclose(m['excess'],expected-m['control'])

    def test_null_simulation_has_zero_true_acquisition_and_preserves_mask(self):
        data,th=self.fixture();m=evaluate(data,th,detail=True);d,total,excess=simulate(data,m,np.random.default_rng(7),'null')
        np.testing.assert_allclose(total,0.,atol=1e-12);np.testing.assert_allclose(excess,0.,atol=1e-12)
        np.testing.assert_array_equal(d.valid,data.valid);np.testing.assert_array_equal(d.times,data.times)

if __name__=='__main__':unittest.main()
