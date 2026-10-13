"""Source-linked age priorities and numerical belief-index adaptation.
No published indexability theorem is asserted for the noisy-feedback adapter.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'results'/'belief_index_policies'

def wiswarm_priority(age,weight,reliability):
    # Tripathi et al., INFOCOM 2023, Eq. (2).
    return np.asarray(weight)*np.asarray(reliability)*np.asarray(age)**2

class RollingReliability:
    """Last W attempted responses, WiSwarm Sec. II-D. Prior fills startup only."""
    def __init__(self,prior,window=20):
        self.window=int(window); self.prior=np.asarray(prior,dtype=float)
        self.history=[[] for _ in self.prior]
    def probability(self):
        return np.array([max(1./self.window,(sum(h)+(self.window-len(h))*self.prior[i])/self.window) for i,h in enumerate(self.history)])
    def update(self,scheduled,ack):
        for i in scheduled:
            self.history[int(i)].append(float(ack[int(i)]))
            self.history[int(i)]=self.history[int(i)][-self.window:]

def tripathi_index(age,weight,reliability,cap=None):
    """Theorem 8, Eq. (12), evaluated for f(a)=w*min(a,cap).
    Constant Bernoulli reliability is frozen per source before evaluation.
    Dynamic event weight is applied at the decision time (declared adaptation).
    """
    a=np.asarray(age,dtype=float); p=np.asarray(reliability,dtype=float)
    tail=1. if cap is None else 1.-np.power(1.-p,np.maximum(float(cap)-a,0.))
    return np.asarray(weight)*(.5*p*a*(a-1.)+a*tail)

def _discounted_table(cap,belief,charges,ln,nl,qL,qN,discount=.999):
    """Sparse policy iteration; discount regularizes rare-reset/high-age classes."""
    from scipy.sparse import coo_matrix,eye
    from scipy.sparse.linalg import spsolve
    B=len(belief);S=cap*B;rows=np.arange(S);a=rows//B;b=rows%B
    inc=np.minimum(a+1,cap-1);th=belief[b];success=th*qL+(1-th)*qN
    postA=np.divide(th*qL,success,out=th.copy(),where=success>0)
    postN=np.divide(th*(1-qL),1-success,out=th.copy(),where=success<1)
    def interp(values,nextage,mass):
        z=np.clip(values,0,1)*(B-1);j=np.minimum(z.astype(int),B-2);t=z-j
        return np.r_[rows,rows],np.r_[nextage*B+j,nextage*B+j+1],np.r_[mass*(1-t),mass*t]
    pred=lambda x:x*(1-ln)+(1-x)*nl
    rr,cc,ww=interp(pred(th),inc,np.ones(S));P0=coo_matrix((ww,(rr,cc)),shape=(S,S)).tocsr()
    r1,c1,w1=interp(pred(postA),np.zeros(S,dtype=int),success)
    r2,c2,w2=interp(pred(postN),inc,1-success)
    P1=coo_matrix((np.r_[w1,w2],(np.r_[r1,r2],np.r_[c1,c2])),shape=(S,S)).tocsr()
    assert np.max(np.abs(np.asarray(P0.sum(1)).ravel()-1))<1e-12
    assert np.max(np.abs(np.asarray(P1.sum(1)).ravel()-1))<1e-12
    identity=eye(S,format='csr');cost=a+1.;pol=np.ones(S,dtype=bool)
    all_diff=[];residual=[];iterations=[]
    for charge in charges:
        for it in range(100):
            P=P0.multiply((~pol)[:,None])+P1.multiply(pol[:,None])
            v=spsolve(identity-discount*P,cost+charge*pol)
            q0=cost+discount*(P0@v);q1=cost+charge+discount*(P1@v)
            new=q1<q0-1e-8
            if np.array_equal(new,pol):break
            pol=new
        err=float(np.max(np.abs(np.minimum(q0,q1)-v)))
        if err>1e-6:raise RuntimeError(('discounted Bellman residual',charge,err,it))
        all_diff.append((q1-q0).reshape(cap,B));residual.append(err);iterations.append(it+1)
    return np.array(all_diff),np.array(residual),np.array(iterations)

class BeliefIndex:
    """Wang-style indifference construction, with matched noisy likelihoods.
    Discount .999 regularizes its numerical single-arm approximation.
    Uniform belief interpolation; charge bracketing, not a myopic priority.
    Its tables concern static unit age cost. Current weights multiply priorities.
    """
    def __init__(self,ln,nl,cap=120,belief_points=41,charge_points=97,qL=.9,qN=.02):
        self.ln=np.asarray(ln);self.nl=np.asarray(nl);self.cap=int(cap)
        self.grid=np.linspace(0.,1.,belief_points); self.tables=[];self.paths=[]
        CACHE.mkdir(parents=True,exist_ok=True)
        for ell,u in zip(self.ln,self.nl):
            spec=dict(protocol='noisy-discounted-index-v2',ln=float(ell),nl=float(u),cap=cap,
                belief_points=belief_points,charge_points=charge_points,qL=qL,qN=qN,
                discount=.999,bellman_tolerance=1e-6,policy_iteration_limit=100)
            key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:24]
            f=CACHE/(key+'.npz');self.paths.append(f)
            if f.exists():z=np.load(f);table=z['index']
            else:
                tic=time.time(); charges=np.r_[0.,np.geomspace(.01,2.*cap*cap,charge_points-1)]
                diff,residual,iterations=_discounted_table(cap,self.grid,charges,float(ell),float(u),qL,qN,.999)
                if residual.max()>2e-6:raise RuntimeError(('RVI convergence',spec,residual.max()))
                active=diff<-1e-6
                nested=int(np.sum(active[1:]&~active[:-1]))
                if nested:raise RuntimeError(('non-nested price grid; cannot label as index',spec,nested))
                crossing=np.argmax(diff>=0.,axis=0)
                if np.any(diff[-1]<-1e-6):raise RuntimeError('index charge range too small')
                rr,cc=np.indices(crossing.shape);low=np.maximum(crossing-1,0)
                d0=diff[low,rr,cc];d1=diff[crossing,rr,cc]
                l0=charges[low];l1=charges[crossing]
                frac=np.divide(-d0,d1-d0,out=np.zeros_like(d0),where=np.abs(d1-d0)>1e-12)
                table=l0+np.clip(frac,0,1)*(l1-l0)
                np.savez_compressed(f,index=table,belief=self.grid,charges=charges,
                    active_sets=np.packbits(active,axis=0),bellman_residual=residual,policy_iterations=iterations,
                    lower_charge=l0,upper_charge=l1,lower_difference=d0,upper_difference=d1,
                    spec_json=json.dumps(spec,sort_keys=True),nested_violations=nested,runtime_seconds=time.time()-tic)
            self.tables.append(table)
    def priority(self,age,belief,weight):
        a=np.clip(np.asarray(age,dtype=int)-1,0,self.cap-1)
        return np.asarray(weight)*np.array([np.interp(b,self.grid,self.tables[i][a[i]]) for i,b in enumerate(belief)])
