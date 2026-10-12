"""Event-weighted, discretized Tang-style CMDP comparison.
This is an adapted finite-state relaxation, not a reproduction of a weighted
optimality theorem for the continuous twelve-source environment. Both known
channel bins and the current Markov event state enter the LP. Bin upper bounds
are used for radio costs. Independent calibration selects nominal budget scale.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,hstack,vstack,eye,block_diag
import env_and_cmdp as F
ROOT=Path(__file__).resolve().parents[1]
class WeightedCMDP:
    _MODELS={}; _POLICIES={}
    def __init__(self,cfg,age_cap=80,bins=8,budget_scale=1.):
        self.cfg=cfg; self.X=int(age_cap); self.B=int(bins); self.scale=float(budget_scale)
        self.rng=np.random.default_rng(F.RNG_MASTER+23000+cfg.seed)
        key=(cfg.seed,cfg.N,cfg.M,cfg.pbar,cfg.Pmax,cfg.H,cfg.rmin,cfg.rmax,
             cfg.mean_sojourn_N,cfg.p_ihigh,cfg.mean_sojourn_I,cfg.I_high,
             cfg.Dbits,cfg.W,cfg.tau,cfg.wev,cfg.p_event,cfg.mean_sojourn_ev,self.X,self.B,self.scale)
        self.cache_key=hashlib.sha256(('global-relaxation-v3'+repr(key)).encode()).hexdigest()
        self.cache_file=ROOT/'results'/'weighted_cmdp_policies'/f'seed{cfg.seed}_a{self.X}_q{self.B}_b{self.scale:.2f}.npz'
        if key not in self._POLICIES:
            if self.cache_file.exists():
                z=np.load(self.cache_file);meta=json.loads(str(z['metadata']))
                if meta.get('cache_key')==self.cache_key:self._POLICIES[key]=(z['edges'],z['policy'],meta)
            if key not in self._POLICIES:self._POLICIES[key]=self._build()
        self.edges,self.xi,self.metadata=self._POLICIES[key]
    def _fit(self):
        c=self.cfg; key=(c.seed,c.N,c.Pmax,c.H,c.rmin,c.rmax,c.mean_sojourn_N,c.p_ihigh,c.mean_sojourn_I,c.I_high,c.Dbits,c.W,c.tau,self.B)
        if key in self._MODELS:return self._MODELS[key]
        env=F.Env(F.clone(c,T=6000)); env.rng=np.random.default_rng(F.RNG_MASTER+731000+c.seed)
        env.beta=env.rng.integers(0,2,c.N); env.iis=(env.rng.random(c.N)<c.p_ihigh).astype(int); env.ev=(env.rng.random(c.N)<c.p_event).astype(int)
        powers=[]; zero={n:0. for n in range(c.N)}
        for _ in range(6000):
            obs=env.observe();env._last=obs;powers.append(obs['pth'].copy());env.step([],zero)
        powers=np.array(powers); edges=np.zeros((c.N,self.B-1)); upper=np.zeros((c.N,self.B+1)); trans=np.zeros((c.N,self.B+1,self.B+1))
        for n in range(c.N):
            good=powers[:,n]<=c.Pmax; vals=powers[good,n]
            edges[n]=np.quantile(vals,np.arange(1,self.B)/self.B) if len(vals)>=self.B else np.linspace(0,c.Pmax,self.B+1)[1:-1]
            q=np.where(good,np.digitize(powers[:,n],edges[n]),self.B)
            upper[n,:self.B]=np.r_[edges[n],c.Pmax]
            counts=np.full((self.B+1,self.B+1),1e-3);np.add.at(counts,(q[:-1],q[1:]),1.)
            trans[n]=counts/counts.sum(1,keepdims=True)
            assert np.all(powers[good,n]<=upper[n,q[good]]+1e-8)
        self._MODELS[key]=(edges,upper,trans);return edges,upper,trans
    def _template(self,P,omega):
        X,Q=self.X,self.B+1;S=X*Q*2;off=1/self.cfg.mean_sojourn_ev;on=off*self.cfg.p_event/(1-self.cfg.p_event)
        E=np.array([[1-on,on],[off,1-off]]);weights=[1.,self.cfg.wev]
        rows=[];cols=[];vals=[];cost0=np.zeros(S);cost1=np.zeros(S)
        def sid(a,q,e):return (a*Q+q)*2+e
        for a in range(X):
            for q in range(Q):
                for e in range(2):
                    s=sid(a,q,e);ap=min(a+1,X-1);cost0[s]=weights[e]*(ap+1);cost1[s]=weights[e]*(1-(ap+1))
                    rows.append(s);cols.append(s);vals.append(1.)
                    for qp in range(Q):
                        for ep in range(2):
                            z=P[q,qp]*E[e,ep]
                            rows.extend([sid(ap,qp,ep),sid(ap,qp,ep),sid(0,qp,ep)])
                            cols.extend([s,S+s,S+s]);vals.extend([-z,z,-z])
        rows.extend([S]*S);cols.extend(range(S));vals.extend([1.]*S)
        eq=coo_matrix((vals,(rows,cols)),shape=(S+1,2*S)).tocsr(); rhs=np.r_[np.zeros(S),1.]
        energy=np.tile(np.repeat(omega,2),X)
        aub=vstack([hstack([-eye(S),eye(S)]),coo_matrix((np.r_[np.zeros(S),energy][None,:]))]).tocsr()
        bub=np.r_[np.zeros(S),self.cfg.pbar*self.scale]
        bounds=[(0,None)]*S+[(0,0) if (s//2)%Q==self.B else (0,None) for s in range(S)]
        return eq,rhs,aub,bub,bounds,cost0,cost1,energy
    def _solve(self,t,price):
        eq,rhs,aub,bub,bounds,c0,c1,energy=t;S=len(c0)
        sol=linprog(np.r_[c0,c1+price],A_ub=aub,b_ub=bub,A_eq=eq,b_eq=rhs,bounds=bounds,method='highs')
        if not sol.success:raise RuntimeError('Weighted CMDP: '+sol.message)
        mu=sol.x[:S].reshape(self.X,self.B+1,2);y=sol.x[S:].reshape(mu.shape)
        cert={'flow_residual':float(np.max(abs(eq@sol.x-rhs))),
              'max_inequality_residual':float(np.max(aub@sol.x-bub)),
              'nominal_power':float(energy@sol.x[S:]),
              'next_weighted_age':float(c0@sol.x[:S]+c1@sol.x[S:])}
        assert cert['flow_residual']<1e-6 and cert['max_inequality_residual']<1e-6
        return mu,y,float(y.sum()),cert
    def _build(self):
        edges,upper,P=self._fit();templates=[self._template(P[n],upper[n]) for n in range(self.cfg.N)]
        S=self.X*(self.B+1)*2;D=2*S
        # Solve the coupled average-bandwidth occupation LP directly.
        # One redundant flow equation per source is removed.
        eq=block_diag([v[0][list(range(S-1))+[S]] for v in templates],format='csr')
        rhs=np.concatenate([v[1][list(range(S-1))+[S]] for v in templates])
        aub=block_diag([v[2] for v in templates],format='csr');bub=np.concatenate([v[3] for v in templates])
        bandwidth=np.tile(np.r_[np.zeros(S),np.ones(S)],self.cfg.N)
        aub=vstack([aub,coo_matrix(bandwidth[None,:])]).tocsr();bub=np.r_[bub,self.cfg.M]
        cost=np.concatenate([np.r_[v[5],v[6]] for v in templates]);bounds=sum([v[4] for v in templates],[])
        tic=time.time();sol=None
        for method,presolve in [('highs-ipm',True),('highs-ds',False)]:
            sol=linprog(cost,A_ub=aub,b_ub=bub,A_eq=eq,b_eq=rhs,bounds=bounds,method=method,
             options={'presolve':presolve,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
            if sol.success:break
        if not sol.success:raise RuntimeError('Weighted coupled CMDP: '+sol.message)
        assert max(abs(eq@sol.x-rhs))<1e-6 and max(aub@sol.x-bub)<1e-6
        mix=[]
        for n,t in enumerate(templates):
            v=sol.x[n*D:(n+1)*D];mu=v[:S].reshape(self.X,self.B+1,2);y=v[S:].reshape(mu.shape)
            cert={'flow_residual':float(max(abs(t[0]@v-t[1]))),
             'max_inequality_residual':float(max(t[2]@v-t[3])),
             'nominal_power':float(t[7]@v[S:]),'next_weighted_age':float(t[5]@v[:S]+t[6]@v[S:])}
            assert cert['flow_residual']<1e-6 and cert['max_inequality_residual']<1e-6
            mix.append((mu,y,float(y.sum()),cert))
        price=float(-sol.ineqlin.marginals[-1]);print(f'CMDP seed={self.cfg.seed} scale={self.scale} solved in {time.time()-tic:.1f}s activation={bandwidth@sol.x:.4f}',flush=True)
        xi=[]
        for mu,y,_,_ in mix:
            ratio=np.divide(y,mu,out=np.zeros_like(y),where=mu>1e-10);ratio=np.clip(ratio,0,1)
            # Numerical-support continuation: at unrepresented high ages use
            # service in a deliverable bin, rather than absorbing into idle.
            for q in range(self.B):
                for e in range(2):
                    active=np.flatnonzero(ratio[:,q,e]>1e-8)
                    boundary=int(active[0]) if len(active) else self.X-1
                    zero=(mu[:,q,e]<=1e-10)&(np.arange(self.X)>=boundary)
                    ratio[zero,q,e]=1.
            ratio[:,self.B,:]=0.;xi.append(ratio)
        metadata={'cache_key':self.cache_key,'age_cap':self.X,'deliverable_bins':self.B,'event_states':2,'budget_scale':self.scale,
                  'train_slots':6000,'bin_power':'upper bounds, including Pmax in highest bin',
                  'cost':'current event weight times expected next age',
                  'bandwidth_price':float(price),'relaxed_activation':sum(v[2] for v in mix),
                  'hard_bandwidth':'uniform truncation of requests to M',
                  'support_continuation':'deliverable zero-mass high-age states inherit service',
                  'model_certificates':[v[3] for v in mix],
                  'relaxed_total_cost':float(sol.fun),'global_primal_residual':float(max(abs(eq@sol.x-rhs))),
                  'relaxed_objective_scope':'discretized average-bandwidth occupation LP'}
        cache=ROOT/'results'/'weighted_cmdp_policies';cache.mkdir(exist_ok=True)
        filename=f'seed{self.cfg.seed}_a{self.X}_q{self.B}_b{self.scale:.2f}.npz'
        np.savez_compressed(cache/filename,edges=edges,upper_power=upper,channel_transition=P,policy=np.array(xi),metadata=json.dumps(metadata),occupancy_mu=np.array([v[0] for v in mix]),occupancy_active=np.array([v[1] for v in mix]))
        return edges,np.array(xi),metadata
    def decide(self,env,obs,eligible=None):
        q=np.where(obs['deliverable'],[np.digitize(obs['pth'][n],self.edges[n]) for n in range(self.cfg.N)],self.B)
        a=np.minimum(obs['A'].astype(int),self.X)-1;e=(obs['w']>1).astype(int)
        request=self.rng.random(self.cfg.N)<self.xi[np.arange(self.cfg.N),a,q,e]
        valid=obs['deliverable'] if eligible is None else eligible
        chosen=np.flatnonzero(request&valid)
        if len(chosen)>self.cfg.M:chosen=self.rng.choice(chosen,self.cfg.M,replace=False)
        p={n:float(obs['pth'][n]) if n in chosen else 0. for n in range(self.cfg.N)}
        return chosen.tolist(),p
