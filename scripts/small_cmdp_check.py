"""Exact joint finite-state CMDP reference with matched information and energy constraints.
Two sources, one band; observed Markov channel/event states; stochastic delivery.
The age cost is w(t)*A(t+1), exactly the manuscript convention. No relaxed fleet constraint.
"""
from pathlib import Path
import json,time,itertools,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'
POW=np.array([1.,1.2]); BUD=.3*POW
QL,QN=.90,.02; EVENT=.08; OFF=.04; ON=OFF*EVENT/(1-EVENT)
T=60000; WARM=10000; VS=[2.,6.,20.,50.,200.]; CAPS=[8,12]; SEEDS=range(5)
def exogenous_transition():
    p=np.zeros((16,16)); cb=np.array([[.875,.125],[.125,.875]]); ce=np.array([[1-ON,ON],[OFF,1-OFF]])
    for b,e in itertools.product(range(4),repeat=2):
        for bn,en in itertools.product(range(4),repeat=2):
            z=1.
            for n in range(2): z*=cb[(b>>n)&1,(bn>>n)&1]*ce[(e>>n)&1,(en>>n)&1]
            p[b*4+e,bn*4+en]=z
    assert np.allclose(p.sum(1),1)
    return p
def solve(cap):
    pe=exogenous_transition(); ns=cap*cap*16; nv=ns*3; rows=[]; cols=[]; vals=[]; costs=np.zeros(nv)
    def sid(a,b,ch,ev): return ((a-1)*cap+b-1)*16+ch*4+ev
    for a,b,ch,ev in itertools.product(range(1,cap+1),range(1,cap+1),range(4),range(4)):
        s=sid(a,b,ch,ev); aged=np.array([min(a+1,cap),min(b+1,cap)]); w=np.array([5. if(ev>>n)&1 else 1. for n in range(2)])
        for act in range(3):
            j=3*s+act; rows.append(s); cols.append(j); vals.append(1.)
            branches=[(1.,aged)]
            if act:
                n=act-1; q=QN if(ch>>n)&1 else QL; reset=aged.copy(); reset[n]=1
                branches=[(1-q,aged),(q,reset)]
            costs[j]=sum(prob*float(w@aa) for prob,aa in branches)
            for prob,aa in branches:
                for ex in range(16):
                    rows.append(sid(int(aa[0]),int(aa[1]),ex//4,ex%4)); cols.append(j); vals.append(-prob*pe[ch*4+ev,ex])
            rows.append(ns); cols.append(j); vals.append(1.)
    eq=coo_matrix((vals,(rows,cols)),shape=(ns+1,nv)).tocsr(); rhs=np.r_[np.zeros(ns),1.]
    ub=coo_matrix((np.tile(POW,ns),(np.tile([0,1],ns),np.repeat(3*np.arange(ns),2)+np.tile([1,2],ns))),shape=(2,nv)).tocsr()
    sol=linprog(costs,A_ub=ub,b_ub=BUD,A_eq=eq,b_eq=rhs,bounds=(0,None),method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    assert sol.success,sol.message
    flow=float(np.max(np.abs(eq@sol.x-rhs))); power=ub@sol.x
    dual=float(rhs@sol.eqlin.marginals+BUD@sol.ineqlin.marginals)
    red=costs-eq.T@sol.eqlin.marginals-ub.T@sol.ineqlin.marginals
    cert=dict(cap=cap,states=ns,actions=3,stationary_optimum=float(sol.fun),power=list(power),budgets=list(BUD),flow_residual=flow,duality_gap=abs(float(sol.fun)-dual),min_reduced_cost=float(red.min()))
    assert flow<1e-7 and cert['duality_gap']<1e-7 and red.min()>-1e-7 and np.all(power<=BUD+1e-7)
    occ=sol.x.reshape(ns,3); norm=occ.sum(1); policy=np.zeros_like(occ); policy[:,0]=1
    ok=norm>1e-12; policy[ok]=occ[ok]/norm[ok,None]
    np.savez_compressed(RES/f'small_cmdp_cap{cap}.npz',policy=policy,occupancy=occ,costs=costs,dual_flow=sol.eqlin.marginals[:-1],dual_normalization=sol.eqlin.marginals[-1],dual_energy=sol.ineqlin.marginals)
    return cert,policy
def trace(seed):
    rng=np.random.default_rng(830000+seed); ch=np.zeros((T,2),bool); ev=ch.copy(); ch[0]=rng.random(2)<.5; ev[0]=rng.random(2)<EVENT
    u=rng.random((T,2)); v=rng.random((T,2)); potential=rng.random((T,2)); act_u=rng.random(T)
    for t in range(1,T):
        ch[t]=np.where(ch[t-1],u[t]>=.125,u[t]<.125); ev[t]=np.where(ev[t-1],v[t]>=OFF,v[t]<ON)
    outcomes=potential<np.where(ch,QN,QL)
    return ch,ev,outcomes,act_u
def episode(cap,policy_name,V,seed,tr,lp_policy):
    ch,ev,outcomes,act_u=tr; a=b=1; q0=q1=0.; total=0.; pow0=pow1=0.; maxq=0.; qstart=None
    for t in range(T):
        w0=5. if ev[t,0] else 1.; w1=5. if ev[t,1] else 1.; d0=min(a+1,cap)-1; d1=min(b+1,cap)-1
        if policy_name=='Joint CMDP':
            s=((a-1)*cap+b-1)*16+(int(ch[t,0])+2*int(ch[t,1]))*4+int(ev[t,0])+2*int(ev[t,1]); pp=lp_policy[s]
            uu=act_u[t]; act=0 if uu<pp[0] else (1 if uu<pp[0]+pp[1] else 2)
        else:
            r0=V*w0*d0*(QN if ch[t,0] else QL)-q0*POW[0]; r1=V*w1*d1*(QN if ch[t,1] else QL)-q1*POW[1]
            act=0 if max(r0,r1)<=0 else (1 if r0>=r1 else 2)
        p0=POW[0] if act==1 else 0.; p1=POW[1] if act==2 else 0.
        a=1 if act==1 and outcomes[t,0] else min(a+1,cap); b=1 if act==2 and outcomes[t,1] else min(b+1,cap)
        if t==WARM: qstart=(q0,q1)
        q0=max(q0-BUD[0],0.)+p0; q1=max(q1-BUD[1],0.)+p1; maxq=max(maxq,q0,q1)
        if t>=WARM: total+=w0*a+w1*b; pow0+=p0; pow1+=p1
    energy=np.array([pow0,pow1])/(T-WARM)
    if policy_name=='FreshSky':
        envelope=V*5*(cap-1)/BUD+POW
        assert maxq<=envelope.max()+1e-7
        assert np.all(energy<=BUD+(np.array([q0,q1])-qstart)/(T-WARM)+1e-9)
    return dict(cap=cap,policy=policy_name,V=V,seed=seed,wAoI=total/(T-WARM),power_1=energy[0],power_2=energy[1],max_power_ratio=float(np.max(energy/BUD)),T=T,warmup=WARM)
def main():
    tic=time.time(); certs=[]; rows=[]
    traces=[trace(sd) for sd in SEEDS]
    for cap in CAPS:
        cert,pp=solve(cap); certs.append(cert); print('LP',json.dumps(cert),flush=True)
        for pol,vs in [('Joint CMDP',[0.]),('FreshSky',VS)]:
            for V in vs:
                for sd in SEEDS: rows.append(episode(cap,pol,V,sd,traces[sd],pp))
                print('done',cap,pol,V,flush=True)
    df=pd.DataFrame(rows); df['optimum']=df.cap.map({c['cap']:c['stationary_optimum'] for c in certs}); df['gap_pct']=100*(df.wAoI/df.optimum-1)
    df.to_csv(RES/'small_cmdp_seeds.csv',index=False)
    q=df.groupby(['cap','policy','V'],sort=False)[['wAoI','gap_pct','power_1','power_2','max_power_ratio']].agg(['mean','std']); q.columns=['_'.join(c) for c in q.columns]; q.reset_index().to_csv(RES/'small_cmdp_summary.csv',index=False)
    manifest=dict(N=2,M=1,caps=CAPS,V_sweep=VS,seeds=list(SEEDS),T=T,warmup=WARM,powers=list(POW),budgets=list(BUD),qL=QL,qN=QN,block_transition=.125,event_occupancy=EVENT,event_off=OFF,weight_values=[1,5],information='both schemes observe current channel and event states; share pre-generated potential delivery outcomes',cost='w(t)*A(t+1)',solver='joint occupancy-measure LP with individual energy constraints and per-slot one-band actions',certificates=certs,runtime_seconds=time.time()-tic)
    (RES/'small_cmdp_manifest.json').write_text(json.dumps(manifest,indent=2)); print(q.reset_index().to_string(index=False),flush=True)
if __name__=='__main__': main()
