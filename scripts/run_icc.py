"""ICC revision: exact capped-age gain, exact noisy-feedback likelihood, paired traces.
Run from this directory's parent with python scripts/run_icc.py. No network calls.
Powers are mW; multiply power-slot sums by 0.1 to obtain mJ.
"""
from pathlib import Path
import json, hashlib, time, platform
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import env_and_cmdp as F
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'; FIG=ROOT/'figures'
SEEDS=range(5)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8,'legend.fontsize':7,'axes.titlesize':8,
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
 'axes.linewidth':.7,'lines.linewidth':1.4,'pdf.fonttype':42,'ps.fonttype':42,
 'savefig.dpi':300,'svg.fonttype':'none'})
COL={'FreshSky':'#0F4D92','No value':'#42949E','Max-weight':'#767676',
 'Fixed price':'#B64342','Dual price':'#8A8A8A','Tang-CMDP':'#CFCECE',
 'Matched dual':'#9A4D8E','Tuned dual':'#2C7C55','Weighted CMDP':'#999999','Belief dual':'#9A4D8E',
 'FreshSky-B':'#0F4D92','Memoryless':'#B64342','Aggregate queue':'#42949E','No queue':'#767676'}

def top(score,allowed,M):
    score=np.where(allowed,score,-np.inf)
    ix=np.argsort(-score,kind='stable')[:M]
    return ix[score[ix]>0]

def full_episode(policy,seed,V=200.,shift=False,lam=.25,T=3000,warmup=300,guard=False,reserve_frac=1.,dual_eta=None,cmdp_budget_scale=1.,cmdp_age_cap=80,cmdp_bins=8):
    cfg=F.clone(F.Cfg(),seed=seed,V=V,T=T)
    env=F.Env(cfg); raw=np.ones(cfg.N); price=np.full(cfg.N,lam)
    tang=F.TangCMDP(cfg) if policy=='Tang-CMDP' else None
    if policy=='Weighted CMDP':
        from weighted_cmdp import WeightedCMDP
        tang=WeightedCMDP(cfg,age_cap=cmdp_age_cap,bins=cmdp_bins,budget_scale=cmdp_budget_scale)
    if policy in ('Matched dual','Tuned dual'): price=np.zeros(cfg.N)
    eta=(1./V) if policy=='Matched dual' else (.003 if dual_eta is None else dual_eta)
    radio=np.full(cfg.N,reserve_frac*cfg.tau*T*cfg.pbar)  # mJ, finite mission reserve
    power=[]; costs=[]; rawcost=[]; saturation=[]; queues=[]; deliveries=[]
    for t in range(T):
        if shift and t==1500:
            env.cfg.p_ihigh=.6; env.cfg.I_high=4e-8
            env.cfg.p_event=.5; env.cfg.mean_sojourn_ev=50.
        obs=env.observe(); env._last=obs
        d=np.minimum(obs['A']+1,cfg.Amax)-1
        allowed=obs['deliverable'].copy()
        if guard: allowed &= cfg.tau*obs['pth'] <= radio+1e-10
        if tang is not None:
            sched,pdct=tang.decide(env,obs,eligible=allowed) if policy=='Weighted CMDP' else tang.decide(env,obs)
        else:
            if policy=='FreshSky': score=V*obs['w']*d-obs['Q']*obs['pth']
            elif policy=='No value': score=V*d-obs['Q']*obs['pth']
            elif policy=='Max-weight': score=obs['w']*d
            elif policy=='Fixed price': score=obs['w']*d-lam*obs['pth']
            elif policy in ('Dual price','Matched dual','Tuned dual'): score=obs['w']*d-price*obs['pth']
            else: raise ValueError(policy)
            sched=top(score,allowed,cfg.M)
            pdct={n:float(obs['pth'][n]) if n in sched else 0. for n in range(cfg.N)}
        if guard and tang is not None:
            sched=[int(n) for n in sched if allowed[n]]
            pdct={n:float(obs['pth'][n]) if n in sched else 0. for n in range(cfg.N)}
        s,p=env.step(sched,pdct)
        raw=np.where(s,1.,raw+1.)
        radio-=cfg.tau*p
        if guard: assert radio.min()>=-1e-8
        if policy in ('Dual price','Matched dual','Tuned dual'): price=np.maximum(price+eta*(p-cfg.pbar),0.)
        if policy in ('FreshSky','No value'):
            qcap=V*10*(cfg.Amax-1)/cfg.pbar+cfg.Pmax
            assert env.Q.max()<=qcap+1e-7
        power.append(p); deliveries.append(s); queues.append(env.Q.sum())
        costs.append(np.sum(obs['w']*env.A)); rawcost.append(np.sum(obs['w']*raw))
        saturation.append(np.mean(raw>cfg.Amax))
    power=np.array(power); queues=np.array(queues)
    ts=1500 if shift else 0
    deficit=np.cumsum(power[ts:]-cfg.pbar,axis=0)
    # Worst individual UAV and worst prefix since shift; not fleet average.
    psi=cfg.tau*max(0.,float(deficit.max()))
    r=dict(policy=policy,seed=seed,V=V,wAoI=np.mean(costs[warmup:]),
     raw_wAoI=np.mean(rawcost[warmup:]),saturation=np.mean(saturation[warmup:]),
     max_power=power[warmup:].mean(0).max(),mean_power=power[warmup:].mean(),
     whole_max_power=power.mean(0).max(),psi_mJ=psi,
     backlog=queues[warmup:].mean(),budget=cfg.pbar,
     battery_min_mJ=radio.min(),guard=guard,reserve_frac=reserve_frac,
     dual_eta=eta if policy in ('Dual price','Matched dual','Tuned dual') else np.nan,
     dual_initial_price=lam if policy=='Dual price' else (0. if policy in ('Matched dual','Tuned dual') else np.nan),
     cmdp_budget_scale=cmdp_budget_scale if policy=='Weighted CMDP' else np.nan,
     tail_wAoI=float(np.mean(costs[-300:])),tail_delivery_rate=float(np.mean(deliveries[-300:])),
     tail_served_fraction=float(np.any(deliveries[-300:],axis=0).mean()),
     radio_spent_max_mJ=float(cfg.tau*power.sum(0).max()),
     max_radial_speed_mps=float(env.peak_radial_speed.max()))
    return r,power,queues

def partial_trace(seed,T=4000,dwell=8):
    rng=np.random.default_rng(7000+seed); N=12
    r=rng.uniform(60,380,N); dist=np.sqrt(100**2+r**2)
    pi=np.clip(F.plos(np.degrees(np.arctan(100/r)),9.61,.16),.05,.95)
    pNL=np.full(N,1/dwell); pLN=np.clip(pNL*(1-pi)/pi,0,1)
    # Clipping changes the invariant distribution; use the actual chain prior.
    pi=pNL/(pLN+pNL)
    assert np.allclose(pi,pi*(1-pLN)+(1-pi)*pNL,rtol=0,atol=1e-14)
    los=np.empty((T,N),bool); los[0]=rng.random(N)<pi
    for t in range(1,T):
        u=rng.random(N); los[t]=np.where(los[t-1],u>=pLN,u<pNL)
    return los,np.broadcast_to(dist,(T,N)),pi,pLN,pNL

def partial_episode(policy,seed,los,dist,pi,pLN,pNL,pbar=.4,V=6.,Amax=120,start=300,label='GE',radio_reserve_mJ=None):
    T,N=los.shape; rng=np.random.default_rng(91000+seed)
    # One random draw per UAV per slot regardless of scheduling: shared potential outcomes.
    delivery=rng.random((T,N))<np.where(los,.90,.02)
    ev=np.empty((T,N),bool); ev[0]=rng.random(N)<.08
    onset=(1/25)*.08/(1-.08)
    for t in range(1,T):
        u=rng.random(N); ev[t]=np.where(ev[t-1],u>=1/25,u<onset)
    fspl=20*np.log10(4*np.pi*2.4e9*dist/3e8)
    sigma=1.38e-23*290*20e6*10**.7*1e3
    pth=3*sigma/10**(-(fspl+1)/10)
    A=np.ones(N); raw=A.copy(); Q=np.zeros(N); th=pi.copy(); Z=0.; dual=np.zeros(N)
    reserve=None if radio_reserve_mJ is None else np.broadcast_to(np.asarray(radio_reserve_mJ,dtype=float),(N,)).copy()
    initial_reserve=None if reserve is None else reserve.copy(); guard_rejections=0
    pw=[]; ca=[]; ra=[]; sat=[]; delivery_log=[]
    # Replay suffix only: prefix calibrates city model and is not scored.
    for t in range(start,T):
        w=np.where(ev[t],5.,1.); d=np.minimum(A+1,Amax)-1
        success=th*.90+(1-th)*.02
        if policy=='Memoryless': success=pi*.90+(1-pi)*.02
        price=np.full(N,Z) if policy=='Aggregate queue' else Q
        if policy=='No queue': price=np.zeros(N)
        if policy=='Belief dual': price=V*dual
        score=V*w*d*success-price*pth[t]
        allowed=pth[t]<=100.
        if reserve is not None:
            ok=.1*pth[t]<=reserve
            guard_rejections+=int(np.sum(allowed&~ok&(score>0)))
            allowed &= ok
        sched=top(score,allowed,3)
        p=np.zeros(N); p[sched]=pth[t,sched]
        if reserve is not None:
            reserve-=.1*p; assert reserve.min()>=0.
        ack=np.zeros(N,bool); ack[sched]=delivery[t,sched]
        A=np.where(ack,1.,np.minimum(A+1,Amax)); raw=np.where(ack,1.,raw+1.)
        Q=np.maximum(Q-pbar,0.)+p; Z=max(Z-N*pbar,0.)+p.sum()
        dual=np.maximum(dual+(p-pbar)/V,0.)
        post=th.copy(); h=th[sched]
        lL=np.where(ack[sched],.90,.10); lN=np.where(ack[sched],.02,.98)
        post[sched]=h*lL/(h*lL+(1-h)*lN)
        th=post*(1-pLN)+(1-post)*pNL
        assert Q.max()<=V*5*(Amax-1)/pbar+100+1e-8 or policy in ('No queue','Aggregate queue','Belief dual')
        delivery_log.append(ack); pw.append(p); ca.append(np.sum(w*A)); ra.append(np.sum(w*raw)); sat.append(np.mean(raw>Amax))
    pw=np.array(pw)
    return dict(policy=policy,seed=seed,setting=label,wAoI=np.mean(ca),raw_wAoI=np.mean(ra),
     saturation=np.mean(sat),max_power=pw.mean(0).max(),mean_power=pw.mean(),budget=pbar,
     psi_mJ=.1*max(0.,np.cumsum(pw-pbar,axis=0).max()),V=V,
     radio_guard=reserve is not None,guard_rejections=guard_rejections,
     battery_min_mJ=np.nan if reserve is None else reserve.min(),
     radio_reserve_mJ=np.nan if initial_reserve is None else initial_reserve.min(),
     tail_wAoI=float(np.mean(ca[-300:])),tail_delivery_rate=float(np.mean(delivery_log[-300:])),
     tail_served_fraction=float(np.any(delivery_log[-300:],axis=0).mean()),
     radio_spent_max_mJ=float(.1*pw.sum(0).max()))

def summarise(df,groups,name):
    cols=[x for x in ('wAoI','raw_wAoI','saturation','max_power','mean_power','psi_mJ','backlog','whole_max_power','battery_min_mJ','tail_wAoI','tail_delivery_rate','tail_served_fraction','radio_spent_max_mJ') if x in df]
    s=df.groupby(groups,sort=False)[cols].agg(['mean','std'])
    s.columns=['_'.join(x) for x in s.columns]; s=s.reset_index()
    s.to_csv(RES/(name+'_summary.csv'),index=False); df.to_csv(RES/(name+'_seeds.csv'),index=False)
    print(name, s.to_string(index=False),flush=True)
    return s

def savefig(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.03)
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight',pad_inches=.03,dpi=300)
    plt.close(fig)

def main():
    tic=time.time(); RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    calibration_seeds=(50,51); tune=[]
    for lam in [.02,.05,.10,.20,.40,.80,1.6,3.2]:
        rr=[full_episode('Fixed price',s,lam=lam,T=1500)[0] for s in calibration_seeds]
        tune.append(dict(lam=lam,cost=np.mean([r['wAoI'] for r in rr]),max_power=max(r['max_power'] for r in rr)))
    pd.DataFrame(tune).to_csv(RES/'fixed_price_tuning.csv',index=False)
    ok=[r for r in tune if r['max_power']<=3.]
    if not ok: raise RuntimeError('No fixed-price candidate satisfies calibration budgets')
    lam=min(ok,key=lambda r:r['cost'])['lam']; print('fixed price',lam,flush=True)
    dual_tune=[]
    for eta in [.003,.01,.03,.1,.3]:
        rr=[full_episode('Tuned dual',s,dual_eta=eta,T=1500)[0] for s in calibration_seeds]
        dual_tune.append(dict(eta=eta,cost=np.mean([r['wAoI'] for r in rr]),max_power=max(r['max_power'] for r in rr)))
    pd.DataFrame(dual_tune).to_csv(RES/'dual_tuning.csv',index=False)
    ok=[r for r in dual_tune if r['max_power']<=3.]
    selected=min(ok,key=lambda r:r['cost']) if ok else min(dual_tune,key=lambda r:(r['max_power'],r['cost']))
    eta=selected['eta']; print('dual eta',eta,'calibration feasible',bool(ok),flush=True)
    cmdp_tune=[]
    for scale in [.5,.75,1.]:
        rr=[full_episode('Weighted CMDP',s,cmdp_budget_scale=scale,T=1500)[0] for s in calibration_seeds]
        cmdp_tune.append(dict(scale=scale,cost=np.mean([r['wAoI'] for r in rr]),max_power=max(r['max_power'] for r in rr)))
    pd.DataFrame(cmdp_tune).to_csv(RES/'weighted_cmdp_tuning.csv',index=False)
    okcmdp=[r for r in cmdp_tune if r['max_power']<=3.]
    chosen=min(okcmdp,key=lambda r:r['cost']) if okcmdp else min(cmdp_tune,key=lambda r:(r['max_power'],r['cost']))
    scale=chosen['scale']; print('CMDP scale',scale,flush=True)
    protocol=dict(version='1.1.0',seeds=list(SEEDS),calibration_seeds=list(calibration_seeds),calibration_slots=1500,
     fixed_price=lam,tuned_dual_eta=eta,tuned_dual_calibration_feasible=bool(ok),weighted_cmdp_budget_scale=scale,
     weighted_cmdp_calibration_feasible=bool(okcmdp),weighted_cmdp_age_cap=80,weighted_cmdp_bins=8,
     matched_dual_eta=.005,matched_dual_initial_price=0.,synthetic_radial_speed_range_mps=[6,14],
     partial_prior='invariant distribution of clipped transition matrix',python=platform.python_version(),
     numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__)
    (RES/'manifest.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    pols=['FreshSky','No value','Max-weight','Fixed price','Matched dual','Tuned dual','Weighted CMDP','Tang-CMDP']
    rows=[]
    for pol in pols:
        for sd in SEEDS: rows.append(full_episode(pol,sd,lam=lam,dual_eta=eta,cmdp_budget_scale=scale)[0])
        print('completed stationary',pol,flush=True)
    summarise(pd.DataFrame(rows),['policy'],'stationary')
    rows=[];arrays={}
    for pol in ['FreshSky','Fixed price','Matched dual','Tuned dual','No value']:
        pp=[]
        for sd in SEEDS:
            r,p,q=full_episode(pol,sd,shift=True,lam=lam,dual_eta=eta);rows.append(r);pp.append(p.mean(1))
        arrays[pol]=np.array(pp)
    summarise(pd.DataFrame(rows),['policy'],'shift');np.savez_compressed(RES/'shift_power_traces.npz',**arrays)
    rows=[]
    for pol in ['FreshSky','Matched dual']:
        for V in [2.,10.,50.,200.,500.]:
            for sd in SEEDS: rows.append(full_episode(pol,sd,V=V)[0])
    sweep=pd.DataFrame(rows);summarise(sweep[sweep.policy=='FreshSky'],['V'],'V_sweep');summarise(sweep,['policy','V'],'matched_pareto')
    rows=[]
    for dwell in [2.,8.,20.]:
        for sd in SEEDS:
            lo,di,pi,ln,nl=partial_trace(sd,dwell=dwell)
            for pol in ['FreshSky-B','Memoryless','Belief dual','Aggregate queue','No queue']:
                rows.append(partial_episode(pol,sd,lo,di,pi,ln,nl,label='dwell='+str(dwell)))
    summarise(pd.DataFrame(rows),['setting','policy'],'belief')
    # Legacy single-patrol replay retained as a secondary artifact.
    rows=[];citymeta=[]
    for city in ['midtown_manhattan','city_of_london']:
        z=np.load(ROOT/'data'/('osm_traces_'+city+'.npz'));lo=z['los'];di=z['dist'];train=lo[:300].astype(bool)
        pi=np.clip(train.mean(0),.05,.95);nL=train[:-1].sum(0);nN=(~train[:-1]).sum(0)
        ln=((train[:-1]&~train[1:]).sum(0)+1)/(nL+2);nl=((~train[:-1]&train[1:]).sum(0)+1)/(nN+2)
        for sd in SEEDS:
            for pol in ['FreshSky-B','Memoryless','Aggregate queue','No queue']:
                rows.append(partial_episode(pol,sd,lo,di,pi,ln,nl,pbar=1.6,label=city))
        citymeta.append(dict(city=city,LoS=float(lo.mean()),calibration_slots=300,evaluation_slots=len(lo)-300))
    summarise(pd.DataFrame(rows),['setting','policy'],'city')
    import itertools
    rng=np.random.default_rng(82);checks=0
    for _ in range(250):
        N=5;M=2;A=rng.integers(1,121,N);w=rng.uniform(0,5,N);prob=rng.random(N);p=rng.uniform(.01,100,N);Q=rng.uniform(0,20,N)
        score=6*w*(np.minimum(A+1,120)-1)*prob-Q*p;choice=top(score,np.ones(N,bool),M)
        best=max(sum(score[list(c)]) for k in range(M+1) for c in itertools.combinations(range(N),k))
        assert abs(score[choice].sum()-best)<1e-8;checks+=1
    Q=0.;cap=6*5*119/.4+100
    for _ in range(100000):
        reward=rng.uniform(0,6*5*119);p=10**rng.uniform(-10,2);spent=p if reward-Q*p>0 else 0.;Q=max(Q-.4,0)+spent
        assert Q<=cap+1e-8
    protocol.update(city_metadata=citymeta,enumeration_checks=checks,queue_stress_slots=100000,runtime_seconds=time.time()-tic,
     hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.npz')})
    (RES/'manifest.json').write_text(json.dumps(protocol,indent=2),encoding='utf8');print('FINISHED',flush=True)
if __name__=='__main__':main()
