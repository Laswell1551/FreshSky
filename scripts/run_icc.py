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
 'Fixed price':'#B64342','Dual price':'#9A4D8E','Tang-CMDP':'#CFCECE',
 'FreshSky-B':'#0F4D92','Memoryless':'#B64342','Aggregate queue':'#42949E','No queue':'#767676'}

def top(score,allowed,M):
    score=np.where(allowed,score,-np.inf)
    ix=np.argsort(-score,kind='stable')[:M]
    return ix[score[ix]>0]

def full_episode(policy,seed,V=200.,shift=False,lam=.25,T=3000,warmup=300,guard=False,reserve_frac=1.):
    cfg=F.clone(F.Cfg(),seed=seed,V=V,T=T)
    env=F.Env(cfg); raw=np.ones(cfg.N); price=np.full(cfg.N,lam)
    tang=F.TangCMDP(cfg) if policy=='Tang-CMDP' else None
    radio=np.full(cfg.N,reserve_frac*cfg.tau*T*cfg.pbar)  # mJ, finite mission reserve
    power=[]; costs=[]; rawcost=[]; saturation=[]; queues=[]
    for t in range(T):
        if shift and t==1500:
            env.cfg.p_ihigh=.6; env.cfg.I_high=4e-8
            env.cfg.p_event=.5; env.cfg.mean_sojourn_ev=50.
        obs=env.observe(); env._last=obs
        d=np.minimum(obs['A']+1,cfg.Amax)-1
        allowed=obs['deliverable'].copy()
        if guard: allowed &= cfg.tau*obs['pth'] <= radio+1e-10
        if tang is not None:
            sched,pdct=tang.decide(env,obs)
        else:
            if policy=='FreshSky': score=V*obs['w']*d-obs['Q']*obs['pth']
            elif policy=='No value': score=V*d-obs['Q']*obs['pth']
            elif policy=='Max-weight': score=obs['w']*d
            elif policy=='Fixed price': score=obs['w']*d-lam*obs['pth']
            elif policy=='Dual price': score=obs['w']*d-price*obs['pth']
            else: raise ValueError(policy)
            sched=top(score,allowed,cfg.M)
            pdct={n:float(obs['pth'][n]) if n in sched else 0. for n in range(cfg.N)}
        s,p=env.step(sched,pdct)
        raw=np.where(s,1.,raw+1.)
        radio-=cfg.tau*p
        if guard: assert radio.min()>=-1e-8
        if policy=='Dual price': price=np.maximum(price+.003*(p-cfg.pbar),0.)
        if policy in ('FreshSky','No value'):
            qcap=V*10*(cfg.Amax-1)/cfg.pbar+cfg.Pmax
            assert env.Q.max()<=qcap+1e-7
        power.append(p); queues.append(env.Q.sum())
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
     battery_min_mJ=radio.min(),guard=guard,reserve_frac=reserve_frac)
    return r,power,queues

def partial_trace(seed,T=4000,dwell=8):
    rng=np.random.default_rng(7000+seed); N=12
    r=rng.uniform(60,380,N); dist=np.sqrt(100**2+r**2)
    pi=np.clip(F.plos(np.degrees(np.arctan(100/r)),9.61,.16),.05,.95)
    pNL=np.full(N,1/dwell); pLN=np.clip(pNL*(1-pi)/pi,0,1)
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
    A=np.ones(N); raw=A.copy(); Q=np.zeros(N); th=pi.copy(); Z=0.
    reserve=None if radio_reserve_mJ is None else np.broadcast_to(np.asarray(radio_reserve_mJ,dtype=float),(N,)).copy()
    initial_reserve=None if reserve is None else reserve.copy(); guard_rejections=0
    pw=[]; ca=[]; ra=[]; sat=[]
    # Replay suffix only: prefix calibrates city model and is not scored.
    for t in range(start,T):
        w=np.where(ev[t],5.,1.); d=np.minimum(A+1,Amax)-1
        success=th*.90+(1-th)*.02
        if policy=='Memoryless': success=pi*.90+(1-pi)*.02
        price=np.full(N,Z) if policy=='Aggregate queue' else Q
        if policy=='No queue': price=np.zeros(N)
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
        post=th.copy(); h=th[sched]
        lL=np.where(ack[sched],.90,.10); lN=np.where(ack[sched],.02,.98)
        post[sched]=h*lL/(h*lL+(1-h)*lN)
        th=post*(1-pLN)+(1-post)*pNL
        assert Q.max()<=V*5*(Amax-1)/pbar+100+1e-8 or policy in ('No queue','Aggregate queue')
        pw.append(p); ca.append(np.sum(w*A)); ra.append(np.sum(w*raw)); sat.append(np.mean(raw>Amax))
    pw=np.array(pw)
    return dict(policy=policy,seed=seed,setting=label,wAoI=np.mean(ca),raw_wAoI=np.mean(ra),
     saturation=np.mean(sat),max_power=pw.mean(0).max(),mean_power=pw.mean(),budget=pbar,
     psi_mJ=.1*max(0.,np.cumsum(pw-pbar,axis=0).max()),V=V,
     radio_guard=reserve is not None,guard_rejections=guard_rejections,
     battery_min_mJ=np.nan if reserve is None else reserve.min(),
     radio_reserve_mJ=np.nan if initial_reserve is None else initial_reserve.min())

def summarise(df,groups,name):
    cols=[x for x in ('wAoI','raw_wAoI','saturation','max_power','mean_power','psi_mJ','backlog','whole_max_power','battery_min_mJ') if x in df]
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
    # Tune static price using independent training seeds, strict individual threshold.
    tune=[]
    for lam in [.02,.05,.10,.20,.40,.80,1.6,3.2]:
        rr=[full_episode('Fixed price',s,lam=lam,T=1500)[0] for s in (50,51)]
        tune.append(dict(lam=lam,cost=np.mean([r['wAoI'] for r in rr]),
                         max_power=max(r['max_power'] for r in rr)))
    pd.DataFrame(tune).to_csv(RES/'fixed_price_tuning.csv',index=False)
    ok=[r for r in tune if r['max_power']<=3.]
    if not ok: raise RuntimeError('No fixed-price candidate satisfies calibration budgets')
    lam=min(ok,key=lambda r:r['cost'])['lam']; print('fixed price',lam,flush=True)
    pols=['FreshSky','No value','Max-weight','Fixed price','Dual price','Tang-CMDP']
    mainrows=[]
    for pol in pols:
        for sd in SEEDS: mainrows.append(full_episode(pol,sd,lam=lam)[0])
        print('completed stationary',pol,flush=True)
    full=summarise(pd.DataFrame(mainrows),['policy'],'stationary')
    shiftrows=[]; trajectories=[]
    for pol in ['FreshSky','Fixed price','Dual price','No value']:
        pp=[]; qq=[]
        for sd in SEEDS:
            r,p,q=full_episode(pol,sd,shift=True,lam=lam); shiftrows.append(r); pp.append(p.mean(1)); qq.append(q)
        trajectories.append((pol,np.array(pp),np.array(qq)))
    shifts=summarise(pd.DataFrame(shiftrows),['policy'],'shift')
    sweeps=[]
    for V in [2.,10.,50.,200.,500.]:
        for sd in SEEDS: sweeps.append(full_episode('FreshSky',sd,V=V)[0])
    sweep=summarise(pd.DataFrame(sweeps),['V'],'V_sweep')
    guards=[]
    for sd in SEEDS: guards.append(full_episode('FreshSky',sd,shift=True,lam=lam,guard=True)[0])
    summarise(pd.DataFrame(guards),['policy','guard'],'battery_guard')
    partial=[]
    for dwell in [2.,8.,20.]:
        for sd in SEEDS:
            lo,di,pi,ln,nl=partial_trace(sd,dwell=dwell)
            for pol in ['FreshSky-B','Memoryless','Aggregate queue','No queue']:
                partial.append(partial_episode(pol,sd,lo,di,pi,ln,nl,label='dwell='+str(dwell)))
    beliefs=summarise(pd.DataFrame(partial),['setting','policy'],'belief')
    urban=[]; citymeta=[]
    for city in ['midtown_manhattan','city_of_london']:
        z=np.load(ROOT/'data'/('osm_traces_'+city+'.npz')); lo=z['los']; di=z['dist']
        train=lo[:300]; pi=np.clip(train.mean(0),.05,.95)
        # Frozen model calibrated only on the preceding 300-slot geometry prefix.
        nL=train[:-1].sum(0); nN=(~train[:-1].astype(bool)).sum(0)
        ln=((train[:-1].astype(bool)&~train[1:].astype(bool)).sum(0)+1)/(nL+2)
        nl=((~train[:-1].astype(bool)&train[1:].astype(bool)).sum(0)+1)/(nN+2)
        for sd in SEEDS:
            for pol in ['FreshSky-B','Memoryless','Aggregate queue','No queue']:
                urban.append(partial_episode(pol,sd,lo,di,pi,ln,nl,pbar=1.6,label=city))
        citymeta.append(dict(city=city,LoS=float(lo.mean()),min_dist=float(di.min()),max_dist=float(di.max()),
                             calibration_slots=300,evaluation_slots=len(lo)-300))
    city=summarise(pd.DataFrame(urban),['setting','policy'],'city')
    # Vector figures with SD across seeds, not confidence intervals.
    fig,axs=plt.subplots(1,2,figsize=(7.,2.15),layout='constrained')
    for pol,pp,qq in trajectories:
        ma=np.array([np.convolve(x,np.ones(100)/100,'valid') for x in pp])
        tt=np.arange(99,3000)
        axs[0].plot(tt,ma.mean(0),color=COL[pol],label=pol)
        axs[0].fill_between(tt,ma.mean(0)-ma.std(0,ddof=1),ma.mean(0)+ma.std(0,ddof=1),color=COL[pol],alpha=.12)
    axs[0].axhline(3,color='k',ls=':',lw=.9); axs[0].axvline(1500,color='0.5',ls=':',lw=.9)
    axs[0].set(xlabel='Slot',ylabel='Fleet mean power (mW)',title='(a) 100-slot moving mean')
    axs[0].legend(ncol=2,fontsize=6)
    for i,r in shifts.iterrows():
        axs[1].bar(i,r.psi_mJ_mean,yerr=r.psi_mJ_std,color=COL[r.policy],edgecolor='k',lw=.5,capsize=2)
    axs[1].set_xticks(range(len(shifts)),shifts.policy,rotation=20,ha='right')
    axs[1].set(ylabel='Worst UAV prefix excess (mJ)',title='(b) Post-shift budget excess')
    savefig(fig,'shift')
    fig,ax=plt.subplots(figsize=(3.4,1.8),layout='constrained'); bx=ax.twinx()
    l1=ax.errorbar(sweep.V,sweep.wAoI_mean,yerr=sweep.wAoI_std,color=COL['FreshSky'],marker='o',ms=3,capsize=2,label='Weighted age')
    l2=bx.errorbar(sweep.V,sweep.backlog_mean,yerr=sweep.backlog_std,color='#42949E',marker='s',ms=3,capsize=2,ls='--',label='Queue sum')
    ax.set(xscale='log',xlabel='Control weight V',ylabel='Weighted age (slots)'); bx.set_ylabel('Mean sum of queues (mW)')
    ax.tick_params(axis='y',colors=COL['FreshSky']); bx.tick_params(axis='y',colors='#42949E'); bx.spines['right'].set_visible(True)
    ax.legend([l1,l2],['Weighted age','Queue sum'],loc='upper center',fontsize=6,ncol=2)
    savefig(fig,'V_sweep')
    fig,axs=plt.subplots(1,2,figsize=(7.,2.1),layout='constrained')
    for pol,mk in [('FreshSky-B','o'),('Memoryless','s')]:
        q=beliefs[beliefs.policy==pol].copy(); xs=q.setting.str.replace('dwell=','').astype(float)
        axs[0].errorbar(xs,q.wAoI_mean,yerr=q.wAoI_std,color=COL[pol],marker=mk,ms=3,capsize=2,label=pol)
    axs[0].set(xlabel='Mean blocked dwell (slots)',ylabel='Capped weighted age (slots)',title='(a) Correlation-aware scheduling')
    axs[0].legend()
    q=beliefs[beliefs.setting=='dwell=8.0'];
    for i,r in q.iterrows():
        x=list(q.index).index(i)
        axs[1].bar(x,r.max_power_mean,yerr=r.max_power_std,color=COL[r.policy],edgecolor='k',lw=.5,capsize=2)
    axs[1].axhline(.4,color='k',ls=':',lw=.9)
    axs[1].set_xticks(range(len(q)),q.policy,rotation=20,ha='right')
    axs[1].set(ylabel='Maximum per-UAV mean power (mW)',title='(b) Individual versus fleet budget')
    savefig(fig,'belief')
    fig,axs=plt.subplots(1,2,figsize=(7.,2.),layout='constrained')
    for pol,mk in [('FreshSky-B','o'),('Memoryless','s')]:
        q=city[city.policy==pol]
        for ax,col,title in [(axs[0],'wAoI','(a) Capped weighted age'),(axs[1],'raw_wAoI','(b) Uncapped weighted age')]:
            ax.errorbar(np.arange(2),q[col+'_mean'],yerr=q[col+'_std'],marker=mk,color=COL[pol],capsize=2,ms=4,label=pol)
            ax.set_xticks([0,1],['Manhattan','London']); ax.set(ylabel='Weighted age (slots)',title=title)
    axs[0].legend()
    savefig(fig,'city')
    # Exact current-state optimum, at the cap, checked by enumeration.
    import itertools
    rng=np.random.default_rng(82); checks=0
    for _ in range(250):
        N=5; M=2; A=rng.integers(1,121,N); w=rng.uniform(0,5,N); s=rng.random(N)
        p=rng.uniform(.01,100,N); Q=rng.uniform(0,20,N); V=6.
        R=V*w*(np.minimum(A+1,120)-1)*s-Q*p
        choice=top(R,np.ones(N,bool),M)
        best=max(sum(R[list(c)]) for k in range(M+1) for c in itertools.combinations(range(N),k))
        assert abs(R[choice].sum()-best)<1e-8; checks+=1
    # Stress the deterministic queue bound with adversarial variable powers and rewards.
    Q=0.; cap=6*5*119/.4+100
    for _ in range(100000):
        reward=rng.uniform(0,6*5*119); p=10**rng.uniform(-10,2)
        spent=p if reward-Q*p>0 else 0.; Q=max(Q-.4,0)+spent
        assert Q<=cap+1e-8
    manifest=dict(seeds=list(SEEDS),fixed_price=lam,python=platform.python_version(),
     numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
     city_metadata=citymeta,enumeration_checks=checks,queue_stress_slots=100000,
     runtime_seconds=time.time()-tic,model='capped age exact marginal; exact .90/.02 success likelihood',
     hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.npz')})
    (RES/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('FINISHED',json.dumps(manifest),flush=True)
if __name__=='__main__': main()
