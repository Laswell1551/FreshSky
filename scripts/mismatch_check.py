"""Feedback-model misspecification with paired traces and a same-history true belief.
Truth stays qL=.90, qN=.02. Only the scheduler's calibrated model is perturbed.
"""
from run_icc import *
import sys
GUARD = '--unguarded' not in sys.argv
TAG = '' if GUARD else '_unguarded'
MODELS={'Calibrated':(1.,.90,.02),'Model A':(.5,.80,.10),'Model B':(2.,.95,.005)}
def episode(model,seed,los,dist,pi,ln,nl,dwell):
    scale,ql,qn=MODELS[model]; eln=np.clip(scale*ln,0,1); enl=np.clip(scale*nl,0,1)
    T,N=los.shape; rng=np.random.default_rng(91000+seed); delivery=rng.random((T,N))<np.where(los,.90,.02)
    ev=np.empty((T,N),bool); ev[0]=rng.random(N)<.08; onset=(1/25)*.08/(1-.08)
    for t in range(1,T):
        u=rng.random(N); ev[t]=np.where(ev[t-1],u>=1/25,u<onset)
    sigma=1.38e-23*290*20e6*10**.7*1e3; pth=3*sigma/10**(-(20*np.log10(4*np.pi*2.4e9*dist/3e8)+1)/10)
    A=np.ones(N); raw=A.copy(); Q=np.zeros(N); h=pi.copy(); ht=pi.copy(); pw=[]; ca=[]; ra=[]; sat=[]; errs=[]; gaps=[]
    V=6.; cap=120; pbar=.4; M=3; reserve=np.full(N,.1*pbar*(T-300) if GUARD else np.inf); guard_rejections=0
    for t in range(300,T):
        w=np.where(ev[t],5.,1.); d=np.minimum(A+1,cap)-1; rh=h*ql+(1-h)*qn; rt=ht*.90+(1-ht)*.02
        allowed=pth[t]<=100.; guard=.1*pth[t]<=reserve; score=V*w*d*rh-Q*pth[t]; guard_rejections+=int(np.sum(allowed&~guard&(score>0))); allowed &= guard; chosen=top(score,allowed,M)
        truth=V*w*d*rt-Q*pth[t]; true_best=top(truth,allowed,M); delta=np.max(np.abs(rh[allowed]-rt[allowed])) if allowed.any() else 0.
        gap=max(0.,float(truth[true_best].sum()-truth[chosen].sum())); assert gap<=2*M*V*5*(cap-1)*delta+1e-7
        errs.append(delta); gaps.append(gap)
        p=np.zeros(N); p[chosen]=pth[t,chosen]; reserve-=.1*p; assert reserve.min()>=0.; ack=np.zeros(N,bool); ack[chosen]=delivery[t,chosen]
        A=np.where(ack,1.,np.minimum(A+1,cap)); raw=np.where(ack,1.,raw+1.); Q=np.maximum(Q-pbar,0)+p
        def update(prior,qL,qN,pLN,pNL):
            post=prior.copy(); hs=prior[chosen]; lL=np.where(ack[chosen],qL,1-qL); lN=np.where(ack[chosen],qN,1-qN)
            post[chosen]=hs*lL/(hs*lL+(1-hs)*lN)
            return post*(1-pLN)+(1-post)*pNL
        h=update(h,ql,qn,eln,enl); ht=update(ht,.90,.02,ln,nl)
        assert Q.max()<=V*5*(cap-1)/pbar+100+1e-8
        pw.append(p); ca.append(np.sum(w*A)); ra.append(np.sum(w*raw)); sat.append(np.mean(raw>cap))
    pw=np.array(pw)
    return dict(model=model,dwell=dwell,seed=seed,transition_scale=scale,qL_hat=ql,qN_hat=qn,wAoI=np.mean(ca),raw_wAoI=np.mean(ra),saturation=np.mean(sat),max_power=pw.mean(0).max(),mean_power=pw.mean(),budget=pbar,rho_error_max=np.max(errs),rho_error_p95=np.quantile(errs,.95),local_loss_mean=np.mean(gaps),local_loss_max=np.max(gaps),radio_guard=GUARD,radio_reserve_mJ=.1*pbar*(T-300) if GUARD else np.nan,battery_min_mJ=reserve.min() if GUARD else np.nan,guard_rejections=guard_rejections)
def main():
    tic=time.time(); rows=[]; frozen=pd.read_csv(RES/'belief_seeds.csv')
    for dwell in [2.,8.,20.]:
        for seed in SEEDS:
            lo,di,pi,ln,nl=partial_trace(seed,dwell=dwell)
            for model in MODELS:
                r=episode(model,seed,lo,di,pi,ln,nl,dwell); rows.append(r)
                if model=='Calibrated':
                    old=frozen[(frozen.policy=='FreshSky-B')&(frozen.seed==seed)&(frozen.setting=='dwell='+str(dwell))].iloc[0]
                    for c in ['wAoI','raw_wAoI','max_power']: assert np.isclose(r[c],old[c],rtol=1e-12,atol=1e-12),(seed,dwell,c)
        print('done mismatch dwell',dwell,flush=True)
    df=pd.DataFrame(rows); base=df[df.model=='Calibrated'].set_index(['dwell','seed']).wAoI
    df['age_change_pct']=[100*(r.wAoI/base.loc[(r.dwell,r.seed)]-1) for r in df.itertuples()]; df.to_csv(RES/f'mismatch_seeds{TAG}.csv',index=False)
    cols=['wAoI','raw_wAoI','saturation','max_power','mean_power','age_change_pct','rho_error_max','rho_error_p95','local_loss_mean','battery_min_mJ','guard_rejections']
    q=df.groupby(['model','dwell'],sort=False)[cols].agg(['mean','std']); q.columns=['_'.join(x) for x in q.columns]; q.reset_index().to_csv(RES/f'mismatch_summary{TAG}.csv',index=False)
    (RES/f'mismatch_manifest{TAG}.json').write_text(json.dumps(dict(models=MODELS,seeds=list(SEEDS),dwells=[2,8,20],T=4000,start=300,truth_likelihood=[.90,.02],paired_potential_outcomes=True,radio_reserve_mJ=.1*.4*3700 if GUARD else None,remaining_energy_guard=GUARD,calibrated_metrics_match_frozen_csv=True,true_belief_information='same observed history and true model; never hidden state',local_sensitivity_check='passed at every evaluated slot',runtime_seconds=time.time()-tic),indent=2))
    print(q.reset_index().to_string(index=False),flush=True)
if __name__=='__main__': main()
