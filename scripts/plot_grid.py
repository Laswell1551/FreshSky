"""Rebuild the six-panel ICC figure from recorded seed-level evidence.
Only panel (b) regenerates trajectories; its per-seed metrics must match the frozen CSV.
Run: python scripts/plot_grid.py
"""
from run_icc import *
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'legend.fontsize':6.6,'xtick.labelsize':7,'ytick.labelsize':7})
sweep=pd.read_csv(RES/'V_sweep_summary.csv')
shift=pd.read_csv(RES/'shift_summary.csv')
shift_seeds=pd.read_csv(RES/'shift_seeds.csv')
belief=pd.read_csv(RES/'belief_summary.csv')
city=pd.read_csv(RES/'city_multitrajectory_summary.csv')
trajectory=pd.read_csv(RES/'city_multitrajectory_means.csv')
mismatch=pd.read_csv(RES/'mismatch_summary.csv')
cmdp=pd.read_csv(RES/'small_cmdp_summary.csv')
lam=json.loads((RES/'manifest.json').read_text())['fixed_price']
cache=RES/'shift_power_traces.npz'
policies=['FreshSky','Fixed price','Dual price','No value']
if not cache.exists():
    arrays={}
    for pol in policies:
        trajectories=[]
        for seed in SEEDS:
            r,p,_=full_episode(pol,seed,shift=True,lam=lam)
            old=shift_seeds[(shift_seeds.policy==pol)&(shift_seeds.seed==seed)].iloc[0]
            for metric in ['wAoI','max_power','psi_mJ']:
                assert np.isclose(r[metric],old[metric],rtol=1e-12,atol=1e-12),(pol,seed,metric)
            trajectories.append(p.mean(1))
        arrays[pol]=np.array(trajectories)
    np.savez_compressed(cache,**arrays)
traces=np.load(cache)
fig,axs=plt.subplots(2,3,figsize=(7.15,3.85),layout='constrained')
a=axs[0,0]; b=a.twinx()
l1=a.errorbar(sweep.V,sweep.wAoI_mean,yerr=sweep.wAoI_std,color=COL['FreshSky'],marker='o',ms=3,capsize=2)
l2=b.errorbar(sweep.V,sweep.backlog_mean,yerr=sweep.backlog_std,color='#42949E',marker='s',ms=3,capsize=2,ls='--')
a.set(xscale='log',xlabel='Control weight V',ylabel='Weighted age (slots)',title='(a) Freshness and energy debt')
b.set_ylabel('Queue sum (mW)',color='#42949E'); b.spines['right'].set_visible(True)
a.tick_params(axis='y',colors=COL['FreshSky']); b.tick_params(axis='y',colors='#42949E')
a.legend([l1,l2],['Age','Debt'],loc='upper center',ncol=2,handlelength=1.2,columnspacing=.7)
bench=cmdp[(cmdp.policy=='FreshSky')&(cmdp.V==6)].set_index('cap')
label='Small CMDP, V=6\n'+f"cap 8: {bench.loc[8,'gap_pct_mean']:.2f}% gap\ncap 12: {bench.loc[12,'gap_pct_mean']:.2f}% gap"
a.text(.35,.79,label,transform=a.transAxes,fontsize=5.9,va='top',bbox=dict(facecolor='white',edgecolor='0.75',boxstyle='round,pad=.25',alpha=.95))
a=axs[0,1]
for pol in policies:
    ma=np.array([np.convolve(x,np.ones(100)/100,'valid') for x in traces[pol]])
    tt=np.arange(99,3000)
    a.plot(tt,ma.mean(0),color=COL[pol],label=pol)
    a.fill_between(tt,ma.mean(0)-ma.std(0,ddof=1),ma.mean(0)+ma.std(0,ddof=1),color=COL[pol],alpha=.12)
a.axhline(3,color='k',ls=':',lw=.8); a.axvline(1500,color='0.5',ls=':',lw=.8)
a.set(xlabel='Slot',ylabel='Fleet mean power (mW)',title='(b) Mid-mission shift')
a.legend(ncol=2,fontsize=5.8,handlelength=1.2,columnspacing=.7,loc='upper left')
a=axs[0,2]
for x,pol in enumerate(policies):
    r=shift[shift.policy==pol].iloc[0]
    a.bar(x,r.psi_mJ_mean,yerr=r.psi_mJ_std,color=COL[pol],edgecolor='k',lw=.5,capsize=2)
a.set_xticks(range(4),['FreshSky','Fixed','Dual','No value'],rotation=20,ha='right')
a.set(ylabel='Worst UAV excess (mJ)',title='(c) Post-shift prefix excess',ylim=(0,570))
a=axs[1,0]
for pol,mk in [('FreshSky-B','o'),('Memoryless','s')]:
    q=belief[belief.policy==pol]; xs=q.setting.str.replace('dwell=','',regex=False).astype(float)
    a.errorbar(xs,q.wAoI_mean,yerr=q.wAoI_std,color=COL[pol],marker=mk,ms=3,capsize=2,label=pol)
for model,color,mk,ls in [('Model A','#42949E','^',':'),('Model B','#767676','D','--')]:
    q=mismatch[mismatch.model==model].sort_values('dwell')
    a.errorbar(q.dwell,q.wAoI_mean,yerr=q.wAoI_std,color=color,marker=mk,ms=2.7,capsize=1.4,ls=ls,lw=1.,alpha=.8,label=model)
a.set(xlabel='Mean blocked dwell (slots)',ylabel='Weighted age (slots)',title='(d) Feedback and model mismatch'); a.legend(loc='upper left',ncol=2,fontsize=5.8,handlelength=1.1,columnspacing=.6)
a=axs[1,1]; q=belief[belief.setting=='dwell=8.0'].copy().reset_index(drop=True)
for x,r in q.iterrows():
    a.bar(x-.15,r.max_power_mean,.29,yerr=r.max_power_std,color=COL[r.policy],edgecolor='k',lw=.5,capsize=2)
    a.bar(x+.15,r.mean_power_mean,.29,yerr=r.mean_power_std,color='white',hatch='///',edgecolor=COL[r.policy],lw=.7,capsize=2)
a.axhline(.4,color='k',ls=':',lw=.8)
a.set_xticks(range(4),['FreshSky-B','Memoryless','Aggregate','No queue'],rotation=25,ha='right')
a.set(ylabel='Mean power (mW)',title='(e) Individual and fleet budgets',ylim=(0,.65))
from matplotlib.patches import Patch
a.legend(handles=[Patch(facecolor='gray',edgecolor='k',label='Max. UAV'),Patch(facecolor='white',edgecolor='gray',hatch='///',label='Fleet / N')],ncol=2,fontsize=6,handlelength=1,columnspacing=.6,loc='upper left')
a=axs[1,2]
for j,pol in enumerate(['FreshSky-B','Memoryless']):
    q=city[city.policy==pol]
    for x,setting in enumerate(['midtown_manhattan','city_of_london']):
        r=q[q.setting==setting].iloc[0]
        a.bar(x+(j-.5)*.30,r.raw_wAoI_mean,.29,yerr=r.raw_wAoI_std,color=COL[pol],edgecolor='k',lw=.5,capsize=2,label=pol if x==0 else None)
        means=trajectory[(trajectory.setting==setting)&(trajectory.policy==pol)].sort_values('geometry_seed').raw_wAoI.to_numpy()
        a.scatter(x+(j-.5)*.30+np.linspace(-.045,.045,len(means)),means,s=8,facecolors='white',edgecolors='black',linewidths=.45,zorder=4)
a.set_xticks([0,1],['Manhattan','London'])
a.set(ylabel='Uncapped weighted age (slots)',title='(f) Mapped long outages',ylim=(0,float(max((city.raw_wAoI_mean+city.raw_wAoI_std).max(),trajectory.raw_wAoI.max())*1.15))); a.legend(loc='upper center',ncol=2,fontsize=5.8,handlelength=1.1,columnspacing=.6)
fig.savefig(FIG/'grid_results.pdf',bbox_inches='tight',pad_inches=.03)
fig.savefig(FIG/'grid_results.svg',bbox_inches='tight',pad_inches=.03)
fig.savefig(FIG/'grid_results.png',bbox_inches='tight',pad_inches=.03,dpi=300)
plt.close(fig)
print(json.dumps({'panels':6,'seeds':list(SEEDS),'trace_source':'per-seed shift metrics matched recorded CSV','summary_sources':['V_sweep','shift','belief','mismatch','small_cmdp','city_multitrajectory'],'city_uncertainty':'SD across five independent patrol means; dots show patrol means'}))
