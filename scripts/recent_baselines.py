"""Four source-linked recent-baseline adaptations; no network access.
New CSVs are separate from the existing controls and share their traces.
"""
from run_icc import *
from recent_ppo import CONFIG,MODELS
RECENT=['WiSwarm','Tripathi index','Wang index','Lya-PPO']
FULL=['WiSwarm','Tripathi index','Lya-PPO']

def select_ppo():
    rows=[]
    for mode in ['full','partial']:
        for model_seed in range(5):
            candidates=[]
            for update in CONFIG['checkpoints']:
                scores=[]
                for seed in [50,51]:
                    if mode=='full':
                        r=full_episode('Lya-PPO',seed,T=1500,ppo_checkpoint=update,ppo_seed=model_seed)[0]
                        scores.append(r)
                    else:
                        for dwell in [2.,8.,20.]:
                            lo,di,pi,ln,nl=partial_trace(seed,T=1500,dwell=dwell)
                            r=partial_episode('Lya-PPO',seed,lo,di,pi,ln,nl,ppo_checkpoint=update,ppo_seed=model_seed)
                            scores.append(r)
                candidate=dict(mode=mode,model_seed=model_seed,update=update,
                    wAoI=float(np.mean([r['wAoI'] for r in scores])),
                    max_power=max(r['max_power'] for r in scores),budget=3. if mode=='full' else .4)
                candidate['feasible']=bool(candidate['max_power']<=candidate['budget'])
                candidates.append(candidate);rows.append(candidate)
                print('calibrated',candidate,flush=True)
            feasible=[r for r in candidates if r['feasible']]
            chosen=min(feasible,key=lambda r:r['wAoI']) if feasible else min(candidates,key=lambda r:(r['max_power'],r['wAoI']))
            selected=dict(chosen,selected_update=chosen['update'],selection='minimum age among every-calibration-source feasible checkpoints; else minimum excess then age',training_config=CONFIG)
            (MODELS/f'{mode}_seed{model_seed}_selection.json').write_text(json.dumps(selected,indent=2))
    pd.DataFrame(rows).to_csv(RES/'recent_ppo_calibration.csv',index=False)

def main():
    tic=time.time();select_ppo()
    rows=[]
    for policy in FULL:
        for seed in SEEDS:rows.append(full_episode(policy,seed)[0])
        print('stationary',policy,flush=True)
    summarise(pd.DataFrame(rows),['policy'],'recent_stationary')
    rows=[]
    for policy in FULL:
        for seed in SEEDS:rows.append(full_episode(policy,seed,shift=True)[0])
    summarise(pd.DataFrame(rows),['policy'],'recent_shift')
    rows=[]
    for reserve in [1.,.8]:
        for policy in FULL:
            for seed in SEEDS:rows.append(full_episode(policy,seed,shift=True,guard=True,reserve_frac=reserve)[0])
    summarise(pd.DataFrame(rows),['reserve_frac','policy'],'recent_battery')
    rows=[];guarded=[]
    for dwell in [2.,8.,20.]:
        for seed in SEEDS:
            lo,di,pi,ln,nl=partial_trace(seed,dwell=dwell)
            for policy in RECENT:
                rows.append(partial_episode(policy,seed,lo,di,pi,ln,nl,label='dwell='+str(dwell)))
                if dwell==8.:
                    for reserve in [148.,118.4]:
                        guarded.append(partial_episode(policy,seed,lo,di,pi,ln,nl,label='dwell=8.0',radio_reserve_mJ=reserve))
        print('feedback',dwell,flush=True)
    summarise(pd.DataFrame(rows),['setting','policy'],'recent_belief')
    summarise(pd.DataFrame(guarded),['radio_reserve_mJ','policy'],'recent_partial_battery')
    rows=[]
    for city in ['midtown_manhattan','city_of_london']:
        for geo in range(1,6):
            file=ROOT/'data'/f'osm_patrol_{city}_geo{geo}.npz';z=np.load(file);lo,di=z['los'],z['dist'];train=lo[:300]
            pi=np.clip(train.mean(0),.05,.95)
            ln=((train[:-1]&~train[1:]).sum(0)+1)/(train[:-1].sum(0)+2)
            nl=((~train[:-1]&train[1:]).sum(0)+1)/((~train[:-1]).sum(0)+2)
            for seed in SEEDS:
                for policy in RECENT:
                    r=partial_episode(policy,seed,lo,di,pi,ln,nl,pbar=1.6,label=city,radio_reserve_mJ=272.)
                    r.update(geometry_seed=geo);rows.append(r)
            print('urban',city,geo,flush=True)
    df=pd.DataFrame(rows);df.to_csv(RES/'recent_city_seeds.csv',index=False)
    cols=['wAoI','raw_wAoI','saturation','max_power','mean_power','psi_mJ','battery_min_mJ','guard_rejections']
    means=df.groupby(['setting','policy','geometry_seed'],sort=False)[cols].mean().reset_index()
    means.to_csv(RES/'recent_city_means.csv',index=False)
    q=means.groupby(['setting','policy'],sort=False)[cols].agg(['mean','std']);q.columns=['_'.join(x) for x in q.columns]
    q.reset_index().to_csv(RES/'recent_city_summary.csv',index=False)
    (RES/'recent_baselines_manifest.json').write_text(json.dumps(dict(protocol='recent-v1',
        main_seeds=list(SEEDS),calibration_seeds=[50,51],methods=RECENT,full_information_methods=FULL,
        full_information_success='known threshold; reliability one on eligible links',
        partial_likelihoods=[.9,.02],wiswarm_window=20,wiswarm_probability_floor=.05,
        tripathi_probability='frozen calibrated invariant marginal; current weight multiplies capped-cost index',
        wang_approximation=dict(cap=120,belief_points=41,charge_points=97,discount=.999,
          noisy_feedback=True,weight='current multiplier',indexability='not claimed for adapted model; nested sampled price sets verified'),
        learning='Long-style scheduling-only PPO adaptation, power debt replacing age-constraint queue; fixed paths/no semantic compression; analytical prescribed-power block',
        ppo_training_config=CONFIG,common_reserves=True,urban_replicates_per_policy_city=25,
        seconds=time.time()-tic),indent=2))
    print('FINISHED RECENT BASELINES',flush=True)
if __name__=='__main__':main()
