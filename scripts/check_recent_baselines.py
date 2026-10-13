"""Independent equation, LP, training-split, energy and summary checks."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,eye
from run_icc import full_episode,partial_episode,partial_trace
from recent_indices import wiswarm_priority,tripathi_index,BeliefIndex
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results'

def equations():
    assert np.allclose(wiswarm_priority([1,5,10],[1,5,1],[1,.2,.1]),[1,25,10])
    for p in [.2,.7,1.]:
        for cap in [8,120]:
            for age in [1,5,cap]:
                k=np.arange(1,10001,dtype=float)
                ref=p*p*age*np.sum(np.minimum(age+k,cap)*(1-p)**(k-1))-p*np.sum(np.minimum(np.arange(1,age+1),cap))
                assert np.isclose(tripathi_index(age,1,p,cap),ref,atol=1e-9)
    z=BeliefIndex([.5],[.5],qL=1,qN=0)
    ages=np.arange(1,21);got=z.priority(ages[:1],[.5],[1])
    table=np.array([np.interp(.5,z.grid,z.tables[0][a-1]) for a in ages]);ref=tripathi_index(ages,1,.5,120)
    iid_error=float(np.max(np.abs(table-ref)/ref));assert iid_error<.02
    return iid_error

def lp_reference():
    # Independent discounted Bellman LP for a small noisy belief grid.
    cap=8;B=11;belief=np.linspace(0,1,B);S=cap*B;discount=.999
    ln=.2;nl=.125;qL=.9;qN=.02;prices=[0.,3.,12.,40.]
    index=BeliefIndex([ln],[nl],cap=cap,belief_points=B,charge_points=97)
    cert=np.load(index.paths[0]);maxres=0.;checks=0
    for ci in [0,30,60,96]:
        charge=float(cert['charges'][ci]);r0=[];c0=[];w0=[];r1=[];c1=[];w1=[]
        for a in range(cap):
            for b,th in enumerate(belief):
                row=a*B+b;inc=min(a+1,cap-1);success=th*qL+(1-th)*qN
                def append(rows,cols,weights,val,newage,mass):
                    j=min(int(np.clip(val,0,1)*(B-1)),B-2);fraction=val*(B-1)-j
                    rows.extend([row,row]);cols.extend([newage*B+j,newage*B+j+1]);weights.extend([mass*(1-fraction),mass*fraction])
                pred=lambda x:x*(1-ln)+(1-x)*nl
                append(r0,c0,w0,pred(th),inc,1.)
                ack=th*qL/success;nack=th*(1-qL)/(1-success)
                append(r1,c1,w1,pred(ack),0,success);append(r1,c1,w1,pred(nack),inc,1-success)
        P0=coo_matrix((w0,(r0,c0)),shape=(S,S)).tocsr();P1=coo_matrix((w1,(r1,c1)),shape=(S,S)).tocsr()
        age=np.repeat(np.arange(1,cap+1),B);I=eye(S,format='csr')
        from scipy.sparse import vstack
        fit=linprog(-np.ones(S),A_ub=vstack([I-discount*P0,I-discount*P1]),b_ub=np.r_[age,age+charge],bounds=[(None,None)]*S,method='highs')
        assert fit.success;v=fit.x;q0=age+discount*(P0@v);q1=age+charge+discount*(P1@v)
        err=float(np.max(np.abs(v-np.minimum(q0,q1))));assert err<2e-6
        active=(q1<q0-1e-6).reshape(cap,B)
        recorded=np.unpackbits(cert['active_sets'],axis=0,count=len(cert['charges']))[ci].astype(bool)
        # LP tie tolerance: permit only genuinely indifferent actions.
        disagreement=active!=recorded
        assert not np.any(disagreement&(np.abs((q1-q0).reshape(cap,B))>2e-6))
        maxres=max(maxres,err);checks+=1
    return dict(independent_LP_prices=checks,bellman_residual=maxres)

def summaries():
    total=0
    specs=[('recent_stationary',['policy'],15),('recent_shift',['policy'],15),
        ('recent_battery',['reserve_frac','policy'],30),('recent_belief',['setting','policy'],60),
        ('recent_partial_battery',['radio_reserve_mJ','policy'],40)]
    for name,groups,count in specs:
        df=pd.read_csv(RES/(name+'_seeds.csv'));sm=pd.read_csv(RES/(name+'_summary.csv'));assert len(df)==count
        assert df.peak_instant_power_mW.max()<=100.+1e-8
        assert set(df.seed)==set(range(5));assert not df.duplicated(groups+['seed']).any()
        for _,row in sm.iterrows():
            m=np.ones(len(df),bool)
            for g in groups:m&=df[g]==row[g]
            selected=df[m];assert len(selected)==5
            for metric in ['wAoI','raw_wAoI','max_power','mean_power','psi_mJ']:
                assert np.isclose(row[metric+'_mean'],selected[metric].mean(),rtol=1e-10,atol=1e-10)
                assert np.isclose(row[metric+'_std'],selected[metric].std(ddof=1),rtol=1e-10,atol=1e-10)
        if name in ['recent_battery','recent_partial_battery']:
            assert df.battery_min_mJ.min()>=-1e-8
            assert (df.radio_spent_max_mJ<= (df.reserve_frac*900 if name=='recent_battery' else df.radio_reserve_mJ)+1e-8).all()
        total+=len(df)
    df=pd.read_csv(RES/'recent_city_seeds.csv');means=pd.read_csv(RES/'recent_city_means.csv');sm=pd.read_csv(RES/'recent_city_summary.csv')
    assert len(df)==200 and len(means)==40 and len(sm)==8
    assert df.peak_instant_power_mW.max()<=100.+1e-8
    assert df.battery_min_mJ.min()>=0 and df.radio_spent_max_mJ.max()<=272+1e-8
    for _,row in sm.iterrows():
        part=means[(means.setting==row.setting)&(means.policy==row.policy)];assert len(part)==5
        for metric in ['wAoI','raw_wAoI','max_power','mean_power','saturation']:
            assert np.isclose(row[metric+'_mean'],part[metric].mean(),rtol=1e-10)
            assert np.isclose(row[metric+'_std'],part[metric].std(ddof=1),rtol=1e-10)
    # Existing trace outputs remain identical after the interface extension.
    expected=pd.read_csv(RES/'stationary_seeds.csv').query('policy=="FreshSky" and seed==0').iloc[0]
    got=full_episode('FreshSky',0)[0]
    for metric in ['wAoI','raw_wAoI','max_power','mean_power','psi_mJ']:assert np.isclose(got[metric],expected[metric],rtol=1e-12)
    lo,di,pi,ln,nl=partial_trace(0,dwell=8.)
    got=partial_episode('FreshSky-B',0,lo,di,pi,ln,nl,label='dwell=8.0')
    expected=pd.read_csv(RES/'belief_seeds.csv').query('policy=="FreshSky-B" and seed==0 and setting=="dwell=8.0"').iloc[0]
    for metric in ['wAoI','raw_wAoI','max_power','mean_power','psi_mJ']:assert np.isclose(got[metric],expected[metric],rtol=1e-12)
    return total+len(df)

def learning():
    from recent_ppo import PPOController
    choices=pd.read_csv(RES/'recent_ppo_calibration.csv');assert len(choices)==30
    records=[]
    for mode in ['full','partial']:
        for seed in range(5):
            base=RES/'ppo_models';record=json.loads((base/f'{mode}_seed{seed}_training.json').read_text())
            cfg=record['config'];history=record['history'];assert len(history)==500 and history[-1]['samples']==1024000
            assert set(cfg['train_trace_seeds']).isdisjoint(set(range(5))|{50,51})
            assert cfg['calibration_seeds']==[50,51] and cfg['training_seeds']==list(range(5))
            assert np.isfinite([[r['reward'],r['loss'],r['approx_kl']] for r in history]).all()
            select=json.loads((base/f'{mode}_seed{seed}_selection.json').read_text())
            part=choices[(choices['mode']==mode)&(choices.model_seed==seed)]
            feasible=part[part.feasible]
            chosen=feasible.sort_values('wAoI').iloc[0] if len(feasible) else part.sort_values(['max_power','wAoI']).iloc[0]
            assert select['selected_update']==chosen['update']
            actor=PPOController(mode,seed)
            for eligible_count in [0,1,2,3]:
                allowed=np.arange(12)<eligible_count
                selected=actor.decide(np.ones(12)*20,np.ones(12)*10,np.ones(12),np.ones(12),np.zeros(12),200,500,3,allowed)
                assert len(selected)<=eligible_count and np.all(allowed[selected])
            hashes=[]
            for update in cfg['checkpoints']:
                file=base/f'{mode}_seed{seed}_update{update}.npz';z=np.load(file)
                assert set(z.files)=={f'{head}.{layer}.{kind}' for head in ['actor','idle','critic'] for layer in [0,2,4] for kind in ['weight','bias']}
                assert all(np.isfinite(z[n]).all() for n in z.files)
                hashes.append(hashlib.sha256(file.read_bytes()).hexdigest())
            assert len(set(hashes))==3
            records.append(dict(mode=mode,seed=seed,selected_update=select['selected_update'],calibration_feasible=bool(select['feasible'])))
    return records

def main():
    iid=equations();lp=lp_reference();count=summaries();trained=learning();paths=[];maxres=0.;nested=0
    for file in (RES/'belief_index_policies').glob('*.npz'):
        z=np.load(file);spec=json.loads(str(z['spec_json']))
        if spec.get('protocol')!='noisy-discounted-index-v2':continue
        residual=float(z['bellman_residual'].max());assert residual<2e-6;maxres=max(maxres,residual)
        active=np.unpackbits(z['active_sets'],axis=0,count=len(z['charges'])).astype(bool)
        violations=int(np.sum(active[1:]&~active[:-1]));assert violations==0;nested+=violations
        assert np.all((z['lower_difference']<2e-6)|((z['lower_charge']==0)&(z['index']==0))) and z['upper_difference'].min()>-2e-6
        assert np.all(z['index']>=z['lower_charge']-1e-8) and np.all(z['index']<=z['upper_charge']+1e-8)
        paths.append(file)
    report=dict(status='passed',seed_records_checked=count,source_equation_fixtures='passed',
        iid_first20_discounted_index_max_relative_error=iid,independent_small_belief_LP=lp,
        index_certificates=len(paths),largest_discounted_bellman_residual=maxres,
        nested_sampled_charge_sets_violations=nested,adapted_indexability='not inferred from a sampled check',
        original_control_outputs_unchanged=True,learning_models=trained,
        all_shared_reserve_balances='nonnegative',urban_uncertainty_unit='five geometry means')
    (RES/'recent_baselines_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
