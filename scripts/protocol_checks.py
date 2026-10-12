"""Independent checks of matched comparisons, priors, units and CMDP flow."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_icc import partial_trace
import env_and_cmdp as F
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';checks={}
m=json.loads((RES/'manifest.json').read_text());assert set(m['seeds']).isdisjoint(m['calibration_seeds'])
assert m['calibration_seeds']==[50,51]
priors=[]
for d in [2.,8.,20.]:
 for seed in range(5):
  lo,di,pi,ln,nl=partial_trace(seed,T=1000,dwell=d)
  error=float(np.max(abs(pi*(1-ln)+(1-pi)*nl-pi)));assert error<1e-13
  priors.append(dict(dwell=d,seed=seed,invariant_error=error,min_prior=float(pi.min()),max_prior=float(pi.max())))
checks['actual_invariant_priors']=priors
speeds=[]
for seed in range(5):
 c=F.clone(F.Cfg(),seed=seed);e=F.Env(c);rr=[]
 for t in range(c.T):
  e.t=t;rr.append([e._rh(n) for n in range(c.N)])
 measured=float(np.max(abs(np.diff(rr,axis=0)))/c.tau);assert measured<=14.+1e-6
 speeds.append(dict(seed=seed,max_radial_speed_mps=measured))
checks['synthetic_speed']=speeds
for n in ['stationary','shift','matched_pareto']:
 df=pd.read_csv(RES/(n+'_seeds.csv'));mdu=df[df.policy=='Matched dual']
 assert np.allclose(mdu.dual_eta,1/mdu.V) and (mdu.dual_initial_price==0).all()
checks['matched_dual_step_and_initialization']='pass'
# Verify saved occupation measures without relying on the solver certificates.
certs=[]
for p in sorted((RES/'weighted_cmdp_policies').glob('*.npz')):
 z=np.load(p);meta=json.loads(str(z['metadata']));mu=z['occupancy_mu'];y=z['occupancy_active'];P=z['channel_transition'];omega=z['upper_power'];N,X,Q,_=mu.shape
 off=1/25;on=off*.08/.92;E=np.array([[1-on,on],[off,1-off]])
 flow=0.;cost=0.;powers=[]
 for n in range(N):
  passive=np.einsum('aqe,qr,ef->arf',mu[n]-y[n],P[n],E)
  active=np.einsum('qe,qr,ef->rf',y[n].sum(0),P[n],E)
  inc=np.zeros_like(mu[n]);inc[1:]+=passive[:-1];inc[-1]+=passive[-1];inc[0]+=active
  flow=max(flow,float(abs(inc-mu[n]).max()));assert abs(mu[n].sum()-1)<1e-6
  assert (y[n]>=-1e-8).all() and (y[n]<=mu[n]+1e-7).all()
  powers.append(float((y[n]*omega[n][None,:,None]).sum()))
  a=np.minimum(np.arange(1,X+1)+1,X)[:,None,None];w=np.array([1.,10.])[None,None,:]
  cost+=float((w*(a*mu[n]+(1-a)*y[n])).sum())
 assert flow<1e-6 and max(powers)<=3*meta['budget_scale']+1e-6 and y.sum()<=3+1e-6
 assert abs(cost-meta['relaxed_total_cost'])<1e-5
 certs.append(dict(file=p.name,flow_error=flow,objective=cost,max_nominal_power=max(powers),activation=float(y.sum())))
assert len(certs)>=7;checks['independent_weighted_cmdp_certificates']=certs
for n in ['battery_reserve','partial_battery']:
 df=pd.read_csv(RES/(n+'_seeds.csv'));assert df.battery_min_mJ.min()>=-1e-8
 assert df.tail_served_fraction.between(0,1).all() and df.tail_delivery_rate.between(0,1).all()
 assert df.groupby(['reserve_frac','policy']).size().eq(5).all()
 assert (df.radio_spent_max_mJ<=df.reserve_frac*(900. if n=='battery_reserve' else 148.)+1e-7).all()
checks['common_reserves_and_late_service']='pass'
# Adversarial positive-score service, including the projection boundary.
rng=np.random.default_rng(765);G=6*5*119;allow=.4;cap=G/allow+100
for form in ['deduct_then_spend','projected_dual']:
 q=0.;history=[q];energy=[]
 for _ in range(20000):
  reward=rng.uniform(0,G);p=10**rng.uniform(-10,2);spent=p if reward-q*p>0 else 0.
  q=max(q-allow,0)+spent if form=='deduct_then_spend' else max(q+spent-allow,0)
  assert q<=cap+1e-8;history.append(q);energy.append(spent)
 csum=np.r_[0,np.cumsum(energy)]
 for _ in range(1000):
  a=int(rng.integers(0,20000));b=int(rng.integers(a+1,20001))
  assert csum[b]-csum[a] <=(b-a)*allow+history[b]-history[a]+1e-7
checks['both_queue_updates_and_window_endpoints']='pass'
(RES/'protocol_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
