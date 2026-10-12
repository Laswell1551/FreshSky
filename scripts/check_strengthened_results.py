"""Validate strengthened claims, including an independent Bellman dual check.
Run from artifact root. This reads recorded experiments; --deep-geometry re-ray-marches all primary paths.
"""
from pathlib import Path
import json,itertools,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'; checks={}
def dual_check():
    m=json.loads((RES/'small_cmdp_manifest.json').read_text()); cb=np.array([[.875,.125],[.125,.875]])
    on=.04*.08/.92; ce=np.array([[1-on,on],[.04,.96]]); ex=np.kron(np.kron(cb,cb),np.kron(ce,ce)); result=[]
    for cert in m['certificates']:
        cap=cert['cap']; z=np.load(RES/f'small_cmdp_cap{cap}.npz'); occ=z['occupancy']; y=z['dual_flow'].reshape(cap,cap,16); eta=float(z['dual_normalization']); mu=z['dual_energy']; power=np.array(m['powers']); budget=np.array(m['budgets'])
        mass=occ.sum(1); assert abs(mass.sum()-1)<1e-8 and np.min(mu)<=1e-8 and np.max(mu)<=1e-8
        incoming=np.zeros_like(mass); obj=0.; spent=np.zeros(2); min_slack=np.inf
        for a,b,ch,ev in itertools.product(range(1,cap+1),range(1,cap+1),range(4),range(4)):
            sid=((a-1)*cap+b-1)*16+4*ch+ev; w=np.array([5 if ev&1 else 1,5 if ev&2 else 1]); next_age=[min(a+1,cap),min(b+1,cap)]
            for action in range(3):
                branches=[(1.,next_age)]; p=np.zeros(2)
                if action:
                    n=action-1; p[n]=power[n]; prob=.02 if ch&(1<<n) else .90; reset=next_age.copy(); reset[n]=1; branches=[(1-prob,next_age),(prob,reset)]
                expected_y=0.; cost=0.; weight=occ[sid,action]
                for prob,aa in branches:
                    expected_y+=prob*float(ex[4*ch+ev]@y[aa[0]-1,aa[1]-1]); cost+=prob*float(w@aa)
                    j=((aa[0]-1)*cap+aa[1]-1)*16; incoming[j:j+16]+=weight*prob*ex[4*ch+ev]
                slack=cost-y[a-1,b-1,4*ch+ev]+expected_y-eta-float(mu@p); min_slack=min(min_slack,slack); obj+=weight*cost; spent+=weight*p
        dual=eta+float(mu@budget); flow=np.max(np.abs(mass-incoming))
        assert min_slack>=-1e-7 and flow<1e-7 and abs(obj-dual)<1e-7 and np.all(spent<=budget+1e-8)
        result.append(dict(cap=cap,objective=obj,dual_lower_bound=dual,min_bellman_slack=float(min_slack),stationary_flow_error=float(flow)))
    return result
checks['independent_cmdp_bellman_certificate']=dual_check()
b=pd.read_csv(RES/'small_cmdp_seeds.csv'); assert len(b)==60 and b.groupby(['cap','policy','V']).size().eq(5).all()
g=b[(b.policy=='FreshSky')&(b.V==6)]; assert (g.max_power_ratio<=1+1e-10).all()
checks['small_cmdp_default_V6_gap_means_pct']=g.groupby('cap').gap_pct.mean().to_dict()
r=pd.read_csv(RES/'mismatch_seeds.csv'); assert len(r)==45 and r.groupby(['model','dwell']).size().eq(5).all()
assert (r.battery_min_mJ>=0).all() and (r.max_power<=.4+1e-10).all()
checks['mismatch_guarded_max_power_mW']=r.max_power.max(); checks['mismatch_min_radio_balance_mJ']=r.battery_min_mJ.min()
old=pd.read_csv(RES/'mismatch_seeds_unguarded.csv'); checks['unguarded_mismatch_max_power_mW']=old.max_power.max()
c=pd.read_csv(RES/'city_multitrajectory_seeds.csv'); assert len(c)==100 and c.groupby(['setting','policy','geometry_seed']).size().eq(5).all()
assert (c.battery_min_mJ>=0).all() and (c.max_power<=1.6+1e-10).all()
tm=pd.read_csv(RES/'city_multitrajectory_means.csv'); expected=c.groupby(['setting','policy','geometry_seed'],sort=False).wAoI.mean().to_numpy(); assert np.allclose(expected,tm.wAoI)
checks['city_independent_geometry_seeds_per_city']=5; checks['paired_event_delivery_seeds_per_geometry']=5
# Reconstruct maps and independently check all primary path samples for geometric feasibility.
import geometry_builder as G
import city_multitrajectory as C
route_checks=[]
for city in ['midtown_manhattan','city_of_london']:
    G.MINLAT,G.MINLON,G.MAXLAT,G.MAXLON=(40.7485,-73.9911,40.7593,-73.9769) if city=='midtown_manhattan' else (51.5100,-.0900,51.5200,-.0740)
    raw=json.loads((ROOT/'data'/f'osm_{city}.json').read_text()); polys,heights,_=G.parse_buildings(raw); H,cell,sx,sy,_=G.build_grid(polys,heights); gs,_=G.place_gs(H,cell)
    for seed in range(1,6):
        z=np.load(ROOT/'data'/f'osm_patrol_{city}_geo{seed}.npz'); pos=z['positions']; ix=(pos[:,:,0]//cell).astype(int); iy=(pos[:,:,1]//cell).astype(int)
        assert np.all((pos[:,:,0]>=0)&(pos[:,:,0]<sx)&(pos[:,:,1]>=0)&(pos[:,:,1]<sy))
        clearance=pos[:,:,2]-H[ix,iy]; assert clearance.min()>=5
        speed=np.linalg.norm(np.diff(pos,axis=0),axis=2)/.1; assert np.all(speed<=z['speeds'][None,:]+1e-7) and np.max(z['speeds'])<=14
        assert np.allclose(np.linalg.norm(pos-gs,axis=2),z['dist'])
        if '--deep-geometry' in sys.argv:
            lo,di=C.traces(H,cell,gs,pos); assert np.array_equal(lo,z['los']) and np.allclose(di,z['dist'])
        route_checks.append(dict(city=city,seed=seed,min_clearance_m=float(clearance.min()),max_sample_speed_mps=float(speed.max())))
checks['patrol_clearance_and_speed']=route_checks; checks['deep_geometry']='passed' if '--deep-geometry' in sys.argv else 'not requested'
checks['corrected_protocol']='v1.1.0; current comparisons recorded by check_results.py'
(RES/'strengthened_checks.json').write_text(json.dumps(checks,indent=2)); print(json.dumps(checks,indent=2))
