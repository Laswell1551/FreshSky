"""Matched hard radio reserves: report late-mission service as well as balance."""
import json
import pandas as pd
from run_icc import *
m=json.loads((RES/'manifest.json').read_text());lam=m['fixed_price'];eta=m['tuned_dual_eta']
rows=[]
for f in [1.,.8]:
 for pol in ['FreshSky','Matched dual','Tuned dual','Fixed price','Max-weight']:
  for seed in SEEDS:rows.append(full_episode(pol,seed,shift=True,guard=True,reserve_frac=f,lam=lam,dual_eta=eta)[0])
summarise(pd.DataFrame(rows),['reserve_frac','policy'],'battery_reserve')
summarise(pd.DataFrame(rows)[lambda x:(x.reserve_frac==1)&(x.policy=='FreshSky')],['policy','guard'],'battery_guard')
rows=[]
for f in [1.,.8]:
 for seed in SEEDS:
  lo,di,pi,ln,nl=partial_trace(seed,dwell=8.)
  for pol in ['FreshSky-B','Memoryless','Belief dual','No queue']:
   r=partial_episode(pol,seed,lo,di,pi,ln,nl,radio_reserve_mJ=f*148.,label='dwell=8.0');r['reserve_frac']=f;rows.append(r)
summarise(pd.DataFrame(rows),['reserve_frac','policy'],'partial_battery')
