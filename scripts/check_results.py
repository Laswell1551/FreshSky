import pandas as pd,json
from pathlib import Path
r=Path('results'); st=pd.read_csv(r/'stationary_summary.csv').set_index('policy')
b=pd.read_csv(r/'belief_summary.csv').set_index(['setting','policy'])
s=pd.read_csv(r/'shift_summary.csv').set_index('policy')
a=100*(1-st.loc['FreshSky','wAoI_mean']/st.loc['No value','wAoI_mean'])
g=100*(1-b.loc[('dwell=8.0','FreshSky-B'),'wAoI_mean']/b.loc[('dwell=8.0','Memoryless'),'wAoI_mean'])
p=100*(1-s.loc['FreshSky','psi_mJ_mean']/s.loc['Dual price','psi_mJ_mean'])
assert round(a,1)==9.3 and round(g,1)==13.9
assert 32<p<34
for name in ['stationary','shift','belief','city']:
 df=pd.read_csv(r/(name+'_seeds.csv')); groups=['policy']+(['setting'] if 'setting' in df else [])
 assert (df.groupby(groups).size()==5).all()
 q=df[df.policy.isin(['FreshSky','FreshSky-B'])]
 assert (q.max_power<=q.budget+1e-9).all()
print(json.dumps({'value_gain_pct':a,'belief_dwell8_gain_pct':g,'postshift_excess_reduction_pct':p,'seed_and_budget_checks':'pass'}))
