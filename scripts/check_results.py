"""Validate seed counts and summaries; report measured comparisons without fixed outcomes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
r=Path('results');checks={}
for name in ['stationary','shift','belief','city','V_sweep','matched_pareto','battery_reserve','partial_battery','cmdp_sensitivity']:
 df=pd.read_csv(r/(name+'_seeds.csv'));su=pd.read_csv(r/(name+'_summary.csv'))
 groups={'V_sweep':['V'],'matched_pareto':['policy','V'],'battery_reserve':['reserve_frac','policy'],'partial_battery':['reserve_frac','policy'],'cmdp_sensitivity':['cmdp_age_cap','cmdp_bins']}.get(name,['setting','policy'] if 'setting' in df else ['policy'])
 assert df.groupby(groups).size().eq(5).all(),name
 for _,row in su.iterrows():
  q=df
  for g in groups:q=q[q[g]==row[g]]
  for col in su.columns:
   if col.endswith('_mean') or col.endswith('_std'):
    field,kind=col.rsplit('_',1);expected=q[field].mean() if kind=='mean' else q[field].std(ddof=1)
    assert np.isclose(expected,row[col],rtol=1e-10,atol=1e-9,equal_nan=True),(name,col)
 assert (df.max_power>=0).all() and (df.max_power<=100+1e-9).all()
 checks[name]={'rows':len(df),'seed_counts_and_summaries':'pass'}
st=pd.read_csv(r/'stationary_summary.csv').set_index('policy');b=pd.read_csv(r/'belief_summary.csv').set_index(['setting','policy']);s=pd.read_csv(r/'shift_summary.csv').set_index('policy')
checks['value_gain_pct']=float(100*(1-st.loc['FreshSky','wAoI_mean']/st.loc['No value','wAoI_mean']))
checks['belief_gain_pct']={str(d):float(100*(1-b.loc[(f'dwell={d:.1f}','FreshSky-B'),'wAoI_mean']/b.loc[(f'dwell={d:.1f}','Memoryless'),'wAoI_mean'])) for d in [2.,8.,20.]}
checks['matched_shift_excess_reduction_pct']=float(100*(1-s.loc['FreshSky','psi_mJ_mean']/s.loc['Matched dual','psi_mJ_mean']))
checks['shift_age_means']=s.wAoI_mean.to_dict()
(r/'measured_claims.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
