"""Discretization sensitivity at the independently selected default budget scale."""
import json
import pandas as pd
from run_icc import *
m=json.loads((RES/'manifest.json').read_text());scale=m['weighted_cmdp_budget_scale'];rows=[]
for cap,bins in [(40,4),(80,4),(80,8)]:
 for seed in SEEDS:
  r=full_episode('Weighted CMDP',seed,cmdp_budget_scale=scale,cmdp_age_cap=cap,cmdp_bins=bins)[0]
  r.update(cmdp_age_cap=cap,cmdp_bins=bins);rows.append(r)
summarise(pd.DataFrame(rows),['cmdp_age_cap','cmdp_bins'],'cmdp_sensitivity')
