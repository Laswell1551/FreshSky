import pandas as pd
from run_icc import *
rows=[]
for f in [1.,.8]:
 for seed in SEEDS:
  rows.append(full_episode('FreshSky',seed,shift=True,guard=True,reserve_frac=f)[0])
summarise(pd.DataFrame(rows),['reserve_frac'],'battery_reserve')
