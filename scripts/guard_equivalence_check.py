"""Check that the nominal feedback reserve is nonbinding for Memoryless."""
from run_icc import *
frozen=pd.read_csv(RES/'belief_seeds.csv');records=[]
for dwell in [2.,8.,20.]:
 for seed in SEEDS:
  lo,di,pi,ln,nl=partial_trace(seed,dwell=dwell)
  q=partial_episode('Memoryless',seed,lo,di,pi,ln,nl,radio_reserve_mJ=148.,label='dwell='+str(dwell))
  old=frozen[(frozen.policy=='Memoryless')&(frozen.seed==seed)&(frozen.setting=='dwell='+str(dwell))].iloc[0]
  for c in ['wAoI','raw_wAoI','max_power']:assert np.isclose(q[c],old[c],rtol=1e-12,atol=1e-12)
  assert q['guard_rejections']==0;records.append(q)
(RES/'memoryless_guard_equivalence.json').write_text(json.dumps(dict(version='1.1.0',paired_runs=len(records),radio_reserve_mJ=148,metrics_match_current_memoryless=True,guard_rejections=0),indent=2));print('15 current-prior Memoryless guarded comparisons passed')
