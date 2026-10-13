"""Build Wang-style matched-feedback indices for synthetic and cached urban chains."""
from run_icc import partial_trace,ROOT
from recent_indices import BeliefIndex
import numpy as np,time
if __name__=='__main__':
    tic=time.time()
    for dwell in [2.,8.,20.]:
        for seed in range(5):
            _,_,_,ln,nl=partial_trace(seed,dwell=dwell);BeliefIndex(ln,nl)
            print('indexed',dwell,seed,'seconds',round(time.time()-tic,1),flush=True)
    for city in ['midtown_manhattan','city_of_london']:
        for geo in range(1,6):
            lo=np.load(ROOT/'data'/f'osm_patrol_{city}_geo{geo}.npz')['los'][:300]
            ln=((lo[:-1]&~lo[1:]).sum(0)+1)/(lo[:-1].sum(0)+2)
            nl=((~lo[:-1]&lo[1:]).sum(0)+1)/((~lo[:-1]).sum(0)+2)
            BeliefIndex(ln,nl);print('indexed',city,geo,'seconds',round(time.time()-tic,1),flush=True)
