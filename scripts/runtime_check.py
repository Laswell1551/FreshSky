import time, json, platform
from pathlib import Path
import numpy as np
rng=np.random.default_rng(803); rows=[]
for N in [12,100,500,2000]:
 A=rng.integers(1,500,N).astype(float); Q=np.zeros(N); w=rng.choice([1.,10.],N)
 c=rng.uniform(.01,120,N); M=max(1,N//4); times=[]
 for repeat in range(5):
  tic=time.perf_counter_ns()
  for k in range(2000):
   score=np.where(c<=100,200*w*(np.minimum(A+1,500)-1)-Q*c,-np.inf)
   sel=np.argsort(-score,kind='stable')[:M]; sel=sel[score[sel]>0]
   p=np.zeros(N); p[sel]=c[sel]; Q=np.maximum(Q-3,0)+p
   A=np.minimum(A+1,500); A[sel]=1
  times.append((time.perf_counter_ns()-tic)/2000/1000)
 rows.append({'N':N,'M':M,'microseconds_mean':float(np.mean(times)),'microseconds_sd':float(np.std(times,ddof=1))})
x={'processor':platform.processor(),'scope':'scores, stable full sort, grants, age/queue update; excludes channel generation, I/O and feedback transport','repeats':5,'slots_per_repeat':2000,'rows':rows}
Path('results/runtime.json').write_text(json.dumps(x,indent=2),encoding='utf8'); print(json.dumps(x))
