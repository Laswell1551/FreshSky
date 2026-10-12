"""Offline reproduction entry point. No network, GPU, or external accounts required.
Default: validate recorded numerical evidence. --full: regenerate experiments and current collage.
"""
from pathlib import Path
import sys,subprocess,argparse
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--full',action='store_true'); p.add_argument('--deep-geometry',action='store_true'); a=p.parse_args()
queue=[]
if a.full:
    queue+=['run_icc.py','battery_check.py','cmdp_sensitivity.py','small_cmdp_check.py','mismatch_check.py','city_multitrajectory.py','guard_equivalence_check.py']
queue+=['check_results.py','protocol_checks.py','check_strengthened_results.py']
if a.full: queue+=['plot_grid.py']
for script in queue:
    args=[sys.executable,str(ROOT/'scripts'/script)]
    if script=='check_strengthened_results.py' and a.deep_geometry: args+=['--deep-geometry']
    print('Running',script,flush=True); subprocess.run(args,cwd=ROOT,check=True)
    if a.full and script=='mismatch_check.py': subprocess.run(args+['--unguarded'],cwd=ROOT,check=True)
print('Completed offline numerical reproduction checks.',flush=True)
