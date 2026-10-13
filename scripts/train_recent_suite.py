"""Train five independent PPO models for each information regime (two CPU workers)."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,sys,json,argparse
FORCE=False
ROOT=Path(__file__).resolve().parents[1]
def task(item):
    mode,seed=item;directory=ROOT/'results'/'ppo_models';directory.mkdir(exist_ok=True)
    record=directory/f'{mode}_seed{seed}_training.json'
    if not FORCE and record.exists() and len(json.loads(record.read_text())['history'])==500:return mode,seed,'cached'
    with (directory/f'{mode}_seed{seed}.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts'/'recent_ppo.py'),'--mode',mode,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return mode,seed,'trained'
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--force',action='store_true');FORCE=parser.parse_args().force
    jobs=[(mode,seed) for seed in range(5) for mode in ['full','partial']]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(task,jobs):print(result,flush=True)
