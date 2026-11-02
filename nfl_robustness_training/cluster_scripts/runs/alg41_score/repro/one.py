"""Re-run ONE trial of an alg41 arm in-process, stdout kept, exception re-raised.
usage (env = the arm's vars): python one.py TARGET_SEED"""
import os, sys
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
sys.path.insert(0, '/home/ryosei_t/CARV/nfl_robustness_training/src')
import numpy as np
import torch
torch.set_num_threads(1)
import run_trials_random_obs as R
target = int(sys.argv[1])
rng = np.random.default_rng(int(os.environ['TTTCARV_SEED']))
for i in range(int(os.environ['TTTCARV_TRIALS'])):
    seed = int(rng.integers(0, 2**31)); obs = R._gen_obstacles(rng)
    if seed == target:
        break
else:
    sys.exit(f'seed {target} not in the stream')
print(f'REPRO trial index {i} (Trial {i+1}) seed {seed} obstacles {obs}', flush=True)
r = R._run_trial_inner((i, seed, obs))
print('REPRO finished without exception: collision', r[9], flush=True)
