"""Did each trial fly through the corridor? entered = came within 0.6 of the NN path inside k=15..33;
through = also came within 0.6 of the exit point P[33]. Ours from RSOA dumps (real_t), R2 from GP traces (x)."""
import json, glob, sys, numpy as np
P = np.load('/home/ryosei_t/runs/alg45_corridor/nnpath.npy'); seg, ex = P[15:34, :2], P[33, :2]
for pat, key in ((sys.argv[1] + '/otr/o_*.json', 'o'), (sys.argv[1] + '/trace/r2_*.json', 'r2')):
    fs = sorted(glob.glob(pat)); ent = thr = 0
    for f in fs:
        d = json.load(open(f))
        X = np.array([s['real_t'][:2] for s in d['steps'] if s.get('real_t')] if key == 'o' else [s['x'][:2] for s in d['trace']])
        dm = np.array([np.linalg.norm(seg - x, axis=1).min() for x in X])
        ent += (dm < 0.6).any(); thr += (dm < 0.6).any() and np.linalg.norm(X - ex, axis=1).min() < 0.6
    print(f'{key:3s} {pat}: trials {len(fs)} entered {ent} through {thr}')
