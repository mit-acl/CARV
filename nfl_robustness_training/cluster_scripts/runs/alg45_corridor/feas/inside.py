"""For each NN-path state inside the corridor: does a max-turn loiter (point state) clear the raw disks?"""
import os, sys, json
sys.argv = sys.argv[:1] + ['[]']
exec(open('feas.py').read().split('starts = json.loads')[0])
import numpy as np
P = np.load('/home/ryosei_t/runs/alg45_corridor/nnpath.npy')
print(os.environ['OBS'], 'loiter-ok along path k=12..36:', ''.join('1' if loiter_ok(tuple(P[k])) else '.' for k in range(12, 37)))
