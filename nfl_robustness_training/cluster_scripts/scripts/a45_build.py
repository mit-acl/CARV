"""Corridor scene: disk pairs at NN-path points k0, k0+3, ..., k0+18, offset +-d along the path normal."""
import json, sys, numpy as np
k0, d = int(sys.argv[1]), float(sys.argv[2])
P = np.load('/home/ryosei_t/runs/alg45_corridor/nnpath.npy')
obs = []
for k in range(k0, k0 + 19, 3):
    th = np.arctan2(P[k + 1, 1] - P[k - 1, 1], P[k + 1, 0] - P[k - 1, 0])
    n = np.array([-np.sin(th), np.cos(th)])
    for s in (1, -1):
        c = P[k, :2] + s * d * n
        obs.append([float(c[0]), float(c[1]), 0.5])
out = f'/home/ryosei_t/runs/alg45_corridor/corridor_k{k0}_d{d}.json'
json.dump(obs, open(out, 'w'))
print(out, len(obs), 'path k0', np.round(P[k0], 2), 'exit', np.round(P[k0 + 18], 2))
