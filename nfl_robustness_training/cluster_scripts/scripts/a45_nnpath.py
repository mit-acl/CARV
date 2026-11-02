import sys, numpy as np, torch
sys.path.insert(0, '/home/ryosei_t/CARV/nfl_robustness_training/src')
from REAL_integrated_sim import setup_analyzer
an = setup_analyzer('Unicycle_NL', 'natural_none_default')
cl = an.cl_system
x = np.array([-9.5, 3.5, 0.0])
P = [x.copy()]
for k in range(60):
    xt = torch.tensor(x, dtype=torch.float32).reshape(1, -1)
    u = cl.controller(xt).detach().cpu().numpy().flatten()
    x = np.array([x[0] + 0.2*np.cos(x[2]), x[1] + 0.2*np.sin(x[2]), x[2] + 0.2*float(u[0])])
    P.append(x.copy())
P = np.array(P)
np.save('/home/ryosei_t/runs/alg45_corridor/nnpath.npy', P)
for k in range(0, 61, 3):
    print(k, np.round(P[k], 3))
