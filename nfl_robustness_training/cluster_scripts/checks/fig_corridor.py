# Corridor figure for the README: ours vs R2 on the k15 corridor scene, same
# 16 seeds, noise +-0.15. Reads the run dirs on Engaging (~/runs/alg45_corridor).
#   python fig_corridor.py <alg45_corridor dir> <out.png>
import ast, glob, json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root, out = sys.argv[1], sys.argv[2]
P = lambda s: ast.literal_eval(s) if isinstance(s, str) else s
obs = json.load(open(os.path.join(root, 'corridor_k15_d1.0.json')))
nnpath = np.load(os.path.join(root, 'nnpath.npy'))
seeds = sorted(os.path.basename(f)[2:-5] for f in glob.glob(os.path.join(root, 'k15/otr/o_*.json')))

ours = []
for s in seeds:
    steps = json.load(open(os.path.join(root, f'k15/otr/o_{s}.json')))['steps']
    ours.append(np.array([P(st['real_t']) for st in steps if P(st.get('real_t')) is not None]))
r2 = []
for s in seeds:
    tr = json.load(open(os.path.join(root, f'k15/trace/r2_{s}.json')))['trace']
    r2.append(np.array([P(st['x']) for st in tr]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
for ax, trajs, title, col in [(axes[0], ours, 'Reach-PSF (mine)', '#c0661a'),
                              (axes[1], r2, 'R2: delay-compensated tube MPC', '#2a6fdb')]:
    for o in obs:
        ax.add_patch(plt.Circle(o[:2], o[2], color='0.6', zorder=1))
    ax.plot(nnpath[:, 0], nnpath[:, 1], '--', color='k', lw=1.2, zorder=2, label='NN alone (no obstacles)')
    for i, X in enumerate(trajs):
        ax.plot(X[:, 0], X[:, 1], color=col, lw=1, alpha=0.7, zorder=3, label='filtered runs' if i == 0 else None)
    ax.plot(*trajs[0][0, :2], 'o', color='k', ms=5, zorder=4)
    ax.plot(0, 0, '*', color='gold', mec='k', ms=14, zorder=4)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
axes[0].legend(loc='lower left', fontsize=8)
axes[0].set_xlim(-10, 1.5)
axes[0].set_ylim(-1.5, 5)
fig.suptitle('Corridor longer than the MPC horizon: both filters go around it (16 runs each, noise ±0.15)', fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=130)
print('wrote', out)
