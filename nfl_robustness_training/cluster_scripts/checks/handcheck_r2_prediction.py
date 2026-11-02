# Hand check of R2's delay-compensation step (the "push the state set forward
# through the command already running" part). Pulls _plant_step,
# _interval_step and _cos_range straight out of alg29_generic_psf.py, so it
# needs only numpy and runs fine on the login node.
#
#   python handcheck_r2_prediction.py ../../src/alg29_generic_psf.py
import ast
import sys

import numpy as np

src = open(sys.argv[1]).read()
ns = {'np': np}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ('_cos_range', '_interval_step', '_plant_step'):
        exec(compile(ast.Module([node], []), 'alg29', 'exec'), ns)
P, I = ns['_plant_step'], ns['_interval_step']
dt, v = 0.2, 1.0

print('1. point step vs hand calc')
for x, u, want in [((0, 0, 0), 1, (0.2, 0, 0.2)),
                   ((1, 2, np.pi / 2), -1, (1, 2.2, np.pi / 2 - 0.2)),
                   ((0, 0, np.pi), 0.5, (-0.2, 0, np.pi + 0.1))]:
    got = P(np.array(x, float), u * dt, dt, v)
    print('  ', x, 'u =', u, '->', np.round(got, 6), 'OK' if np.allclose(got, want) else 'FAIL')

print('2. zero-width box steps exactly like a point')
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(10000):
    x = rng.uniform([-10, -10, -7], [10, 10, 7])
    u = rng.uniform(-1, 1)
    b = I(np.stack([x, x], 1), u * dt, dt, v)
    worst = max(worst, np.abs(b[:, 0] - P(x, u * dt, dt, v)).max(), np.abs(b[:, 1] - b[:, 0]).max())
print('   max difference =', worst)

print('3. box vs hand calc: x, y, heading all in +-0.15, u = 0')
b = I(np.array([[-.15, .15]] * 3), 0.0, dt, v)
hand = np.array([[-.15 + .2 * np.cos(.15), .35],
                 [-.15 - .2 * np.sin(.15), .15 + .2 * np.sin(.15)],
                 [-.15, .15]])
print('   got ', np.round(b, 5).tolist())
print('   hand', np.round(hand, 5).tolist(), 'OK' if np.allclose(b, hand) else 'FAIL')

print('4. random states inside random boxes stay inside the stepped box')
miss = n = 0
for _ in range(20000):
    c = rng.uniform([-5, -5, -7], [5, 5, 7])
    hw = rng.uniform(0, [0.3, 0.3, 1.5])
    S = np.stack([c - hw, c + hw], 1)
    u = rng.uniform(-1, 1)
    B = I(S, u * dt, dt, v)
    xs = rng.uniform(S[:, 0], S[:, 1], size=(50, 3))
    xs[:3, 2] = [S[2, 0], S[2, 1], c[2]]          # include the heading edges
    for x in xs:
        y = P(x, u * dt, dt, v)
        n += 1
        miss += bool(np.any(y < B[:, 0] - 1e-12) or np.any(y > B[:, 1] + 1e-12))
print(f'   {miss} misses out of {n}')
