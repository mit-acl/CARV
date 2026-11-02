"""Optimistic feasibility of an R2 plan from the start: point state (no uncertainty box, no margin).
Path must clear raw disks; terminal must admit a max-turn loiter (33 Euler steps) clear of raw disks.
Piecewise-constant controls, up to 3 segments, u in {-1,-.5,0,.5,1}. If this finds nothing, the
real R2 (boxes + margins) cannot be feasible either."""
import json, math, itertools, sys
import os
obs = json.load(open(os.environ.get('OBS', '/home/ryosei_t/runs/alg45_corridor/corridor_d1.0.json')))
obs = [(o[0], o[1], o[2]) if isinstance(o, list) else (o['x'], o['y'], o['r']) for o in obs]
dt, v = 0.2, 1.0
def clear(x, y): return min(math.hypot(x - a, y - b) - r for a, b, r in obs)
def step(s, u): x, y, th = s; return (x + v * dt * math.cos(th), y + v * dt * math.sin(th), th + u * dt)
def loiter_ok(s):
    for u in (1.0, -1.0):
        z, ok = s, True
        for _ in range(33):
            z = step(z, u)
            if clear(z[0], z[1]) < 0: ok = False; break
        if ok: return True
    return False
U = (-1, -.5, 0, .5, 1)
def search(s0, N):
    n = 0
    for a, b in itertools.combinations(range(1, N), 2):
        for us in itertools.product(U, repeat=3):
            seq = [us[0]] * a + [us[1]] * (b - a) + [us[2]] * (N - b)
            s, ok = s0, True
            for u in seq:
                s = step(s, u)
                if clear(s[0], s[1]) < 0: ok = False; break
            if ok and loiter_ok(s):
                return seq
    return None
starts = json.loads(sys.argv[1])
for s0 in starts:
    print('start', [round(c, 2) for c in s0], 'loiter-from-start ok', loiter_ok(tuple(s0)))
    for N in [int(n) for n in os.environ.get('NS', '12,16,20,24,30').split(',')]:
        r = search(tuple(s0), N)
        print(f'   N={N:2d}', 'FEASIBLE ' + str(r) if r else 'none')
