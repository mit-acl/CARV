"""Capture WHY alg15's MPC backup build is rejected on the trials that go unsafe.

alg15 already prints the diagnostics we need -- `[MPC build] tau=N STATUS`, the
`[PSF] No backup at t=` state dump, `[PSF buf] MPC solve failed` -- but the batch
driver's _worker_init devnulls worker stdout, so they are thrown away. Here each
trial runs with sys.stdout swapped for a filter that keeps only those lines, so
the failures can be attributed without touching alg15 itself.

The failure is load-dependent (0/200 at 8 workers, 3/200 at 64), so this has to
run under sustained contention to catch any.
"""
import numpy as np, multiprocessing as mp, os, sys, re, json
from collections import Counter

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'nfl_robustness_training', 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

CASES_FILE = os.environ.get('TTTCARV_CASES', '/home/ryosei_t/alg15_repro_cases.txt')
N_WORKERS  = min(int(os.environ.get('TTTCARV_WORKERS', 64)),
                 len(os.sched_getaffinity(0)))
ALG_MODULE = os.environ.get('TTTCARV_ALG', 'alg15_split_terminal')
REPEATS    = int(os.environ.get('TTTCARV_REPEATS', 10))
OUT        = os.environ.get('TTTCARV_OUT', '/home/ryosei_t/alg15_reasons.jsonl')

KEEP = re.compile(r'\[MPC build\]|\[PSF\] No backup at t=|\[PSF buf\] MPC solve failed'
                  r'|\[PSF PASSTHROUGH\]')
ANSI = re.compile(r'\x1b\[[0-9;]*m')


class ReasonTap:
    """Line-buffered stdout sink that retains only the diagnostic lines.

    A whole trial's stdout is far too verbose to keep (60 timesteps of RSOA
    dumps); filtering at write() keeps this bounded to a few hundred lines.
    """

    def __init__(self, cap=2000):
        self.buf, self.hits, self.cap = '', [], cap

    def write(self, s):
        self.buf += s
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            line = ANSI.sub('', line).strip()
            if KEEP.search(line) and len(self.hits) < self.cap:
                self.hits.append(line)

    def flush(self):
        pass


def load_cases():
    out = []
    for line in open(CASES_FILE):
        line = line.strip()
        if not line:
            continue
        seed, obs_s = line.split('|')
        out.append((int(seed), [np.array([float(v) for v in g.split(',')])
                                for g in obs_s.split(';')]))
    return out


def point_collision(traj, obstacles):
    for s in traj:
        f = np.asarray(s).flatten()
        for cx, cy, r in obstacles:
            if (f[0] - cx) ** 2 + (f[1] - cy) ** 2 <= r ** 2:
                return True
    return False


def _worker_init():
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'
    import torch
    torch.set_num_threads(1)
    # stderr still goes to the void: that is the acados SQP_RTI firehose, written
    # from C on fd2, which no sys.stderr swap would catch anyway.
    sys.stderr = open(os.devnull, 'w')


def _run(args):
    i, seed, obstacles = args
    from REAL_integrated_sim import setup_analyzer
    from importlib import import_module
    test = import_module(ALG_MODULE).test
    analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')
    tap, real = ReasonTap(), sys.stdout
    sys.stdout = tap
    try:
        (traj, _, _, mpc_calls, no_diverge, queue_empty,
         pt_found, pt_used) = test(seed=seed, analyzer=analyzer,
                                   obstacles=obstacles)
    finally:
        sys.stdout = real
    return dict(i=i, seed=seed, mpc=mpc_calls, nd=no_diverge, pf=pt_found,
                pu=pt_used, unsafe=point_collision(traj, obstacles),
                lines=tap.hits)


def classify(line):
    """Collapse a diagnostic line to a category, dropping the numbers."""
    if '[MPC build]' in line:
        m = re.search(r'\]\s+\S+\s+(VALID|INFEASIBLE\()', line)
        if not m:
            return 'MPC build ?'
        if m.group(1) == 'VALID':
            return 'MPC build VALID'
        kind = 'term_D' if 'term_D' in line else (
               'path_raw' if 'path_raw' in line else '?')
        return f'MPC build INFEASIBLE({kind})'
    if '[PSF buf] MPC solve failed' in line:
        return 'MPC solve raised'
    if '[PSF] No backup at t=' in line:
        buf = re.search(r'buf\[\d+\]=(\w+)', line)
        td  = re.search(r'td=(\w+)', line)
        inD = re.search(r'bnext_in_D=(\w+)', line)
        raw = re.search(r'bnext_in_raw=(\w+)', line)
        return (f"PSF no-backup: buf={buf.group(1) if buf else '?'} "
                f"td={td.group(1) if td else '?'} "
                f"in_D={inD.group(1) if inD else '?'} "
                f"in_raw={raw.group(1) if raw else '?'}")
    if '[PSF PASSTHROUGH]' in line:
        return 'PSF passthrough'
    return 'other'


if __name__ == "__main__":
    cases = load_cases()
    args = [(i, s, o) for _ in range(REPEATS) for i, (s, o) in enumerate(cases)]
    print(f"{len(cases)} cases x{REPEATS} = {len(args)} runs at {N_WORKERS} workers",
          flush=True)
    ctx = mp.get_context('spawn')
    res = []
    with ctx.Pool(processes=min(N_WORKERS, len(args)),
                  initializer=_worker_init) as pool:
        for r in pool.imap_unordered(_run, args):
            res.append(r)
            if r['unsafe']:
                print(f"  UNSAFE seed={r['seed']} mpc={r['mpc']} nd={r['nd']} "
                      f"({len(r['lines'])} diag lines)", flush=True)

    with open(OUT, 'w') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')

    bad = [r for r in res if r['unsafe']]
    print(f"\n{'='*70}")
    print(f"{len(bad)}/{len(res)} unsafe")
    for r in bad:
        print(f"\n--- seed={r['seed']}  mpc={r['mpc']} nd={r['nd']} "
              f"pt_found={r['pf']} pt_used={r['pu']} ---")
        for cat, n in Counter(classify(l) for l in r['lines']).most_common():
            print(f"   {n:>4}x  {cat}")
        print("   first 3 raw diagnostic lines:")
        for l in r['lines'][:3]:
            print(f"     | {l[:150]}")
    ok = [r for r in res if not r['unsafe']]
    print(f"\n--- category totals across {len(ok)} SAFE runs (for contrast) ---")
    for cat, n in Counter(classify(l) for r in ok for l in r['lines']).most_common(8):
        print(f"   {n:>5}x  {cat}")
    print(f"\nfull per-run detail: {OUT}")
