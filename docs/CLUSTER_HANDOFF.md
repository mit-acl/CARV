# TTT-CARV — cluster setup handoff

Written on the i9 dev box for a fresh Claude Code agent working on the MIT HPC
(Engaging). Goal: get this repo running in a conda env + Jupyter kernel, then run
parallel trials with ~64 workers.

Read this whole file before running anything.

---

## 0. The clone will be incomplete — read this first

`git clone` alone does **not** give you a working checkout. As of 2026-08-20 the
dev box has uncommitted and untracked work that the algorithm depends on:

- **`nfl_robustness_training/src/alg14_split_terminal.py` is UNTRACKED.**
  This is the algorithm actually being run. It is not in the remote.
  So are `anim_alg14.py`, `alg_with_backwards.py`, and ~20 other `alg*.py`.
- **9 tracked files are modified but uncommitted**, including
  `time_budget.py`, `run_trials_random_obs.py`, `run_trials.py`,
  `alg12_mpc_every_timestep.py`, `mpc_safety_filter_acados.py`,
  `utils/unicycle_mpc_acados.py`.

These modifications are not cosmetic — they contain the parallelism fixes
described in section 4. Running the clone without them reproduces known bugs.

**Ask the user to commit and push before you start**, or to rsync the working
tree over. Do not assume a clean clone is enough. Verify on the cluster:

    ls nfl_robustness_training/src/alg14_split_terminal.py
    grep -n "calibrated_costs" nfl_robustness_training/src/time_budget.py

If either fails, stop and tell the user the transfer is incomplete.

What *does* travel correctly with a clone:
- `third_party/` (auto_LiRPA, nfl_veripy, CROWN-IBP) — vendored as regular files,
  738 of them. Note `.gitmodules` still declares them as submodules; that file is
  stale, ignore it. No `--recursive` needed.
- `nfl_robustness_training/src/controller_models/` — 74 tracked `.pth` files,
  the trained NN controllers. These are the weights the sim loads.

---

## 1. Environment

Python **3.10.3** on the dev box. Match the minor version (3.10.x); torch 2.2.2 +
jaxlib 0.4.33 + TF wheels are all built against it.

    conda create -n carv python=3.10 -y && conda activate carv
    pip install -r requirements-cluster.txt

That file is a cleaned freeze of the working env (154 packages). Versions are
pinned deliberately — see the header comment in the file.

Then the four packages that are NOT on PyPI:

    pip install -e third_party/auto_LiRPA
    pip install -e third_party/nfl_veripy
    pip install git+https://gitlab.com/neu-autonomy/certifiable-learning/jax_verify.git@c9dedb6a60a84a9262d07d1a7efa67106649ece8

`jax_verify` is a **GitLab** package, not PyPI, and it is genuinely required —
`nfl_veripy/src/nfl_veripy/constraints/Constraints.py:8` imports it at module
level. If the cluster blocks outbound git to gitlab.com, it must be vendored.

The fourth (`acados_template`) comes from section 2.

### Known hazard: two TensorFlow installs

The dev env contains BOTH `tensorflow==2.13.0` and `tensorflow_cpu==2.20.0`.
Both provide the `tensorflow` module, so one silently shadows the other there.
On a clean env this may resolve differently and cause a behavior difference that
looks like a cluster problem but is not. TF is not droppable —
`nfl_veripy/src/nfl_veripy/utils/nn.py:8` imports keras at module level.
If imports misbehave, this is the first thing to check.

### GPU

Not needed. The sim path is CPU-only (`device='cpu'` throughout
`REAL_integrated_sim.py`). Install CPU wheels. This is why the usual HPC
CUDA/driver matching pain does not apply here.

---

## 2. acados — the only hard dependency

Not pip-installable. The current install points at another user's home
directory (`/home/sazhang/acados`) and does not transfer. Build from source.
The user has approved a ~20 minute build.

Pin to the exact commit the working setup uses — current acados master has API
changes that break `utils/unicycle_mpc_acados.py`:

    git clone https://github.com/acados/acados.git
    cd acados
    git checkout 20bed35244ece7d5bd2e72da554b14fdaf3091e9   # tag v0.2.4-2, Sept 2023
    git submodule update --init --recursive

    mkdir -p build && cd build
    cmake -DACADOS_WITH_QPOASES=ON \
          -DACADOS_WITH_OPENMP=OFF \
          -DCMAKE_BUILD_TYPE=Release \
          -DBLASFEO_TARGET=X64_AUTOMATIC \
          -DHPIPM_TARGET=X64_AUTOMATIC ..
    make install -j16

    pip install -e <acados_root>/interfaces/acados_template

Then export **both** (the code reads both independently —
`utils/unicycle_mpc_acados.py:36-37`):

    export ACADOS_SOURCE_DIR=<acados_root>
    export ACADOS_LIB_DIR=<acados_root>/lib

No `LD_LIBRARY_PATH` needed: the module explicitly `ctypes.CDLL(..., RTLD_GLOBAL)`
preloads libblasfeo/libhpipm/libqpOASES_e/libacados.

Three things that matter and are easy to get wrong:

1. **Keep `ACADOS_WITH_OPENMP=OFF`.** Turning it on makes each of 64 workers
   spawn its own OpenMP pool, recreating exactly the oversubscription bug fixed
   in section 4. This is not the upstream default recommendation — it is a
   deliberate choice for this workload.
2. **`BLASFEO_TARGET=X64_AUTOMATIC` detects at build time.** Build it on the same
   node architecture you will run on. Do not copy `.so` files from the dev box
   (Comet Lake) to an EPYC node, and be careful if the cluster is heterogeneous —
   node3113 is a 192-CPU EPYC 9474F, node3107 has 128 CPUs. Mismatched builds
   fail with illegal-instruction crashes.
3. **`t_renderer`** is a prebuilt binary acados downloads into `bin/` during the
   build, used for template codegen. Compute nodes often have no outbound
   internet. Build on a login node, or copy the known-good one from
   `/home/sazhang/acados/bin/t_renderer` (static x86-64, 6.5 MB, portable).

Verify:

    python -c "from acados_template import AcadosOcpSolver; print('acados ok')"

---

## 3. Running things

**Scripts must be launched from the repo root**, not from `src/`.
`nfl_robustness_training/src/utils/nn.py:11` does `PATH = os.getcwd()` and builds
all controller-model paths from it. Running from anywhere else fails to load
weights.

Entry points:
- `nfl_robustness_training/src/run_trials_random_obs.py` — drives **alg14**, the
  current algorithm. This is the one to use.
- `run_trials.py` (alg12) and `run_trials_di.py` (alg13) are older.

Worker count is env-overridable, no source edit needed:

    TTTCARV_WORKERS=64 python nfl_robustness_training/src/run_trials_random_obs.py

It is clamped to `len(os.sched_getaffinity(0))`, so it cannot overcommit past
what SLURM actually granted. The startup line prints both numbers.

**Do not launch the trial pool from a notebook.** It uses
`mp.get_context('spawn')`, which re-imports `__main__` in every worker; in a
Jupyter kernel `__main__` is not a file, so workers break. Use a terminal inside
the allocation (JupyterLab: File -> New -> Terminal). The notebook kernel is fine
for analysis and plotting.

### Sizing workers

Check what you actually own before choosing a number:

    python -c "import os; print(len(os.sched_getaffinity(0)))"

`nproc` reports the whole node and is misleading — the login shell on node3107
showed 128 CPUs while `load average` was already 96 from another tenant. Inside a
proper `-c 64` allocation the cgroup fences you off and affinity should report 64.
If affinity reports the full node count, you are NOT inside the allocation.

Stay off hyperthreads: physical cores = logical / `Thread(s) per core` (2 here).

---

## 4. Why the parallelism fixes exist — do not undo these

The algorithm is **wall-clock gated**. `TimeBudget` (`src/time_budget.py`)
measures `time.perf_counter()` against a fixed 0.20 s/timestep deadline, and
every phase is gated on `budget.remaining`. So *machine load, not the seed,
decides how much verification runs.* Two fixes address this; both are in the
uncommitted work from section 0.

**(a) Thread pinning** in every `_worker_init`. Without it each worker spawns
`torch.get_num_threads()` (10) intra-op threads and N workers oversubscribe the
box ~N-fold. Measured at 16 workers on the dev box: concrete cost per step went
0.0142 s -> 0.172 s, a 12x error. `max_affordable_concrete()` then computed
`0.200 // 0.0142` = 14 steps, which actually cost 2.4 s and blew the entire
timestep budget in one call, so later phases were skipped.

This caused two safety-relevant bugs, both closed by pinning:
- `psf_no_diverge` fired 0-5x per trial, applying **unverified nominal control**
  (the branch commented `# SHOULD NOT RUN IF WORKING!!!!!`).
- Trajectories recorded only 17-22 of 61 states, so `point_collision` scanned a
  truncated path and reported **false "safe"** results.

The pinning block sets `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` and
`torch.set_num_threads(1)`, and **must run before the worker imports torch**.

**(b) Per-process calibration** — `time_budget.calibrated_costs()`. Measures op
costs inside each worker under real load instead of shipping a static table
calibrated on an idle box. Cached per process. Disable with
`TTTCARV_CALIBRATE=0` to fall back to the static table.

### What is still broken (unfixed, known)

Calibration tightens the decision spread but buys **no reproducibility**. Same
seed still gives different trajectories under parallelism, because `elapsed`
still reads the real clock. Measured on the dev box at 16 workers with pinning +
calibration, same seed repeated 16x: **alg14 gave 3/16 distinct trajectories**
(mpc_calls 3-3, no_diverge 0, all 61 states, ~17 s/trial). alg12 under the same
conditions gave 16/16 distinct — alg14 is substantially more stable, but neither
is reproducible.

The real fix — not implemented, repeatedly raised with the user — is a **virtual
budget**: charge modeled costs instead of reading `perf_counter`. That is the
only change that makes runs seed-reproducible and machine-independent.

**Consequence for the migration:** with the wall-clock budget live, results from
the EPYC cluster and the i9 dev box are **not directly comparable**, and neither
are results from different cluster node types. Run any given experiment entirely
on one machine, or implement the virtual budget first. Raise this with the user
before they generate paper numbers on mixed hardware.

---

## 5. Other project facts worth knowing

- **acados codegen dirs are pid-keyed**: `mpc_safety_filter_acados.py:93` uses
  `solver_name=f'unicycle_acados_sf_{os.getpid()}'`, and export paths are
  `/tmp/{solver_name}_gen`. Parallel workers therefore do not race. But a *second
  filter in the same process* clobbers the first one's generated C code — which
  is why calibration reuses the caller's existing `mpc_sf` rather than building
  its own. On the cluster, check `/tmp` is node-local and writable, and that
  stale dirs from previous jobs get cleaned.
- **Backprojection index ordering** (already fixed, 2026-02-10, do not
  reintroduce): `nfl_veripy`'s `get_backprojection_set` returns sets ordered
  index 0 = 1 step back (closest to target), index N-1 = furthest. Correct
  mapping is `current_timestep = target_timestep - (i + 1)`. Both `backward()`
  and `backward_from_set()` in `REAL_integrated_sim.py` originally had this
  reversed.
- **RNG is not the problem.** `REAL_integrated_sim.py:191` seeds a dedicated
  `np.random.RandomState(seed)`, used at only 4 sites. Nondeterminism under
  parallelism was verified to be the wall-clock budget, not RNG.
- The dev box has a broken `pyenv` (two installs on PATH, shim recurses and
  hangs forever). That is **local only** — irrelevant on the cluster with conda.
  Mentioned so you do not copy any `pyenv`-specific workaround you find in old
  notes or scripts.

---

## 6. Suggested order

1. Confirm the transfer is complete (section 0). Stop if not.
2. Get an interactive allocation with 64 CPUs; verify affinity reports 64.
3. `module avail` -> load python/gcc/cmake; create the conda env; install deps.
4. Build acados on the allocated node; verify the import.
5. Smoke test: one trial, `TTTCARV_WORKERS=1`, from the repo root. Confirm it
   finishes and reports 61 states with `no_diverge=0`.
6. Only then scale to 64 workers, and compare the per-trial wall time and the
   `[calibrate]` line against the dev-box baseline
   (`concrete=0.0134s mpc=0.0554s`, ~10 s/trial solo).
