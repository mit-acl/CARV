# alg20 — REACH-PSF on the unicycle, 10,000 random obstacle scenes

**10000 / 10000 safe. Zero collisions. Zero fallbacks of any kind.**

Run under low contention — 8 workers on an exclusive node, 4–8 physical cores
per worker — because deployment is one robot with a machine to itself, not a
throughput-packed cluster. The 0.20 s per-timestep budget only means what it
claims if each trial actually gets the compute it was calibrated against.

| | |
|---|---|
| scenes | 10,000 (4 chunks × 2,500), all seeds distinct |
| collisions | **0** |
| PSF no-diverge fallbacks | **0** timesteps across 0 trials |
| PSF queue-empty fallbacks | **0** timesteps across 0 trials |
| episode length | 61 timesteps, every trial (no early aborts) |
| PSF activations | mean 2.52 / trial, median 2, max 9; 99.6% of trials activated at least once |
| compute | 22.5 CPU-hours, median 8.08 s per trial |

## Scenario

| | value |
|---|---|
| system | unicycle, `dt = 0.2 s`, NN controller `Unicycle_NL / natural_none_default` |
| start set | `x ∈ [-9.55, -9.45]`, `y ∈ [3.45, 3.55]`, `θ ∈ [-π/24, π/24]` |
| concrete start | drawn uniform from that box per seed |
| goal | origin |
| obstacles | 1–5 per scene, uniform count |
| obstacle radius | 0.5 |
| obstacle centres | uniform in `x ∈ (-7.0, -1.0)`, `y ∈ (-1.0, 2.0)` — **unrestricted**, no keep-out |
| horizon | 60 timesteps max |
| MPC | acados, 12-step horizon, split-terminal (PATH nodes clear raw radius `r`, TERMINAL clears inflated danger region `D` for a max-turn loiter) |
| compute budget | 0.20 s per timestep, costs calibrated per trial |

Realised obstacle-count distribution across the 10,000 scenes: 1974 / 1981 /
2044 / 2025 / 1976 for n = 1…5 — flat, as intended.

A collision is the true trajectory entering any obstacle disc. All 10,000
trials completed the full 61 timesteps and none did.

## Per-chunk

| chunk | node | seed | elapsed | safe | contention spread | probe median |
|---|---|---|---|---|---|---|
| 1 | node4007 | 20260824 | 42:54 | 2500 / 2500 | 1.1× | 0.34 ms |
| 2 | node2114 | 20260825 | 59:10 | 2500 / 2500 | 1.3× | 0.70 ms |
| 3 | node2423 | 20260826 | 1:00:16 | 2500 / 2500 | 1.4× | 0.70 ms |
| 4 | node3609 | 20260827 | 47:51 | 2500 / 2500 | 1.1× | 0.59 ms |

`contention spread` is max/min of a fixed 256×256 matmul timed inside each
trial under whatever load was really present. 1.0× means every trial saw
identical compute. 1.1–1.4× confirms the low-load regime was achieved rather
than merely requested — this is the number that makes the run's headline claim
meaningful, and it is reported rather than assumed.

Timestep cost stayed inside budget: median 0.132 s against the 0.20 s budget,
p90 0.161 s, worst trial mean 0.232 s. 10 trials of 10,000 (0.10%) had a mean
timestep above budget; all 10 were safe.

## The nine historically-failing seeds

These nine collided under earlier algorithm versions on byte-identical scenes.
All nine are safe here:

| seed | obstacles | PSF activations | passthrough steps used |
|---|---|---|---|
| 242244880 | 2 | 3 | 1 |
| 856113296 | 2 | 3 | 4 |
| 982114962 | 4 | 3 | 4 |
| 1091579819 | 1 | 4 | 1 |
| 1471202357 | 2 | 3 | 2 |
| 1844520186 | 1 | 2 | 0 |
| 1956405985 | 3 | 3 | 2 |
| 2095399857 | 1 | 4 | 1 |
| 2132122960 | 4 | 2 | 1 |

Historical collision counts on this same scene set: alg14 = 3, alg19 = 8,
alg19 rerun = 4. The mechanism was confirmed directly on seed 2095399857 in a
single-process A/B: alg19 activates at t=40 and collides at t=46–49, while
alg20 raises `VacuousHeadingBound` (`hw_theta = 4.9158 ≥ π`) at τ=39, activates
two steps earlier at t=38, and clears. Caveat: two of the nine
(2132122960, 1471202357) produce byte-identical traces under both algorithms
and were already safe under alg19, so the guard is not the explanation for
every one of the nine — part of that record is wall-clock jitter. Attribution
across all nine is not measured.

## What alg20 adds

`verify_heading_bounds`: when the heading interval half-width reaches π the
interval spans ≥ 2π and certifies nothing. alg20 raises `VacuousHeadingBound`
rather than building a backup plan on it; the exception is caught in
`build_mpc_backup`, the buffer entry becomes `None`, and the step falls through
to passthrough. Fail-safe by construction — a vacuous certificate can no longer
be mistaken for a valid one. The flag reached the solver on all four chunks
(`heading_flag_wired=4` in every log header); this is checked explicitly
because `make_mpc_safety_filter_acados` swallows unknown kwargs silently and
has shipped inert flags before.

## Provenance

- code: commit `ae8b11d`, `heading_flag_wired=4`, logs stamp `dirty=14 files`
- driver: `nfl_robustness_training/src/run_trials_random_obs.py`
- SLURM job 21152033, array 1–4, `mit_preemptable`, `--exclusive`
- `TTTCARV_ALG=alg20_vacuity`, `TTTCARV_TRIALS=2500`, `TTTCARV_WORKERS=8`,
  `TTTCARV_SEED=20260823+chunk`, `TTTCARV_GOAL_KEEPOUT=0`
- raw logs `chunk{1..4}.log`, per-trial fingerprints `fp_{1..4}.json`
  (seed, collision, timesteps, wall, probe before/after, pid)

**Scene reproducibility.** The working tree carried an unused goal-keep-out
option, disabled for this run (`TTTCARV_GOAL_KEEPOUT=0`). At 0 its rejection
test never rejects, so the generator draws `n`, then `cx`, `cy` per obstacle in
the same order and consumes the identical RNG stream as the committed
`_gen_obstacles` at `ae8b11d`. **The 10,000 scenes are therefore reproducible
from `ae8b11d` directly**, by seeding `TTTCARV_SEED = 20260823 + chunk`. The
remaining uncommitted changes to the driver are the fingerprint/probe
instrumentation, which records timing and does not affect scene generation or
trial outcomes.

**Not bit-reproducible.** Admission to the 0.20 s budget depends on wall-clock,
so a rerun on identical scenes will not reproduce identical intervention
counts. The safety outcome is the reproducible quantity; per-trial `mpc` counts
are not.

## Relationship to the earlier packed runs

| run | obstacle region | load | result | fallbacks |
|---|---|---|---|---|
| wide | unrestricted | 1 worker / physical core | 10000 / 10000 | 1 timestep / 1 trial |
| **wide (this run)** | **unrestricted** | **8 workers, exclusive node** | **10000 / 10000** | **0** |

The single fallback in the packed run was traced to budget overrun, not a logic
error: `can_afford` admits work whose calibrated median cost fits the remaining
budget, and once admitted nothing preempts it, so an under-predicted MPC build
overruns. Measured at 0.216 s and 0.209 s against the 0.200 s budget on an idle
node, and 0.555 s under contention. It does not appear in this run.

One sample of zero cannot distinguish "low load removed the fallback" from "a
rare event did not recur." The overrun measurement above is the mechanism
evidence and stands independently of this run.
