# Reach-PSF: where this project ended up

Last updated 2026-11-02. This is the short version of what I built, what I
compared it against, and why I stopped. The long lab notebook is
`results_alg29/alg29_results.md`. Some of its tables are from older revisions,
so trust the numbers here first.

## The idea

A neural-network controller drives a unicycle robot (constant speed 1 m/s,
turn rate limited to ±1 rad/s, 0.2 s time steps) through a field of circular
obstacles. The robot only gets noisy measurements of its position and heading.

A safety filter sits between the NN and the robot. Most of the time it lets the
NN command through. When it can't prove the NN is safe, it takes over with a
backup plan.

My filter (Reach-PSF, `src/alg28_sound.py`) checks the NN with reachability
analysis: it computes a box that is guaranteed to contain every state the robot
could reach under the NN, given the measurement noise. If those boxes stay clear
of obstacles and end somewhere the robot can safely circle forever, the NN is
allowed to drive. Otherwise it switches to an MPC backup plan.

The pitch was that a reachability filter can run its checks one step ahead, so
it never has to make the robot wait on a solver, and that it would step in less
often than a standard MPC-based filter.

## What I compared against

`src/alg29_generic_psf.py` has the baselines. The important ones:

- **R2, delay-compensated tube MPC filter.** This is the fair competitor. It
  knows its solve takes time, so it takes the current set of possible states,
  pushes it forward through the command already being applied, solves from that
  predicted set, and applies the result at the next step.
- **R1**: the same tube MPC, but it pretends solving takes zero time. Not
  something you could actually deploy. It's here as a reference.
- **G1**: a textbook MPC filter that ignores its own solve time. This is the
  "what goes wrong if you don't compensate" baseline.

All filters see the same scenes, the same noise draws and the same NN.

## Results

"Intervention" means the percentage of time steps where the filter changed the
NN's command. Lower is better. Brackets are 95% paired confidence intervals.

**Main comparison** (500 random scenes, uniform noise ±0.15):

| Filter | Collisions | Intervention | Compute per step |
|---|---|---|---|
| Reach-PSF (mine, real-time version) | 0 | 4.81% | 94 ms |
| R2, delay-compensated tube MPC | 0 | 4.97% | 75 ms |

Difference: R2 is +0.16 pp [−0.10, +0.43], which isn't significant. Even if I
let my filter skip the delay entirely (an unrealistic best case, 4.71%), the
gap only grows to +0.26 pp [−0.01, +0.54]. That's still not significant.

For comparison, G1, which ignores its solve time, collided in up to 31% of
scenes (at a 0.15 s delay). So delay compensation matters a lot, but R2 already
does it, and once it does, my filter has no edge.

**Corridor test.** I built a corridor longer than the MPC's 2.4 m planning
horizon, to see whether the reachability check would let the NN fly through
where MPC can't plan far enough. 100 trials per setting, 0 collisions for
everyone:

| Noise | Corridor gap | Mine | R2 |
|---|---|---|---|
| ±0.15 | 1.0 m | 25.4% | 23.3% |
| ±0.15 | 1.4 m | 25.6% | 26.7% |
| ±0.05 | 1.0 m | 23.9% | 23.2% |
| ±0.05 | 1.4 m | 28.1% | 25.8% |
| ±0.017 | 1.0 m | 22.7% | 22.8% |
| ±0.017 | 1.4 m | 27.3% | 25.8% |

Neither filter ever went into the corridor. Both went around it. The reason mine
didn't: its reachability boxes grow fast (one went from 7 cm to 43 cm wide in 3
or 4 steps, even at ±0.05 noise). By the corridor entrance, the boxes are
wider than the gap.

The first corridor scene I tried (`corridor_d1.0.json`) looked like a big win:
R2 collided in 88/100 trials. That turned out to be unfair. R2 had no valid
12-step plan from the starting position, so it fell back to circling and hit
the first wall. I moved the corridor further away (`corridor_k15_*`) so both
filters started from a feasible position.

## Checking the delay compensation

I wanted to be sure R2 wasn't secretly broken in a way that made it look good.

- **By hand** (`cluster_scripts/checks/handcheck_r2_prediction.py`, runs on
  the login node with just numpy). One plant step matches the hand-computed
  Euler step. A ±0.15 box stepped forward matches the box I computed by hand.
  Out of 1,000,000 random states, none landed outside the predicted box.
- **In the sim** (500 scenes, 30,000 steps). The true state was inside R2's
  planned tube at every step. R2's remaining plan, re-simulated from the true
  state, was still valid at every step.
- This is the standard way to handle solver delay in MPC (predict where you'll
  be when the command lands, then solve from there). R2 does the set version of
  it.

These checks did catch a real bug earlier. R2 was predicting its state forward
but not its uncertainty box, so in effect it planned from old information. That
version collided in 269/500 scenes. It's fixed, and all the numbers above are
after the fix.

## Known bug in my filter (not fixed)

In mode J, the code that sizes the reachability boxes assumes the estimate is
never further from the truth than the sensor noise. That's only true once the
robot has taken a measurement. At the very first step it hasn't, and its
starting heading uncertainty is 0.131 rad. If the noise is set lower than that
(`TTTCARV_VMAX_SIGMA` below about 2.6), the code underestimates the uncertainty
for the first few steps. The boxes it builds can then miss the true state, by
up to 0.055 rad in heading.

- At the default noise (±0.15) the bug can't trigger. All the main results are
  unaffected.
- In the low-noise corridor runs it did trigger (24/100 trials at ±0.05, 60/100
  at ±0.017). It never caused a collision.
- The fix: in `joint_step`, only apply the `min(vmax_e, hw)` cap after the first
  measurement has been used.

## Where things are

| What | Where |
|---|---|
| My filter | `src/alg28_sound.py` |
| Baselines (R1, R2, G0–G3) | `src/alg29_generic_psf.py` |
| Double-integrator version | `src/alg22_di.py`, `src/run_trials_di.py` |
| Animations | `src/anim_alg28.py`, `results_alg28/anim/` |
| Full lab notebook and figures | `results_alg29/` |
| Error-box proof sketch | `results_alg28/error_box_lemma.md` |
| Cluster scripts | `cluster_scripts/scripts/` (copies of `~/scripts`) |
| Per-experiment scripts and predictions | `cluster_scripts/runs/alg41_score` … `alg45_corridor` |
| Older superseded code | `archive/`, see its README |
| Raw run data (fingerprints, logs) | `~/runs/` on Engaging, not in git |

Cluster setup is in `../CLUSTER_HANDOFF.md`. In short: use the `carv` conda env
and submit through sbatch. Trials can't run on the login node.

## Noise knob

`TTTCARV_VMAX_SIGMA` sets the noise bound on both sides (bound = value × 0.05,
default 3 → ±0.15).

## If someone picks this up again

The open question is whether there's any scene where the reachability check
lets the NN keep driving and the tube MPC can't. What I'd try:

- Tighter reachability boxes, re-computed each step instead of reused from
  early steps. This costs compute, and my filter is already slower than R2.
- Compare against R2 with a longer horizon, to see whether its horizon is really
  the limit.
- Fix the mode J bug first.
