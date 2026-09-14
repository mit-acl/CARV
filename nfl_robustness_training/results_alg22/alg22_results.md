# Reach_PSF_FINAL (alg22) — receding-horizon REACH-PSF on the unicycle, 10,000 random obstacle scenes

## Final evaluation: Reach_PSF_FINAL (job 22728795, 2026-09-14)

**10000 / 10000 safe, and every timestep inside the 0.20 s budget: 0 of
600,000 steps over, slowest 198.3 ms.** Zero collisions, crashes and fallback
events.

This is the headline run. `alg22.py` is now `Reach_PSF_FINAL.py` (commit
`83ae8e2`, run with `TTTCARV_ALG=Reach_PSF_FINAL`). It is the final-1
configuration below plus a budget that prices work on measured cost, fewer
solver iterations per one-shot solve, and a deadline between concrete steps.
Same 10,000 scenes as final 1, final 2, v2 and `results_alg20` (trial seeds
verified identical to final 2).

**Budget requirement.** Every timestep must fit in 0.20 s. A step is scored as
its elapsed time minus symbolic reachability time, because symbolic is allowed
to span steps by design. Scores come from the per-step budget ledger (one
record per trial per timestep).

| | Reach_PSF_FINAL (final 3) | final 2 (job 22710554) |
|---|---|---|
| scenes | 10,000 (4 chunks × 2,500) | same |
| collisions | **0** | 0 |
| crashed trials (`error` rows) | **0** | 0 |
| PSF no-diverge / queue-empty fallbacks | **0 / 0** | 0 / 0 |
| t=0 backup build failures (`p1_t0_infeasible`) | **0** | 0 |
| terminal-loiter steps, unrecoverable, stale-refused | **0** | 0 |
| no-backup / loiter / unrecoverable / stale-refused log events | **0** | 0 |
| calibration clamps, certificate-spent steps | **0** | 0 |
| episode length | 61 timesteps, every trial | same |
| **steps over 0.20 s (elapsed − symbolic)** | **0 of 600,000** | 8 of 600,000 |
| **slowest step (elapsed − symbolic)** | **198.3 ms** | 229.8 ms |
| PSF activations | mean 2.46 / trial, median 2, max 8 | same |
| re-sim bootstrap builds | 120,944, of which 17,126 infeasible | 121,904 / 17,135 |
| re-solves | 9,412 succeeded, 1,163 failed; 1,106 uncertified fallback steps | 9,436 / 1,166; 1,114 |
| failed re-solve reasons | `rollout_rejected` only | same |
| wall / timestep (includes symbolic) | median 0.122 s, p95 0.136 s, max 0.170 s | 0.123 / 0.137 / 0.174 |
| compute | 20.0 CPU-hours, median 7.45 s per trial | 20.1 / 7.47 s |

### What changed from final 1

**1. Measured-cost budget (`TTTCARV_MPC_PRICING=1`, code in `time_budget.py`).**
Final 1 priced every MPC operation at the median of one calibration probe per
worker, and concrete propagation at a calibrated median per step. Real costs
ran higher: on EPYC 7542 a single one-shot solve at the final-1 iteration count
took median 71.5 / p90 92 / max 104 ms (per-op profile, job 22706513), a
two-solve multi-start up to 188 ms, and the re-solve was admitted at half the
probe price. Admitted work regularly ran past 0.20 s. Now:

- An MPC solve is priced at the 99.5th percentile of this worker's last 1,000
  complete solve times, plus a 3 ms margin (`TTTCARV_BUDGET_MARGIN`). Final
  price in this run: median 52.6 ms (p5 48.3, p95 57.4).
- A concrete step is priced at the 95th percentile of the wall time of each
  concrete call divided by the steps it completed. Final price: median 8.83 ms.
- Before 50 samples, a price is the larger of 1.25× the calibration and the
  slowest time seen.
- Every solve must fit before it starts, including the receding re-solve and
  the second multi-start solve (`mpc_safety_filter_acados.py`, `MPCDeadline`).

**2. Hard deadlines inside operations.**

- The acados RTI loop stops before an iteration that cannot finish
  (`utils/unicycle_mpc_acados.py`). A one-shot plan cut short is still
  re-simulated through the plant and must pass the acceptance test. In this run
  no solve was cut short (`mpc_truncated_solves = 0`).
- The re-simulation rollout stops at the deadline.
- P2 stops between concrete steps when less than max(price, previous step's
  wall time) is left, and advances only by the steps completed
  (`REAL_integrated_sim.py`, `concrete(..., time_left, step_price)`). This
  fired 14 times in 10,000 trials.

**3. One-shot RTI iterations (`TTTCARV_ONESHOT_RTI=10`).** Each of the two
one-shot refine passes was hard-wired to 20 SQP_RTI iterations: 40 in total,
about 69 ms of a 71.5 ms solve on EPYC 7542. Ten per pass halves the solve
cost. Convergence is not taken on trust: the plan flown is the plant
re-simulation of the solver's controls, which must clear every obstacle and
end outside D, so an under-converged plan is rejected, not flown. Rejections
did not rise (below). Without this change, honest pricing starved the
re-solve.

**4. Garbage collection per trial (`TTTCARV_GC_PER_TRIAL=1`).** The budget
ledger's wrappers formed a reference cycle that kept the previous trial's
acados solver alive, so the next trial could load a solver generated for the
previous trial's obstacles. `gc.collect()` at the start of `test()` fixes it.

**5. Hardware.** All four chunks on EPYC 9474F (Zen 4), about 1.7× faster on
MPC than final 1's EPYC 7542, chosen by a node benchmark (one-shot median
38–44 ms vs 64–65 ms). As established under final 1, re-solve and budget
counts are only comparable on the same CPU model; collisions are comparable
across all runs.

### Same-CPU comparison (500 chunk-3 scenes, EPYC 7542)

The effect of changes 1 and 3 without the hardware change. Each arm on its own
EPYC 7542 node, 8 workers, per-step ledger on, 500/500 safe and 0 errors in
every arm (`~/runs/alg22_pricing/arm_*`).

| arm | pricing | RTI per pass | resolved / failed / uncertified | steps over 0.20 s |
|---|---|---|---|---|
| final-1 config | off | 20 | 378 / 83 / 72 | 1,606 |
| pricing only | on (MPC) | 20 | 255 / 176 / 160 | 496 |
| RTI only | off | 10 | 412 / 55 / 54 | 1,666 |
| pricing + RTI | on (MPC) | 10 | 387 / 75 / 69 | 473 |
| pricing incl. concrete | on | 20 | 245 / 182 / 160 | 2 |
| **pricing incl. concrete + RTI** | **on** | **10** | **385 / 76 / 67** | **0** |

Pricing alone fixes the budget but halves successful re-solves, because solves
priced at their real cost fit less often. Halving the iterations pays for it.

### Final 2 and the concrete deadline

Final 2 (job 22710554) was this configuration without change 2's concrete
deadline, on the same scenes and CPU. It left 8 of 600,000 steps over budget,
2 per chunk, max 229.8 ms. All 8 were at t=0, with concrete propagation
144–170 ms plus an MPC build of 36–68 ms. A 200-trial diagnostic (job
22713530) reproduced one: P2 requested 15 steps at a price of 8.42 ms/step, and
that call ran at 11.77 ms/step against a usual 7.3 ms, while the same trial
was back to normal by t=1 and the same seed was fine in final 2. The slowness
is transient per call, so no admission price can cover it; stopping between
steps can. The concrete price was also fed biased low (propagation time only,
divided by the requested rather than completed steps on collision stops),
which change 1's wall-time-per-completed-step fixes.

### Budget skips are expected

| counter | total | meaning |
|---|---|---|
| `mpc_skipped_no_budget` | 13,370 | a build the loop would have started did not fit this step |
| `mpc_deadline_hits` | 447 | a solve refused at its final check before starting |
| `mpc_second_start_skipped` | 2,774 | re-solve tried only the NN start; the swerve start did not fit |
| `concrete_deadline_hits` | 14 | P2 stopped between concrete steps |
| `mpc_truncated_solves` | 0 | solves cut short mid-iteration |

A skipped solve never discards an existing backup; the robot keeps flying the
certified plan it has. Skipped lookahead builds are retried on later steps. The
second-start skip is a real trade: with one candidate a re-solve can fail where
the swerve start might have succeeded, and such failures count as
`rollout_rejected`. Failed re-solves are still half of final 1's.

### Where the unused budget goes

From the final-3 ledger (600,000 steps), scoring `0.20 − (elapsed − symbolic)`:

| step type | share of steps | share of unused budget |
|---|---|---|
| symbolic ran | 28.6% | 34.5% (not idle: symbolic time is excluded from the score) |
| waiting for symbolic | 14.8% | 18.9% |
| truly quiet (< 5 ms of work) | 16.1% | 20.6% |
| did work, then ran out | 40.4% | 26.0% |

- **Waiting for symbolic.** A symbolic chunk costs ~205 ms, more than a step,
  so `symbolic_step` credits one step's budget to the job and runs the chunk
  on the next.
- **Truly quiet.** The t+1 backup is already in the buffer, concrete has
  reached `MAX_TIME` or the conflict, and there is no conflict for symbolic.
- **Did work, then ran out.** The typical case is NOMINAL with no conflict,
  where P1 builds the single t+1 backup (~33 ms) and P3 builds nothing ahead.

Slack smaller than one MPC price, the only place the pricing reserve can show,
is about 5% of the truly unused time. A traced example (seed 715529158,
`~/runs/alg22_trace/`) shows every case.

### Per-chunk

| chunk | node | CPU | elapsed | safe | resolved / failed / uncertified | steps over 0.20 s | slowest step |
|---|---|---|---|---|---|---|---|
| 1 | node3617 | EPYC 9474F | 47:15 | 2500 / 2500 | 2,424 / 281 / 268 | 0 | 198.3 ms |
| 2 | node3621 | EPYC 9474F | 47:26 | 2500 / 2500 | 2,416 / 302 / 286 | 0 | 196.6 ms |
| 3 | node3622 | EPYC 9474F | 47:22 | 2500 / 2500 | 2,313 / 316 / 297 | 0 | 197.3 ms |
| 4 | node3623 | EPYC 9474F | 47:08 | 2500 / 2500 | 2,259 / 264 / 255 | 0 | 196.6 ms |

### Configuration

```
TTTCARV_ALG=Reach_PSF_FINAL       # was alg22_receding_mpc when the run executed
TTTCARV_RECEDING=1
TTTCARV_ONESHOT=1
TTTCARV_ONESHOT_REFINE=2
TTTCARV_ONESHOT_VERIFY=1
TTTCARV_BOOTSTRAP_RESIM=1
TTTCARV_PSF_LOITER=1
TTTCARV_PSF_TAIL=1
TTTCARV_FIRST_STEP_COST=0
TTTCARV_TAIL_COST_WEIGHT=1e-3
TTTCARV_RESOLVE_RESERVE=0
TTTCARV_SINGLE_START_FRAC=0.50    # inert under pricing: every solve is priced whole
TTTCARV_MPC_PRICING=1             # new
TTTCARV_ONESHOT_RTI=10            # new; code default 20
TTTCARV_BUDGET_MARGIN=0.003       # new
TTTCARV_GC_PER_TRIAL=1            # new; code default 1
TTTCARV_GOAL_KEEPOUT=0
TTTCARV_WORKERS=8
```

Defaults: `TTTCARV_PRICE_QUANTILE=0.995`, `TTTCARV_CONCRETE_QUANTILE=0.95`,
`TTTCARV_PRICE_WINDOW=1000`. With `TTTCARV_MPC_PRICING=0` the budget and the
concrete loop behave as in final 1.

### Provenance

- source: per-file sha256 frozen in `~/runs/alg22_final3/expected.sha256`
  (`alg22_receding_mpc.py`, `REAL_integrated_sim.py`, `time_budget.py`,
  `mpc_safety_filter.py`, `mpc_safety_filter_acados.py`,
  `run_trials_random_obs.py`, `utils/robust_training_utils.py`,
  `utils/unicycle_mpc_acados.py`), checked before and after every chunk,
  unchanged. The same blobs are committed in `83ae8e2` on
  `realtime_integrated`, with `alg22_receding_mpc.py` renamed to
  `Reach_PSF_FINAL.py` (content identical).
- SLURM array 22728795, `mit_preemptable --exclusive`, script refuses any CPU
  but EPYC 9474F, 0 restarts. Smoke test 22728675 (48 trials) first.
- script `~/scripts/carv_alg22_final3.sbatch`; results in
  `~/runs/alg22_final3/`: `fp_{1..4}.json`, `chunk{1..4}.log.gz`,
  `events_{1..4}.txt` (all empty), `budget_{1..4}.tar.gz` (per-step ledger).
- final 2: `~/runs/alg22_final2/`, script `~/scripts/carv_alg22_final2.sbatch`.
- supporting runs: per-op profile 22706513 (`~/runs/alg22_opcost/`); pricing
  arms and node benchmark `~/runs/alg22_pricing/`; concrete diagnostic
  22713530; single-trial trace `~/scripts/carv_psf_trace.sbatch`.

---

## Earlier run: final 1 (jobs 22681480 + 22686859, 2026-09-13)

**10000 / 10000 safe. Zero collisions, zero crashes, zero fallback events.**

Superseded by *Final evaluation: Reach_PSF_FINAL* above: same code path, but
admitted work could run past 0.20 s, and the trial log was not checked per
timestep. It superseded v2 (job 22616750), which is kept below under *Earlier
run: v2*. Same 10,000 scenes as v2 and `results_alg20` (`TTTCARV_SEED = 20260823 + chunk`,
2,500 trials per chunk, `GOAL_KEEPOUT=0`).

| | |
|---|---|
| scenes | 10,000 (4 chunks × 2,500) |
| collisions (true state inside a raw obstacle radius) | **0** |
| crashed trials (`error` rows) | **0** |
| PSF no-diverge fallbacks | **0** |
| PSF queue-empty fallbacks | **0** |
| t=0 backup build failures (`p1_t0_infeasible`) | **0** |
| terminal-loiter steps | **0** |
| no-backup / loiter / unrecoverable / stale-refused log events | **0** (uncapped grep of every worker log) |
| calibration clamps | **0** |
| certificate-spent steps (`mpc_tail_exhausted`) | **0** |
| episode length | 61 timesteps, every trial |
| PSF activations | mean 2.68 / trial, median 2, max 8; 99.2% of trials activated at least once |
| re-sim bootstrap builds | 182,703, of which 14,878 infeasible |
| re-solves | 7,699 succeeded, 2,321 failed; 2,037 uncertified fallback steps |
| wall / timestep | median 0.182 s, p95 0.205 s, max 0.265 s; 1,070 trials with mean > 0.20 s |
| compute | 30.8 CPU-hours, median 11.1 s per trial |

### Configuration

```
TTTCARV_ALG=alg22
TTTCARV_RECEDING=1
TTTCARV_ONESHOT=1
TTTCARV_ONESHOT_REFINE=2
TTTCARV_ONESHOT_VERIFY=1
TTTCARV_BOOTSTRAP_RESIM=1         # new since v2
TTTCARV_PSF_LOITER=1
TTTCARV_PSF_TAIL=1
TTTCARV_FIRST_STEP_COST=0
TTTCARV_TAIL_COST_WEIGHT=1e-3
TTTCARV_RESOLVE_RESERVE=0
TTTCARV_SINGLE_START_FRAC=0.50
TTTCARV_GOAL_KEEPOUT=0
TTTCARV_WORKERS=8
```

`P1_BOOTSTRAP` no longer exists in the code. Worker logs went to node-local
disk, not NFS.

### Per-chunk

| chunk | node | CPU | elapsed | safe | contention spread | probe median | re-solves failed | uncertified | trials > 0.20 s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | node1409 | EPYC 7542 | 1:15:03 | 2500 / 2500 | 1.16× | 0.78 ms | 613 | 526 | 209 |
| 2 | node1409 | EPYC 7542 | 1:14:59 | 2500 / 2500 | 1.15× | 0.78 ms | 640 | 566 | 280 |
| 3 | node1411 | EPYC 7542 | 1:15:47 | 2500 / 2500 | 1.13× | 0.78 ms | 532 | 474 | 335 |
| 4 | node1447 | EPYC 7542 | 1:14:48 | 2500 / 2500 | 1.20× | 0.78 ms | 536 | 471 | 246 |

Chunks 1–2 ran back to back on one exclusive node. To finish sooner, chunks 3–4
were split off to two more exclusive nodes of the **same CPU model** (array job
22686859, which refuses to run on any other CPU), and 22681480 was cancelled
right after chunk 2 was saved. Every chunk used 8 workers on its own node, so
contention is the lowest of any alg22 run (spread ≤ 1.20× vs 1.10–1.40× in v2).

### Re-solve counts depend on the CPU, not on BOOTSTRAP_RESIM

On identical scenes the final run has far more failed re-solves than v2
(2,321 vs 1,355) and more uncertified steps (2,037 vs 1,269). The config
changed only in `BOOTSTRAP_RESIM`, but the hardware changed too: v2's chunks
ran on EPYC 7543 / EPYC 7513 / EPYC 9135 / EPYC 9474F, and on the two Zen 4
chunks (9135, 9474F) it sat at 0.12–0.13 s per step, well under budget. The final run is on
EPYC 7542 throughout, which binds the 0.20 s wall-clock budget in every chunk.

Test B (job 22685570) separates the two causes: chunk-1 scenes on one EPYC 7542
node (node1403), final config, only `BOOTSTRAP_RESIM` changed.

| chunk-1 scenes | CPU | BOOTSTRAP_RESIM | resolved | failed | uncertified | trials > 0.20 s | collisions |
|---|---|---|---|---|---|---|---|
| v2 | EPYC 7543 | 0 | 2,285 | 387 | 361 | 339 | 0 |
| final | EPYC 7542 (node1409) | 1 | 1,942 | 613 | 526 | 209 | 0 |
| test B, arm 0 | EPYC 7542 (node1403) | 0 | 1,914 | 618 | 536 | 239 | 0 |
| test B, arm 1 | EPYC 7542 (node1403) | 1 | 1,938 | 616 | 532 | 220 | 0 |

On the same node the two arms agree within 1.3% on every count, and both
reproduce the final run's chunk 1, not v2. The shift is the CPU, and
`BOOTSTRAP_RESIM` is free on these scenes (both arms: 2500/2500 safe, 0 t=0
failures, 0 events). Wall per step cannot
show the speed gap because the budget caps it (0.182 vs 0.181 s median); the
in-trial probe does (0.78 vs 0.72 ms, about 8% slower).

**Consequence:** re-solve, uncertified and over-budget counts are only
comparable between runs on the same CPU model. Collisions are comparable across
all of them. This run used the slowest hardware of any alg22 evaluation, so its
safety result is the conservative one.

**What "uncertified" measures.** It counts fallback steps where the robot's
current Kalman box is not inside the box the committed plan predicted for that
step. Plan boxes are the starting box copied at the same size along the plan,
and the starting box is a converged Kalman box, so two same-size boxes only nest
when their centres coincide. Almost every failed re-solve therefore becomes
uncertified (2,037 of 2,321). It is a bookkeeping count of the containment test,
not a safety event: collisions are checked on the true state.

### Provenance

- source: `src_hash=39cf242e9aa4d028` (sha256 over `alg22.py`,
  `mpc_safety_filter_acados.py`, `run_trials_random_obs.py`,
  `utils/robust_training_utils.py`, `utils/unicycle_mpc_acados.py`), checked
  before every chunk and unchanged at the end of every chunk. The same blobs are
  committed in `a1d6fce` on `realtime_integrated`; job 22681480 started at
  `bf1bb94` + working copy, just before that commit.
- `alg22.py` was renamed to `alg22_receding_mpc.py` at 12:57 during the run
  (identical content). A temporary symlink `alg22.py → alg22_receding_mpc.py`
  kept the hash check and `import alg22` valid for jobs started after that.
- SLURM 22681480 (chunks 1–2, cancelled after chunk 2 by design) and array
  22686859 (chunks 3–4), all `mit_preemptable --exclusive`, 0 restarts.
- scripts `~/scripts/carv_alg22_final.sbatch`,
  `~/scripts/carv_alg22_final_par.sbatch`, `~/scripts/carv_alg22_bsr_disamb.sbatch`
- results in `~/runs/alg22_final/`: `fp_{1..4}.json`, `chunk{1..4}.log.gz` (per-trial
  obstacle centres), `events_{1..4}.txt` (all empty). Job logs for chunks 3–4 in
  `~/runs/alg22_final_par/`; test B in `~/runs/alg22_bsr_disamb/`.

---

## Earlier run: v2 (job 22616750)

Everything from here down describes v2 and the investigations that led to the
final configuration.

**10000 / 10000 safe. Zero collisions. Zero fallbacks of any kind.**

Same 10,000 scenes as `results_alg20`, byte-identical (see *Scene pairing*).
Same low-contention regime — 8 workers on an exclusive node — because
deployment is one robot with a machine to itself, not a throughput-packed
cluster. The 0.20 s per-timestep budget only means what it claims if each trial
actually gets the compute it was calibrated against.

| | |
|---|---|
| scenes | 10,000 (4 chunks × 2,500), identical to `results_alg20` |
| collisions | **0** |
| PSF no-diverge fallbacks | **0** |
| PSF queue-empty fallbacks | **0** |
| unrecoverable states | **0** |
| terminal-loiter steps | 1 (certified; see *The no-diverge hole*) |
| calibration clamps | **0** workers of 32 |
| episode length | 61 timesteps, every trial (no early aborts) |
| PSF activations | mean 2.47 / trial, median 2, max 7; 99.6% of trials activated at least once |
| compute | 24.6 CPU-hours, median 8.67 s per trial |

## The no-diverge hole, and how it was closed

An earlier 10,000-trial run of this same configuration (job 22600133) recorded
**one `psf_no_diverge` timestep**, on seed 556390336. That branch is not a
designed fallback. It applies the raw NN control at the moment the PSF has
decided that control is unsafe, because no plan is committed at the current
timestep — the code comment on it reads `SHOULD NOT RUN IF WORKING!!!!!`. It is
categorically different from a `path_raw` or `term_D` rejection, where a backup
exists and is ridden. Here there is no backup at all, and the step is
uncertified. One such step voids the claim that the run was filtered, whether
or not it collides.

**Cause.** Not budget starvation, and not the scene. The diagnostic
distinguishes the two cases in one field: `buf[t+1]=absent` means the build was
never attempted, `buf[t+1]=None` means it ran and the acceptance test rejected
it. Every event measured in a healthy run reads `None`, with `mpc_hz` advanced,
and `psf_no_diverge_reasons` records `rejected:term_D`. `p1_skipped_no_budget`
was 0 across 25,000 trials. (Under a starved worker the signature is `absent`
instead — P1's budget gate refusing the build. See *Configuration*.)

The rejection comes from the solver, not the geometry. On seed 746751656, which
reproduces the event deterministically, the t=0 build is INFEASIBLE with
`TTTCARV_ONESHOT=1` and VALID with `ONESHOT=0`; `RECEDING`, `PSF_TAIL` and
`ONESHOT_VERIFY` make no difference, and alg21 — which has no one-shot path —
builds VALID (job 22637397). A plan ending outside `D` exists. The one-shot path
solves the OCP once, so the terminal constraint binds on the endpoint it
returns, and then certifies the *plant* re-simulation of the solver's controls.
When SQP_RTI has not converged, that real endpoint drifts off the solver's
iterate and lands inside `D`. Nothing unsafe is certified — the check runs on
the plant rollout — but the build fails.

It only bites at t=0. Mid-flight a rejected build is absorbed, because
`t_diverge` still points at a valid backup banked earlier. At t=0 the buffer is
structurally empty (`td=never_set`), so one rejection has nowhere to land. That
is why every recorded event sits at t=0 even though one-shot returns a non-OK
acados status on 11–23% of solves, depending on the scenario.

*An earlier draft of this document attributed the rejection to genuine
geometric infeasibility at t=0. The probe above disproves that.*

**Fix — terminal safe controller, Algorithm 1 branch 3.** When no plan is
available the choice is between the raw NN control and a max-turn loiter,
applied for one timestep, after which the filter re-plans. The predicate is the
**current** bound clearing `D = √(r² + 2rR)` = 1.118, not the bound at t+1: the
latter is the RSOA under the nominal control, so it cannot certify a step on
which something else is applied.

**What certifies it is one-step margin, not circle invariance.** One plant step
moves at most `v·dt` = 0.2, so from outside `D` the next state is at least 0.918
from every obstacle centre — clear of `r` = 0.5 whatever the control. `D`'s
stronger property, that a max-turn circle of radius `R` from outside `D` never
enters the obstacle, holds only if the turn is chosen *away* from the obstacle
(worst case exactly `r`, tangent). The implementation always turns left
(`u = +u_max`), and for about 35% of headings on the `D` boundary that circle
passes through the obstacle, so *repeated* loiter steps are not covered by the
argument. *An earlier draft of this document claimed invariance; that was
wrong.* In practice the loiter has fired at most once per trial, and
`psf_unrecoverable` has been 0 everywhere, so the gap has not been exercised.
With `BOOTSTRAP_RESIM` on it has fired 0 times in 12,500 trials. Choosing the
turn direction away from the nearest obstacle would close it.

Branch 3 had been excluded on the measured grounds that
`mpc_tail_exhausted = 0`. That argument covers only "a plan exists and its tail
runs out"; it says nothing about having no plan at all, so it never applied to
this case.

**Evidence.** Paired A/B on identical scenes, 4–8 obstacle scenario where the
event rate runs ~2.5× production (job 22615021):

| arm | loiter | `psf_no_diverge` | `psf_loiter_used` | unrecoverable | collisions |
|---|---|---|---|---|---|
| seed 20260941 | off | 2 | 0 | 0 | 0 |
| seed 20260941 | **on** | **0** | **2** | 0 | 0 |

And in this run, seed 556390336 — the exact trial that produced the original
event — reproduced the condition and was caught:

```
[PSF] No backup at t=0  buf[1]=None  td=never_set  mpc_hz=1  (seed=556390336)
[PSF LOITER] t=0 — current RSOA clears D, holding max-turn loiter u=[1.]
```

The loiter fires **at most once per trial** in every run measured: the filter
recovers on the next timestep rather than circling. `psf_unrecoverable` — no
backup *and* the current RSOA already inside `D` — is 0 everywhere it has been
measured, so the loiter has never been unavailable when needed.

### Preventing the failed build instead of catching it (job 22638543)

The loiter catches a failed t=0 build; it does not prevent one. Two candidate
prevention knobs were A/B'd as a 2×2 on identical scenes: 6,250 trials per arm,
4–8 obstacles, seed 20260941, 32 workers, one exclusive node per arm, loiter on
in every arm.

- `BOOTSTRAP_RESIM=1` runs any P1 build with no valid fallback ahead on the
  re-simulation solver instead of one-shot.
- `ONESHOT_REFINE=3` adds a third fixed-point pass to every one-shot solve.

| arm | t=0 build failures | `psf_no_diverge` | collisions | crashed | re-solves ok / failed | uncertified | wall/step median |
|---|---|---|---|---|---|---|---|
| resim off, refine 2 (control) | 2 | 0 | 0 | 1 | 8,141 / 4,612 | 4,204 | 0.184 |
| **resim on, refine 2** | **0** | 0 | 0 | 2 | 8,107 / 4,726 | 4,292 | 0.180 |
| resim off, refine 3 | 7 | 0 | 0 | 3 | 5,462 / 7,659 | 6,901 | 0.207 |
| resim on, refine 3 | 0 | 0 | 0 | 2 | 4,754 / 8,239 | 7,431 | 0.187 |

t=0 failures are counted from the event log, not `p1_t0_infeasible`: four
trials failed at t=0, loitered, and then crashed, which discards their counters.
Every t=0 failure was caught by the loiter.

Paired per-seed effects, per trial, over the 6,245 seeds that completed in all
four arms:

| | refine 2→3 (resim off) | refine 2→3 (resim on) | resim off→on (refine 2) | resim off→on (refine 3) |
|---|---|---|---|---|
| re-solves succeeded | −0.43 | −0.54 | −0.005 | −0.11 |
| re-solves failed | +0.49 | +0.56 | +0.018 | +0.09 |
| uncertified steps | +0.43 | +0.50 | +0.014 | +0.08 |
| wall/step, median ratio | 1.12 | 1.04 | 0.98 | 0.91 |

Wall ratios are node-confounded (EPYC 7513 vs 7543 across arms). The resim
comparison at refine 3 has the resim arm on the *slower* CPU, so its direction
survives the confound.

- **`BOOTSTRAP_RESIM` — adopted, default 1.** t=0 build failures 2→0 and 7→0.
  No measurable cost at refine 2: re-solves unchanged, wall not higher. It fires
  on every trial, ~18 times, because the no-fallback condition recurs each time
  the robot catches up to `t_diverge`, not only at t=0. That it is still free is
  consistent with re-simulation costing ~31 RTI iterations against one-shot's
  2×20 = 40.
- **`ONESHOT_REFINE=3` — rejected, stays 2.** It fixes seed 746751656 but is
  worse in every other column: t=0 failures 2→7, about half a re-solve per trial
  moved from succeeded to failed, and +12% wall, with over-budget trials 579 →
  4,504 despite the faster CPU.
- **Loiters precede crashes.** Of the 9 trials that loitered at t=0 in the two
  resim-off arms, 4 then crashed in reachability at t=2–5; of the ~24,990
  trials that did not loiter, 4 crashed. Removing the loiter's trigger removes
  that route, though the resim arms crash on two seeds of their own, so the net
  crash count is not lower (1 / 2 / 3 / 2). See *Known defect*.

## Head-to-head against alg20, same scenes

| | alg20 | alg22 |
|---|---|---|
| collisions | 0 | 0 |
| no-diverge fallbacks | 0 | 0 |
| queue-empty fallbacks | 0 | 0 |
| trials completing 61 steps | 10,000 | 10,000 |
| wall / timestep, median | 0.132 s | 0.142 s |
| trials over budget | 10 (0.10%) | 346 (3.5%) — see caveat |

**alg22 does not beat alg20 on safety, and this run cannot show that it would.**
Both are saturated at 10,000 / 10,000. The scene set has no discriminating
power left; it was already solved before the receding horizon was added. What
this run establishes is narrower: the receding horizon costs nothing in safety,
and the one certificate gap either algorithm exhibited on these scenes is now
closed.

**Timing caveat.** The 346 over-budget trials are not attributable to the
loiter, which fired once. 339 of them are in chunk 1 alone, the chunk with the
highest contention spread (1.40×), and this run had `TTTCARV_WORKER_LOG_DIR`
enabled — every worker writing its full per-timestep trace to a shared
filesystem. The previous run of the same configuration without worker logging
recorded 5 over-budget trials. Worker logging was enabled deliberately: the
original event's cause was lost to `/dev/null` because it was not, and the
diagnostic for this branch is printed from the worker.

## Receding-horizon behaviour

| counter | total | meaning |
|---|---|---|
| `mpc_calls` | 24,746 | MPC activations |
| `mpc_resolved` | 9,191 | fresh solve from the current reachable set succeeded and was applied |
| `mpc_resolve_failed` | 1,355 | re-solve ran, rollout rejected by the acceptance test |
| `mpc_fallback_uncertified` | 1,269 | applied input came from the committed plan while the RSOA lay outside `traj_bounds[ctrl_idx]` |
| `mpc_tail_exhausted` | 0 | branch-2 shifted tail ran out — never once |
| `psf_loiter_used` | 1 | branch-3 terminal controller |
| `psf_stale_refused` | 0 | stale-plan activation refused |

Re-solve success rate is 9191 / 10546 = **87.2%**. A rejected rollout is the
acceptance test working, not a fault: the fresh solve was not certifiable and
the committed plan is ridden instead. The production driver does not print
rejection labels, so this run does not break them down; four diagnostic runs
that do print them put `path_raw` at nodes 1–5 at 91–92% and `term_D` at node
12 at 8–9%.

**Homotopy reversals.** `plan_flips = 176 / 8217` plan pairs (2.1%) and
`applied_u_flips = 79 / 4531` applied-input pairs (1.7%). A re-solve sometimes
reverses which side of an obstacle the plan passes, because the warm start is a
fresh NN rollout rather than the previous solution, so nothing carries the
left/right choice forward. Every such reversal is individually certified; the
rate is reported because a reversal is visible as a direction change in the
rendered animations and would otherwise look like a fault.

## Scenario

Identical to `results_alg20`:

| | value |
|---|---|
| system | unicycle, `dt = 0.2 s`, NN controller `Unicycle_NL / natural_none_default` |
| start set | `x ∈ [-9.55, -9.45]`, `y ∈ [3.45, 3.55]`, `θ ∈ [-π/24, π/24]` |
| goal | origin |
| obstacles | 1–5 per scene, uniform count, radius 0.5 |
| obstacle centres | uniform in `x ∈ (-7.0, -1.0)`, `y ∈ (-1.0, 2.0)` — **unrestricted**, no keep-out |
| horizon | 60 timesteps max |
| MPC | acados, 12-step horizon, split-terminal (PATH nodes clear raw radius `r`, TERMINAL clears danger region `D`) |
| compute budget | 0.20 s per timestep, costs calibrated once per worker process |

## Scene pairing

The 10,000 trial seeds match `results_alg20/fp_{1..4}.json` **exactly and in
order**, verified after the run. `TTTCARV_GOAL_KEEPOUT=0` is what makes this
hold: at 0 the rejection test in `_gen_obstacles` never rejects, so the
generator consumes an identical RNG stream.

**This pairs the scenes, not the trajectories.** Admission to the 0.20 s budget
depends on wall-clock, so which branch of the active phase supplies the applied
input shifts with machine speed and contention. Measured directly, not assumed:
re-rendering one seed with identical obstacles and config on a different node
moved `mpc_calls` from 2 to 3 on seed 703745641 and from 4 to 2 on seed
1998241395. A per-seed step-by-step diff against alg20 would be reading noise.
Outcomes and aggregates are the comparable quantities.

`GOAL_KEEPOUT=0` also reinstates obstacles near the origin, where the NN
controller is untrained, so a collision here could be the controller's endgame
rather than a filter failure. That confound is the price of comparability with
alg20 and is deliberate; it did not arise.

## Per-chunk

| chunk | node | seed | elapsed | safe | contention spread | probe median | over budget |
|---|---|---|---|---|---|---|---|
| 1 | node2309 | 20260824 | 1:11:29 | 2500 / 2500 | 1.40× | 0.72 ms | 339 |
| 2 | node2602 | 20260825 | 1:01:44 | 2500 / 2500 | 1.34× | 0.70 ms | 7 |
| 3 | node4004 | 20260826 | 44:23 | 2500 / 2500 | 1.16× | 0.34 ms | 0 |
| 4 | node2502 | 20260827 | 50:17 | 2500 / 2500 | 1.10× | 0.59 ms | 0 |

`contention spread` is max/min of a fixed 256×256 matmul timed inside each
trial under whatever load was really present. The over-budget trials track it
directly and are confined to the two chunks with the highest spread.

## The nine historically-failing seeds

These nine collided under earlier algorithm versions on byte-identical scenes.
All nine are safe under alg22, with no rejected re-solve, no uncertified
fallback and no loiter on any of them:

| seed | MPC activations | re-solves applied |
|---|---|---|
| 242244880 | 2 | 1 |
| 856113296 | 3 | 1 |
| 982114962 | 1 | 4 |
| 1091579819 | 2 | 1 |
| 1471202357 | 3 | 1 |
| 1844520186 | 2 | 2 |
| 1956405985 | 3 | 1 |
| 2095399857 | 4 | 1 |
| 2132122960 | 3 | 1 |

As with alg20, attribution is not measured — these nine were already safe under
alg20 on this scene set, so alg22 is not the reason they pass.

## What alg22 adds over alg20

alg20 and alg21 commit to a backup plan at activation and fly it open loop for
the whole intervention. The applied input at step `k` is certified against the
reachable set computed at `k̄`, the activation time, propagated forward and
looser than the truth by the time it is used.

alg22 re-solves from the current reachable set at each active timestep (step 4a
of the active phase). When it succeeds — 87.2% of the time — the applied input
carries a certificate computed at `k`, not at `k̄`. When it fails, alg22 falls
back to exactly alg21's behaviour: ride the committed plan. Safety therefore
cannot be worse than alg21 by construction; the fallback path *is* alg21.

The gain is the tightness of the certificate behind each applied input, which
this scene set cannot measure — both algorithms are saturated. The difference
is shown instead in the rendered animations (`~/runs/alg22_anim/`), where the
same scene with `TTTCARV_RECEDING=1` and `=0` gives 4 re-solves versus 0.

## Configuration

```
TTTCARV_ALG=alg22
TTTCARV_RECEDING=1
TTTCARV_ONESHOT=1
TTTCARV_ONESHOT_VERIFY=1
TTTCARV_PSF_TAIL=1
TTTCARV_FIRST_STEP_COST=0
TTTCARV_TAIL_COST_WEIGHT=1e-3     # inert while FIRST_STEP_COST=0
TTTCARV_RESOLVE_RESERVE=0
TTTCARV_SINGLE_START_FRAC=0.50
TTTCARV_PSF_LOITER=1              # terminal safe controller; code default
TTTCARV_P1_BOOTSTRAP=1            # used in this run; flag since REMOVED from alg22.py
TTTCARV_GOAL_KEEPOUT=0            # 1.5 outside the alg20-paired comparison
TTTCARV_WORKERS=8                 # part of the configuration, not a convenience
```

`ONESHOT`, `ONESHOT_VERIFY` and `PSF_TAIL` default to `0` in code and must be
set explicitly. `PSF_LOITER` defaults to `1`.

Three knobs were A/B'd at 10,000+ trials each, one node, alternating or paired
blocks. Two measured **inert**, one measured **decisive**:

- `RESOLVE_RESERVE` — **inert**. Reserving budget so the re-solve cannot be
  starved by the revert-check rebuild loop. Paired delta on active steps −0.6% /
  +0.3%; `no_budget` 5 vs 5. Inert because the loop's first build *is* the
  revert test, and later builds only happen when the revert will fire, so they
  can never compete with the re-solve. Left at `0`.
- `SINGLE_START_FRAC` — **inert**. Assumed cost of a single-start solve as a
  fraction of a full build, 0.25 vs 0.50. Collisions 0 vs 0, uncertified 265 vs
  265 on the paired seed, `resolved` within 0.7%, budget overruns
  indistinguishable with the sign flipping between seeds. Set to **0.50**
  because it is the conservative side of a knob that measures inert, so
  conservatism is free.
- `PSF_LOITER` — **decisive**, and the only one that changed a safety-relevant
  quantity. See *The no-diverge hole*.

`P1_BOOTSTRAP` let P1's build at t+1 exceed the timestep budget when no valid
backup existed ahead. It was on for this run and measured **inert in healthy
runs** (`p1_skipped_no_budget = 0` across 25,000 trials, `p1_forced_bootstrap =
0` here). **It has since been removed.** Its only target was the starved-worker
case, where the budget gate refuses the build (`buf[t+1]=absent`); the decision
is to rule that case out by running the final evaluation on a single
uncontended node rather than to absorb it in the filter. That does *not* touch
the healthy-run `term_D` rejection, which reproduces as a lone process on a
dedicated node (job 22638437) — see *The no-diverge hole*.

## Worker count is part of the configuration

A load sweep (3 arms × 5,000 trials, same seed and config, 64-core allocation)
measured 0 calibration clamps at 16 and 32 workers against 3 of 64 at 64
workers. Each clamp is a worker whose once-per-process `mpc_cost` calibration
landed too high to ever afford a build — before the guard existed such a worker
ran with the filter silently disabled for its entire share of the run. Every
collision on record in the instrumented era came from one of those workers;
with the filter verifiably running the record is 53,607 trials and 0 collisions.
**Do not exceed 32 workers per 64-core allocation.**

**Clamp threshold corrected 2026-09-12.** The guard tested
`mpc_cost >= timestep_budget`, but P1 must first propagate concrete bounds to
t+1 before it can build, so the real requirement is
`mpc_cost + concrete_cost <= timestep_budget`. A measured worker at
`mpc=0.1782s, concrete=0.0235s` passed the old test, then spent 0.0259 s on
bounds and failed `0.174 >= 0.1782` — it could never build at t=0 and produced
7 of the 11 no-diverge events in its run. The old clamp *target*, `0.9 × 0.20 =
0.18`, fails the same test at `0.18 + 0.0235 > 0.20`, so the guard could clamp a
worker into the state it existed to prevent.

This run: 0 clamps in 32 worker processes.

## Provenance

- code: commit `bf1bb94`, logs stamp `dirty=28 files`
- driver: `nfl_robustness_training/src/run_trials_random_obs.py`, algorithm
  `nfl_robustness_training/src/alg22.py`
- SLURM job **22616750**, array 1–4, `mit_preemptable`, `--exclusive`
- `TTTCARV_TRIALS=2500`, `TTTCARV_WORKERS=8`, `TTTCARV_SEED=20260823+chunk`,
  `TTTCARV_WORKER_LOG_DIR` set
- raw logs `~/runs/alg22_10k_v2/chunk{1..4}.log.gz` (they hold the per-trial
  obstacle centres and passthrough detail the fingerprints do not);
  per-trial fingerprints `fp_{1..4}.json`; branch diagnostics
  `~/runs/alg22_10k_v2/lines_{1..4}.txt`
- supporting A/Bs: 22615021 (loiter), 22612045 (P1 bootstrap), 22598979
  (single-start frac), 22595209 (resolve reserve), 22581818 (load sweep)
- animations `~/runs/alg22_anim/`, SLURM job 22599238
- superseded first run of this configuration: job 22600133, identical scenes,
  1 `psf_no_diverge` timestep

**`alg22.py` is untracked in git at the time of writing.** The commit hash pins
the driver and surrounding library, not the algorithm file. Reproducing this run
requires the working copy of `alg22.py`, not just `bf1bb94`.

## Known defect, out of scope for this run

A trial can hit a NaN in the reachable-set bounds during a symbolic step. It
first surfaced inside auto_LiRPA's backward propagation as
`IndexError: index -9223372036854775808` — `INT64_MIN`, a NaN cast to an
integer — on the dense 4–8 obstacle scenario (job 22615021 task 4 and its
re-run). It has not been observed on the 1–5 obstacle production scenario in
35,000+ trials. At the time, one crash took down the whole `imap_unordered`
pool, and the sbatch still reported `COMPLETED` because the script ended with a
trailing `echo`.

Both halves have since been addressed in code, after this run. Bounds are
checked before auto_LiRPA is called (`ReachableSetBoundsError: reachable set at
t=… is not a valid interval, refusing to call auto_LiRPA`), and the driver
isolates per-trial exceptions: a failed trial is excluded from the safety record
and reported as `!! N/M trials FAILED`. Its fingerprint row still reads
`collision=False`, with no stats and `wall_per_timestep_s=nan`, so analyses must
filter on the row's `error` key — `nan` passes a truthiness test and corrupts
medians.

**The evidence points to path-dependence, not timing** — correcting an earlier
reading here. In the 2×2 A/B (job 22638543, same scenes in every arm), both
`BOOTSTRAP_RESIM=1` arms crashed on the *same* two seeds at the *same*
timesteps (1789447268 at t=2, 2022753759 at t=6) and completed seed 771627023;
the control arm did the reverse, crashing on 771627023 at t=2, immediately after
a t=0 loiter, and completing the other two. The same flown controls reproduce
it. The earlier crashes at different trial indices on identical scenes are
consistent with this, because wall-clock budget admission makes the flown
trajectory itself vary between runs. Rate: 1–3 per 6,250 trials on the 4–8
obstacle scenario.

**Aggressive control appears to trigger it.** In the same A/B, 4 of the 9
trials that applied the max-turn loiter at t=0 crashed within the next five
timesteps, against 4 of ~24,990 that did not loiter. The loiter is the hardest
turn the plant can make. This matters beyond the crash itself: a crashed trial
is excluded from the safety record, so a loiter followed by a crash is an
unevaluated trial, not a demonstrated recovery.

## Fingerprint limitation

Obstacle layouts are not stored in the fingerprints — only in the chunk logs.
Recovering the layout for a given seed means replaying the generator's RNG
stream, which was done for the animation case selection and verified against all
5,000 trial seeds of that run. Adding layouts to the fingerprint would remove
that step.
