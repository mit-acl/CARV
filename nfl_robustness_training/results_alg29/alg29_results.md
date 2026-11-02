# alg29 — generic PSF baselines under computation latency and state noise

**Question.** A textbook predictive safety filter (PSF) solves its OCP from the
measured point x_t and applies the result "at t". A real solve takes δ seconds.
Does the generic PSF fail once δ > 0? Is its cached backup still able to save
the robot by the time it falls back on it? And how does ours (alg28, set-based,
backup built for t+1 ahead of time) compare under the same timing?

Run: `~/runs/alg29_main` (2026-09-21). Array job 23408567 ran the deterministic
fixed-δ arms. Job 23408568 ran every wall-clock-priced arm on one exclusive
node, sequentially, with 8 workers.
Sources were pinned by snapshot: `alg29_generic_psf` 6b512633…, `alg28_sound`
537c00a8…. Unicycle scenes: seed 20260824 with GOAL_KEEPOUT=0, so the first
500 scenes of the alg28 batch stream are paired across all arms. DI: 500
trials, seed 20260824, start set p ∈ [0.55, 0.80], v ∈ [−0.95, −0.85].

## Rev 8 (2026-10-16): the intervention gap is mostly a measurement artefact — READ BEFORE QUOTING 2.4×

**Question.** Rev 7/7b report R2 − ours = +7.4 pp at |applied − NN| > 0.05
(12.11% vs 4.74%, uniform, path-only) and that is quoted as "ours changes the
NN ~2.4× less often". R2 ties ours on safety, so this one number is the whole
differentiator. Two checks: score both sides on a *threshold-free* count taken
from each filter's own mode bookkeeping instead of from |Δu|, and strip the
two reference asymmetries the |Δu| metric carries.

**Answer: the +7.37 pp is mostly how the metric is built.** About 3.1 pp is
the ±1 input clip and about 2.7 pp is scoring the two sides against estimates
of different freshness. Three corrected metrics then agree at roughly
**+1 pp, about 1.2×**, and they disagree with each other by less than they
disagree with 7.37:

| corrected metric | R2 − ours, uniform | trunc |
|---|---|---|
| pinned reference + clip (17:45 addendum, ours at `USAT=0`) | +1.45 [1.23, 1.68] | +1.22 [0.99, 1.47] |
| per-step engaged count (rev 8b) | +0.99 [0.75, 1.22] | +1.09 [0.80, 1.39] |
| \|applied − clip(NN)\| scored on both sides (rev 8b) | +0.99 [0.63, 1.36] | +0.02 [−0.44, 0.47] |

Two further things a reader needs: corrected, R2 is higher in only **48–54%**
of scenes against 96% on the raw metric, so this is a small average effect and
not a scene-by-scene one; and on the third metric the trunc arms are a **dead
tie**. The 2.4× does not survive on any reading.

*(The −0.57 pp "R2 engages less often" reading further down came from a proxy
that undercounts R2 by 1.7 pp. Rev 8b corrects it, in R2's direction.)*

Data: the completed runs `~/runs/alg39_pinfan` (ours, path-only) and
`~/runs/alg41_score` (baselines, path-only, with the alternative references
scored in-module). Both are 500 scenes, seed 20260824, `GOAL_KEEPOUT=0`,
scene-aligned trial for trial (checked). Collisions are 0 in every arm below.

### The four candidate asymmetries, checked against the code

1. **alg28 appends a literal `0.0` on every pass-through step**
   (`alg28_sound.py:2574` nominal, `:2611` passthrough, `:2086` revert,
   `:2688` queue-empty, `:2759` no-diverge) while alg29 computes a real-valued
   difference every step (`alg29_generic_psf.py:822`). **Does not survive.**
   Each of those five sites is immediately followed by
   `tester.real_state_empirical(...)`: the NN's own command is what flies, so
   `0.0` is the exact value, not a stand-in. The asymmetry is confined to the
   `> 0` column, where acados returns the pinned input only to solver
   tolerance and the baselines read 97–100% against ours' 4.9% — already
   flagged in rev 6b. It does not touch `> 0.01` and above.

2. **Mid-step plan landings make an exact match impossible for the baseline.**
   **Does not survive; it never applied.** Every rev 7 arm sets
   `TTTCARV_GP_ALIGN=1`, and with `dt = 0.2`, `delay = 0.1` the alignment gives
   `delta = ceil(0.1/0.2 − eps)·0.2 = 0.2 = dt`, so `m = 1` and the fractional
   part `f = 0`. The landing is exactly on the tick, `pieces` holds one entry,
   and `u_avg = dth/dt` *is* the landed command. G0 and R1 run `delta = 0` and
   also land on the tick. No rev 7 or 7b arm has a mid-step landing.

3. **The `nn(x_pred)` vs `nn(x̂)` reference mismatch** (`:207`). **Survives;
   second largest term.** R2's pinned plan certifies `nn` at the state its OCP
   planned from, and the metric compares the landed command with `nn` at the
   estimate at the landing tick. Scoring R2 against its own pinned reference
   moves it 12.11 → 9.38.

4. **The ±1 input clip — not in the original list, and the largest single
   term.** The pin sets `lbu = ubu = clip(nn, ±U_MAX)`
   (`alg29_generic_psf.py:538`); ours flies the NN unclipped
   (`TTTCARV_USAT=0`). The NN leaves ±1 on 3.8% (uniform) and 4.8% (trunc) of
   the baselines' steps (`gp_nn_over`, measured in rev 8b with USAT=0), and on
   each of those the baseline is charged a large deviation on a step where its
   filter overrode nothing. Removing it moves R2 12.11 → 8.99. The peak the NN
   reaches is not reliably measured: `u_raw_max` is only recorded under
   `USAT=1`, which the 19:50 retraction shows is unsound in mode J. Rev 2's
   "up to 1.75" is the last sound observation and is probably an understatement.

### Levels (percentage of steps, 500 scenes, path-only)

`clip` = both sides scored as |clip(applied) − clip(NN)|. The baselines'
column is measured; ours was not scored that way in `alg39_pinfan`, so its raw
value is carried over and bounded with "≤" (clipping is 1-Lipschitz, so
|clip(a) − clip(b)| ≤ |a − b|). The 17:45 addendum tried to replace those
carry-overs with measured `USAT=1` values, but its 19:50 retraction withdraws
every `USAT=1` figure, so ours' clip column is still a bound, not a
measurement. **Read the addendum and its retraction before using the `clip`
column at all: it is not a like-for-like comparison.** `pinned ref` = also scored
against the NN at the state the plan was pinned at, not at the landing tick.
`engaged` = threshold-free, from the filter's own bookkeeping: for ours, steps
that flew a backup control rather than the NN (`u_diff_n_active`, i.e. the
MPC_ACTIVE / activate / loiter branches); for the baselines, solves whose
pinned stage-0 input was refused (`pin_tries − pin_ok`), which is the same
question — did the filter accept the NN's input or substitute its own.

| arm | raw > 0.05 | clip | pinned ref + clip | **engaged** | \|NN\| > 1.05 |
|---|---|---|---|---|---|
| **ours pipelined, uniform** | **4.74** | ≤ 4.74 | ≤ 4.74 | **4.77** | — |
| R2 W&Z tube + comp. | 12.11 | 8.99 | 6.26 | **4.20** | 3.32 |
| G2 textbook + comp. | 11.12 | 8.39 | 5.65 | 3.38 | 2.90 |
| R1 W&Z tube, δ=0 | 8.16 | 4.82 | 4.82 | 4.34 | 3.53 |
| G3 box-tightened, δ=0 | 7.80 | 4.52 | 4.52 | 3.80 | 3.44 |
| G0 textbook, point | 7.25 | 4.17 | 4.17 | 3.50 | 3.22 |
| **ours pipelined, trunc** | **6.55** | ≤ 6.55 | ≤ 6.55 | **6.62** | — |
| R2, trunc | 14.54 | 10.49 | 7.85 | 6.13 | 4.25 |
| R1, trunc | 11.04 | 6.49 | 6.49 | 6.38 | 4.74 |
| G0, exact state | 7.20 | 4.08 | 4.08 | 3.47 | 3.23 |
| G2 δ=0.1, exact state | 8.10 | 5.70 | 5.70 | 3.39 | 2.55 |

### Paired against ours, same 500 scenes, bootstrap 95% CI

The `engaged` column here is proxy-based and **superseded by rev 8b for the
baselines** (the proxy undercounts R2 by 1.7 pp and the sign flips); the other
three columns stand.

| baseline | raw > 0.05 | clip | pinned ref + clip | **engaged** (proxy, see rev 8b) |
|---|---|---|---|---|
| R2 − ours, uniform | +7.37 [6.90, 7.87] | +4.25 [3.99, 4.52] | +1.53 [1.31, 1.76] | **−0.57 [−0.80, −0.34]** |
| R2 − ours, trunc | +8.00 [7.44, 8.59] | +3.95 [3.69, 4.21] | +1.30 [1.07, 1.53] | **−0.50 [−0.78, −0.21]** |
| G2 − ours, uniform | +6.39 [5.95, 6.85] | +3.66 [3.40, 3.92] | +0.91 [0.69, 1.13] | −1.39 [−1.61, −1.18] |
| R1 − ours, uniform | +3.42 [3.02, 3.85] | +0.09 [−0.14, 0.32] | +0.09 [−0.15, 0.32] | −0.43 [−0.66, −0.20] |
| R1 − ours, trunc | +4.49 [3.95, 5.07] | −0.05 [−0.30, 0.21] | −0.05 [−0.30, 0.20] | −0.24 [−0.51, 0.04] |
| G3 − ours, uniform | +3.06 [2.63, 3.49] | −0.22 [−0.44, 0.01] | −0.22 [−0.44, 0.02] | −0.97 [−1.19, −0.76] |
| G0 − ours, uniform | +2.52 [2.13, 2.92] | −0.57 [−0.77, −0.37] | −0.57 [−0.77, −0.37] | −1.27 [−1.48, −1.05] |

### How much is artefact

Of R2 − ours = **+7.37 pp** (uniform):

| term | pp | what it is |
|---|---|---|
| ±1 input clip | **3.12** | artefact. The baseline's pin cannot request \|ω\| > 1; ours' NN can and does. Scoring charges the baseline for an input-set difference, not a filter decision. |
| stale NN reference | **2.72** | artefact (established by the 17:45 addendum below). Both sides act on information one tick old, but only R2 is scored against the fresher estimate; ours is scored against the same stale box centre it flies. |
| residual at matched reference | **1.53** | the like-for-like difference, and the only part that is the filter choosing differently. Even this is 2.1 pp *above* R2's own engaged count (6.26 vs 4.20), i.e. partly steps where an accepted pinned plan did not reach the plant unchanged. Not fully attributed. |
| **mode-based (engaged)** | **−0.57** | R2's filter refuses the NN's input slightly *less* often than ours does. |

**Against the δ=0 idealisations the behavioural gap vanishes entirely.** Once
the clip is removed, R1 (+0.09) and G3 (−0.22) are statistically
indistinguishable from ours, and G0 is *better* than ours (−0.57). The rev 7
"~1.7× vs the δ=0 baselines" claim does not survive at all.

### What the headline claim should be

- **Do not write "ours changes the NN 2.4× less often."** On the question that
  phrasing implies — how often does the safety filter override the policy —
  the per-step count (rev 8b) gives R2 5.86% against ours 4.87%, **+0.99 pp
  [0.75, 1.22]**. Real, in ours' favour, and a seventh of the headline.
- **Do not claim it holds scene by scene.** Corrected, R2 is higher in 48–54%
  of scenes. The "higher in 95% of scenes" line belongs to the raw metric and
  measures the clip and the reference mismatch, not the filter.
- **Quote the trunc caveat.** On the symmetric action-space metric
  (|applied − clip(NN)|, both sides) the trunc arms are a dead tie:
  +0.02 pp [−0.44, 0.47]. The advantage is not robust across noise laws on
  every reading of the metric.
- **Do not quote the clip column either (1.9×).** An earlier draft of this
  section called it defensible on the grounds that ours "passes the live
  command through". In the pipelined deployment it does not — see the 17:45
  addendum. With `MEAS_DELAY=1` ours' box at t is F(S_{t−1} ∩ z_{t−1}, u_{t−1})
  (`alg28_sound.py:195`), it flies the NN at that box's centre and is scored
  against that same value, so it is exactly as stale as R2's x̂_pred. The clip
  column charges only R2 for a staleness both sides have.
- **What is defensible: about 1.2–1.3×, on the like-for-like column.** Each
  side scored against the NN at the state its own filter actually used:
  R2 − ours = **+1.45 pp [1.23, 1.68]** (uniform) and **+1.22 pp [0.99, 1.47]**
  (trunc), against ours at `USAT=0`; R1 ties at +0.11. The mechanism is
  latency, not conservatism, and that framing is still the paper's thesis —
  but the size is a one-percentage-point story, not a 2.4× one.
- **The action spaces are not matched in the runs, but they can be matched in
  the scoring.** Ours flies |ω| well above 1 against the baselines' hard ±1,
  and the obvious fix — saturating ours — is unavailable, since the 19:50
  retraction shows `TTTCARV_USAT=1` is unsound in mode J. Rev 8b's third
  metric gets around it without touching either controller: score both sides
  as |applied − clip(NN, ±1)|, charging ours for leaving the input set the
  baseline is held to. That puts ours at 8.00 / 10.47 against R2's
  8.99 / 10.49 — +0.99 pp uniform and a tie under trunc. Report it alongside
  the pinned-reference column.
- **But it matches the action spaces in the scoring, not in the behaviour.**
  Ours still *flies* |ω| > 1 in these runs, so its trajectory, and therefore
  which conflicts arise and when the filter engages at all, is not the
  trajectory a saturated ours would have. Holding a trajectory fixed,
  saturating ours could only lower its score on this metric (its pass-through
  steps would be charged 0 instead of |NN| − 1), so +0.99 pp is a lower bound
  on the gap *for these trajectories*. That argument does not carry over to a
  real saturated run, because saturation changes which states are visited.
  **Ours saturated remains unmeasured, and this metric does not bracket it.**
  It is the best available comparison, not a substitute for fixing the
  bounder.
- **Say plainly that the gap disappears against δ=0 baselines.** That is the
  cleanest evidence that the effect is latency and not conservatism, and a
  reviewer will find it anyway.
- **Fixing the action space is now a code task, not a run.** `TTTCARV_USAT=1`
  looked like the answer at 17:45 and was retracted at 19:50: in mode J
  `decoupled_step` returns `joint_step(t)` (`alg28_sound.py:801`) before the
  `_sat` at `:804`, and `joint_bounds` never saturates inside the CROWN graph,
  so the lookahead box is built from the unsaturated NN while the plant flies
  the saturated one — `est_look_truth_out` goes 0 → 217–256 per 500 trials.
  Ours saturated is **unmeasured** until the joint bounder saturates inside
  the graph. Until then, quote the pinned-ref column and state the caveat.
- **Intervention is still only a proxy.** Task progress is not measured.

### Reproducing

- Levels and CIs: `~/scripts/alg42_modemetric_cmp.py`, and the inline
  bootstrap over `runs/alg39_pinfan/fp_o_pipe{,_trunc}_fan.json` against
  `runs/alg41_score/fp_b_*.json`.
- `alg29_generic_psf.py` and `alg28_sound.py` now carry threshold-free
  counters of their own (`gp_eng_steps` / `eng_steps`, plus `nn_over` and
  `u_diff_clip_gt`), so the mode metric no longer has to be reconstructed from
  `pin_tries`/`u_diff_n_active`. Bookkeeping only; verified no behaviour
  change by the rev 8b controls reproducing 12.11 / 8.99 exactly.
- Rev 8b levels and paired CIs: `~/runs/alg42_modemetric`, arms in
  `arms_base.txt` / `arms_base2.txt` / `arms_ours.txt`, jobs 26108091,
  26118265 and 26108092. Ours ran on one allocation (wall-clock priced);
  baselines are fixed-δ and node-independent.

### Rev 8b (2026-10-16, 19:40): the per-step engaged count, and a bug in R2's exact-state path (`~/runs/alg42_modemetric`)

Array job 26108091, three arms, all rc=0. Sources pinned in
`alg42_modemetric/src`: `alg29_generic_psf` 7a1018e6…, `alg28_sound`
7de495a8…, `mpc_safety_filter_acados` 378fa856…. Both modules now carry a
real per-step engaged counter (`gp_eng_steps` / `eng_steps`) instead of the
two proxies used above, plus `nn_over` and `u_diff_clip_gt`.

**The patch is bookkeeping-only, confirmed.** The uniform and trunc arms
reproduce the numbers above to the last digit: 12.11 / 14.54 raw and
8.99 / 10.49 clip.

**1. The `pin_tries − pin_ok` proxy undercounts R2's engagement by 1.7 pp, and
the sign of the mode comparison flips.**

| R2 arm | raw > 0.05 | clip | **engaged (per step)** | pin-refused (proxy) |
|---|---|---|---|---|
| uniform | 12.11 | 8.99 | **5.86** | 4.20 |
| trunc | 14.54 | 10.49 | **7.77** | 6.13 |

The proxy counts *solves* whose pinned plan was accepted. The per-step counter
asks what actually flew, and catches the steps where an accepted plan did not
reach the plant unchanged — `cache.u(k)` returning None so the previous
command is held, the fallback and loiter steps, and the t=0 startup step where
no plan has landed yet. That is the 2.1 pp of unattributed pinned-reference
steps flagged above: it is real engagement, not a scoring artefact.

Ours' proxy has no such gap — its pass-through branches append exactly 0.0, so
`u_diff_n_active` can only miss an engaged step whose command coincides with
the NN bit-for-bit. **Taking ours at its proxy value (4.77% uniform, USAT=0),
R2 − ours on the mode metric is about +1.1 pp, not −0.57 pp.** The paired
figure needs ours' own per-step count; job 26108092 is still PENDING behind
fair-share. The earlier −0.57 [−0.80, −0.34] in this section, and the −0.68 /
−0.53 in the 17:45 addendum, both come from the biased proxy and should not be
quoted — the addendum's pair is doubly void, since its 19:50 retraction
withdraws every `USAT=1` figure as well.

This *strengthens* the overall reading rather than weakening it: two
independent metrics now agree. Like-for-like |Δu| gives +1.45 pp and the
threshold-free mode count gives about +1.1 pp — both far from +7.4, and both
consistent with ~1.2–1.3×.

**2. R2 cannot be run with the noise off as the code stood — a latent bug, now
fixed.** The first exact-state arm returned 269/500 collisions and 57.96%
intervention. That is not a property of R2. In the delay-compensation
prediction loop, `Sp` was propagated only when `NOISE != 'none'`
(`alg29_generic_psf.py`, the `G2 / G3 / R2` branch), but `solve()` takes the
ROBUST modes' start state from the *box*, not from `xs`. With noise off the
box was never advanced, so R2 planned from x_k and applied the result at
k+1 — uncompensated latency, which is exactly the G1 failure mode and exactly
what the collision count shows.

- Scope: R2 with `NOISE=none` only. R1 is naive and calls `solve(xh, S, k)`
  with no prediction loop; G2 is not ROBUST and takes `ctr = xs`; G3 runs at
  `delay = 0.0`, so `m = 0` and nothing needs propagating. **No published
  number is affected** — every R2 arm quoted anywhere in this file runs with
  noise on, where `Sp` was already propagated correctly.
- Fix: propagate `Sp` unconditionally. With `NOISE='none'` the box is the
  degenerate `[x, x]`, so it is the deterministic step and nothing widens.
  `alg29_generic_psf` is now 339b78f4….
- The broken fingerprint is kept at
  `alg42_modemetric/broken_nonefix/fp_m_r2_0.1_none.json`.
- Rerun submitted as job 26118265 (`arms_base2.txt`): the fixed exact-state R2
  arm, a uniform control that must still read 12.11, and a `G2` exact-state
  arm as the reference the fixed R2 arm should land near (G2 exact-state is
  8.10 raw / 5.70 clip / 3.39 pin-refused). Still pending.

**3. Both reruns landed (jobs 26118265 and 26108092, all arms rc=0). The fix
is validated and the picture is settled.**

With `Sp` propagated, R2 with the noise off becomes G2 with a zero tube, which
is what it should be — the two arms agree to 0.04 pp on every column:

| arm | coll. | raw > 0.05 | clip | engaged | pin-refused |
|---|---|---|---|---|---|
| R2 δ=0.1, exact state (fixed) | **0** | 8.09 | 5.67 | 5.02 | 3.35 |
| G2 δ=0.1, exact state (reference) | 0 | 8.10 | 5.70 | 5.06 | 3.39 |
| R2 δ=0.1, uniform (control) | 0 | 12.11 | 8.99 | 5.86 | 4.20 |
| **ours pipelined, uniform** | **0** | **4.82** | **8.00** | **4.87** | — |
| **ours pipelined, trunc** | **0** | **6.61** | **10.47** | **6.68** | — |

The uniform control reproduces 12.11 / 8.99 / 5.86 exactly, so the fix touches
only the `NOISE='none'` path, as intended.

**Exact-state R2, the row rev 7 was missing.** Turning the noise off takes R2
from 12.11 to 8.09 raw — 4.0 pp of its raw intervention is noise — but only
from 5.86 to 5.02 engaged. So just 0.8 pp of R2's *engagement* is noise-driven;
the rest of the raw drop is the reference mismatch disappearing, because with
exact state `nn(x_pred)` and `nn(x̂)` at the landing tick coincide. Ours has no
exact-state arm to pair against (alg28 rejects `TTTCARV_NOISE=none`), so this
row is read against R2's own uniform arm, not against ours.

**Ours' proxy was near-exact; R2's was not.** Ours' per-step engaged count is
4.87% (uniform) and 6.68% (trunc) against the `u_diff_n_active` proxy's 4.77 /
6.62 — 0.1 pp out. R2's proxy was 1.7 pp out. That asymmetry is why the sign
appeared to favour R2.

**Paired, same 500 scenes, bootstrap 95% CI, R2 − ours:**

| metric | uniform | trunc |
|---|---|---|
| raw > 0.05 | +7.29 [6.81, 7.79], R2 higher in **96%** of scenes | +7.93 [7.39, 8.50], **95%** |
| **engaged (per step)** | **+0.99 [0.75, 1.22]**, 53% | **+1.09 [0.80, 1.39]**, 48% |
| **\|applied − clip(NN)\| > 0.05** | **+0.99 [0.63, 1.36]**, 54% | **+0.02 [−0.44, 0.47]**, 48% |

**A symmetric action-space metric that does not need `USAT`.** The last row
scores *both* sides as |applied − clip(NN, ±1)|: the baseline is not charged
for a clip it had to apply, and ours *is* charged for flying outside the input
set the baseline is held to. It asks one well-posed question of both — how far
is the flown command from the NN command the common input set permits — and it
needs no change to either controller, so it sidesteps the mode-J saturation
bug entirely. Ours reads 8.00 / 10.47 on it, against 4.82 / 6.61 raw.

It matches the action spaces in the *scoring* only. Ours still flies |ω| > 1
in these runs, so its trajectory — and therefore which conflicts arise and
when the filter engages at all — is not the one a saturated ours would have.
On a fixed trajectory, saturating ours could only lower its score here (its
pass-through steps would be charged 0 instead of |NN| − 1), so +0.99 pp is a
lower bound on the gap *for these trajectories*; that does not carry over to a
real saturated run. **This metric does not bracket the saturated number.**

**The per-scene consistency collapses.** On the raw metric R2 is higher in 95–96%
of scenes. On both corrected metrics it is 48–54% — a coin flip. The mean
difference is real and positive on the engaged metric, but it is a small
average effect, not something that shows up scene by scene. Any figure built
on "higher in 96% of scenes" is a figure about the clip and the reference
mismatch.

### Rev 8 addendum (2026-10-16, 17:45): ours with the NN saturated, measured (`~/runs/alg41_score/nx`)

> **RETRACTED at 19:50 by the session that wrote this section — every `USAT=1`
> figure below is invalid.** In mode J, `decoupled_step` returns
> `joint_step(t)` (`alg28_sound.py:801`) before the `_sat` at `:804`, and
> `joint_bounds` never saturates inside the CROWN graph, so the lookahead box
> is built from the unsaturated NN while the plant flies the saturated one.
> It is not an over-approximation, and it is filed as the next step's set:
> `est_look_truth_out` goes 0 → 217–256 per 500 trials,
> `est_refresh_uncontained` ~10×, `buffer_dropped_uncontained` ~3×, and the
> `KeyError: 49` below is the likely downstream effect. Ours with the NN
> saturated is **unmeasured** until the joint bounder saturates inside the
> graph. Still valid here: the baseline rescoring, and ours at `USAT=0` —
> against which pinned ref + clip gives R2 +1.45 [1.23, 1.68] uniform and
> +1.22 [0.99, 1.47] trunc, R1 +0.11 n.s.
>
> *(Banner added by the Rev 8/8b session so the file is not self-contradictory;
> the owning session may replace it with its own fuller box.)*


The USAT run above has landed. It is job 26098199, non-exclusive (mit_normal,
16 CPUs, 8 workers), with the sources from `alg41_score/src`: `alg28_sound`
b497fea7, PASSTHROUGH at its default 1, FAN_TERMINAL=0. It has six arms on one
node: USAT=1 vs USAT=0 for pipelined uniform, pipelined trunc and ideal uniform.
The table replaces the "≤" carry-overs above with measured values. Analysis:
`~/runs/alg41_score/a41_cmp.py`, output in `a41_cmp.out`.

- **Saturating ours' NN changes nothing.**
  - Pipelined uniform: 4.81 → 4.84 (+0.03 [−0.06, 0.12]).
  - Pipelined trunc: 6.63 → 6.57 (−0.05 [−0.20, 0.07]).
  - Ideal uniform: 4.70 → 4.67 (−0.03 [−0.14, 0.07]).
- **Safety is unchanged** under USAT=1: 0 collisions, 0 audit misses and 0
  uncertified fallbacks in every arm. Max wall per step is 0.152 / 0.159 /
  0.184 s.
  - Loiter steps: 8 (uniform) and 6 (trunc), against 0 in rev 7.
  - psf_no_diverge: 2 and 1.
  - The NN exceeds ±1 on about 3.9% (uniform) and 4.9% (trunc) of ours' steps.
    Its raw output reaches 2.9–3.1. The 1.75 quoted above is not the maximum.
- **One crash.** In ideal uniform with USAT=1, seed 1083342926 (trial 348)
  raised `KeyError: 49` in `joint_step` (`tester.horizons[t]`). It is excluded,
  so that pairing uses n=499. Not investigated.

Same action space (both clipped at ±1), paired against ours with USAT=1:

| | clip | pinned ref + clip | engaged |
|---|---|---|---|
| R2 − ours pipelined, uniform | +4.15 [3.90, 4.41] | **+1.43 [1.22, 1.63]**, 1.29× | −0.68 [−0.90, −0.45] |
| R2 − ours pipelined, trunc | +3.92 [3.64, 4.19] | **+1.27 [1.04, 1.53]**, 1.19× | −0.53 [−0.80, −0.24] |
| R1 − ours ideal, uniform | +0.16 [−0.06, 0.37] | +0.16 [−0.05, 0.38] | — |
| G3 − ours ideal, uniform | −0.15 [−0.35, 0.06] | −0.15 [−0.36, 0.06] | — |
| G0 − ours ideal, uniform | −0.50 [−0.68, −0.31] | −0.50 [−0.68, −0.31] | — |

**Correction to "What is defensible" above.** That paragraph treats the clip
column (R2 8.99 vs ours 4.74, about 1.9×) as defensible, on the grounds that
ours "can pass the live command through". In the pipelined deployment it
cannot. With `MEAS_DELAY=1`, ours' box at t is F(S_{t−1} ∩ z_{t−1}, u_{t−1})
(`alg28_sound.py:195`). It flies the NN at that box's centre and is scored
against the same value. R2's x̂_pred is the box from z_k pushed one tick, which
is the same information.

The clip column therefore scores R2 against an estimate that uses z_{k+1},
while ours is never scored against z_t. Ours is just as stale, and the column
charges only R2. The like-for-like intervention column is "pinned ref + clip":
+1.3 to +1.4 pp, about 1.2–1.3×. For ours to be credited with passing the live
command through, it would need its own score against the NN at the
fresh-measurement box. That is not logged.

## Rev 7 (2026-10-15): textbook stage-0 cost on every arm — CURRENT

*(Rev 8 shows most of this section's intervention gap is a scoring
artefact. The settings below are still current; the +7.0 pp reading is not.)*

Run: `~/runs/alg38_textbook`.
- Jobs: array 25937159 (the 15 rev 6 baseline arms) and exclusive job 25937158
  (ours, 4 arms, one node). Both mit_preemptable, all rc=0.
- Settings: rev 6b, plus `TTTCARV_GP_FIRSTSTEP=1` on the baselines and
  `TTTCARV_FIRST_STEP_COST=1` on ours. That is cost |u₀ − u_NN|² with tail
  weight 1e-3, on both sides (rev 6c/6d).
- Pinned sources: `alg29_generic_psf` a7c42a37…, `alg28_sound` 005fb2d4…
  (REANCHOR_REARM=0), `mpc_safety_filter_acados` 378fa856….
- Figure: `fig_intervention.png`, repointed to this run.

Intervention is the percentage of steps with |applied − NN| > 0.05; same 500
scenes and seed 20260824 for every arm.

| arm | coll. unif / trunc | > 0.05 unif | > 0.05 trunc | mean \|Δu\| unif |
|---|---|---|---|---|
| **ours pipelined** | **0 / 0** | **4.88** | **6.62** | **0.059** |
| ours ideal | 0 / 0 | 4.76 | 6.52 | 0.058 |
| R2 W&Z tube + comp. | 0 / 0 | 11.90 | 14.84 | 0.104 |
| R1 W&Z tube, δ=0 (not deployable) | 0 / 0 | 8.24 | 11.13 | 0.071 |
| G3 box-tightened, δ=0 (not deployable) | 0 / — | 7.75 | — | 0.068 |
| G0 textbook, point | 63 / 88 | 7.25 | 7.30 | 0.062 |
| G2 textbook + comp., point | 48 / 92 | 11.17 | 11.19 | 0.098 |
| G0, exact state (sanity) | 0 | 7.14 | | 0.060 |
| G0 unpinned single solve | 1 | 72.69 | | 0.282 |
| R2 unpinned single solve | 0 | 71.84 | | 0.309 |

Paired at > 0.05 against ours pipelined, same law:

| baseline | uniform | trunc |
|---|---|---|
| R2 − ours | +7.0 pp [6.6, 7.5], higher in 95% of scenes (≈2.4×) | +8.2 pp [7.6, 8.8], 94% |
| R1 − ours | +3.4 pp [2.9, 3.8], 71% | +4.5 pp [3.9, 5.1], 72% |
| G3 − ours | +2.9 pp [2.5, 3.3], 66% | — |

What changed from rev 6:
- **Sound baselines:** still 0 collisions and 0 invalid certificates. They
  intervene 0.4–0.7 pp less than in rev 6.
- **Ours:** unchanged. Audits 0, uncertified fallbacks 0, max wall per step
  0.161 s against the 0.20 s budget.
- **Point PSFs** (G0, G2) collide slightly more under uniform noise (+10 each),
  because the old cost no longer adds clearance.
- **The baselines never loiter;** failed solves fly the cached tail.
- **G0 unpinned:** 1 collision (was 0). The trace has not been examined.

**Correction to rev 6's framing.** Rev 6 said the baselines' ~75% was our
cost design. Rev 6c refutes this: with the textbook cost a single unpinned
solve still intervenes on ~72% of steps. The likely cause is the soft
path-constraint slack penalties in the OCP, or local solutions; not verified.
The pin is what makes the baselines reach the PSF optimum. Say that, not
"cost design".

**Not rerun yet with rev 7 settings:** the uncompensated-latency arms (G1, and
R1 at δ > 0, measured δ). Their numbers are still rev 4's.

### Rev 7b (2026-10-15): path-only fan margin, and pinning the engaged re-solve (`~/runs/alg39_pinfan`)

Two tests on top of rev 7. Each arm is paired with a control rerun on the same
node in the same job, over 500 scenes. Intervention is the percentage of steps
above 0.05. Collisions were 0 in every arm.

**1. Path-only fan margin** (`TTTCARV_FAN_TERMINAL=0` for ours,
`TTTCARV_GP_FAN_TERMINAL=0` for R1/R2). The OCP's obstacle margin grows as
k·dt·v·hw_θ along the horizon. Rev 7 applied that growth at k = N too, where the
terminal constraint is hard. Path-only applies it on path stages only; the
terminal keeps the start inflation. Acceptance boxes are unchanged, so
soundness is unaffected.

| arm | control | path-only | Δ [95% CI] | bad one-shot solves |
|---|---|---|---|---|
| ours pipe, uniform | 4.84 | 4.74 | −0.10 [−0.19, −0.01] | 32% → 18% |
| ours pipe, trunc | 6.60 | 6.55 | −0.05 [−0.14, +0.04] | 38% → 19% |
| R1, uniform | 8.23 | 8.16 | −0.07 [−0.27, +0.14] | — |
| R1, trunc | 11.03 | 11.04 | +0.00 [−0.29, +0.30] | — |
| R2, uniform | 11.99 | 12.11 | +0.12 [−0.10, +0.35] | — |
| R2, trunc | 14.92 | 14.54 | −0.38 [−0.72, −0.08] | — |

What this shows:
- The fix halves ours' failed solves, as the 2026-09-25 bench predicted.
- It barely moves intervention, because few of the failed solves were the
  ones deciding an engagement.
- R1/R2 don't change: the pin succeeds on ~96% of steps, so the unpinned
  solve, where the terminal margin matters, rarely runs.
- Max wall per step for ours is 0.169 s (path-only) vs 0.163 s (control).

**2. Pinning the engaged re-solve** (`TTTCARV_ACTIVE_PIN=1`, ours only). In
MPC_ACTIVE, before the unpinned re-solve, ours tries one re-solve with ω₀ pinned
to u_NN. The pin was accepted 3/444 times (uniform) and 9/543 (trunc).
Intervention was unchanged: +0.00 pp [−0.06, +0.06] in both noise models. The
extra solve uses budget: failed re-solves rose 34 → 58 (uniform) and 15 → 25
(trunc).

So when ours is engaged, u_NN is almost never certifiable. The revert test,
which builds a backup at t+1 under the NN, already returns control in the cases
where it is. **Ours' engaged-step interventions are not an artifact of skipping
the pin.** Leave this off.

**Where the code is.** The rev 7b arms ran from
`~/runs/alg39_pinfan/src/alg28_sound.py`, a copy of hash 2365104b, through
`CARV_SRC_OVERRIDE` in `~/scripts/carv_alg29_arms.sh`.

**Now the default (2026-10-15):** path-only on both sides. The checkout
alg28_sound.py has `TTTCARV_FAN_TERMINAL` (default 0) and is now hash 486c4ca2
(was 64c8b9c8). alg29 has `TTTCARV_GP_FAN_TERMINAL` (default 0). Set either one
to 1 to reproduce rev 7. `ACTIVE_PIN` was not ported; it exists only in the
alg39 copy.

**Ours vs baselines, both path-only (same runs):**

| comparison | gap [95% CI] | ratio | collisions |
|---|---|---|---|
| R2 − ours, uniform | +7.37 pp [6.91, 7.86] | 2.56× | 0 / 0 |
| R2 − ours, trunc | +8.00 pp [7.46, 8.58] | 2.22× | 0 / 0 |
| R1 − ours, uniform | +3.42 pp [3.02, 3.85] | 1.72× | 0 / 0 |
| R1 − ours, trunc | +4.49 pp [3.95, 5.06] | 1.69× | 0 / 0 |

These are consistent with rev 7 (+7.0 / +8.2 for R2). G0, G2 and G3 never
install the stage inflation, so their rev 7 numbers stand. Not yet rerun
path-only: ours ideal, and R2 unpinned.

## Rev 6 (2026-10-14): intervention at equal safety (superseded by rev 7)

Run: `~/runs/alg34_pin`, array job 25856425 (15 fixed-δ arms, mit_preemptable,
all COMPLETED rc=0). Pinned: `alg29_generic_psf` df290eed…, `alg28_sound`
456d44ce…, `mpc_safety_filter_acados` 378fa856…. Same 500 scenes, seed
20260824, rev 4/5 settings (`TTTCARV_GP_SHRINK=1`). Ours is not rerun: its
numbers are the rev 4 (uniform) and rev 5 (trunc) fingerprints.
Figure: `fig_intervention.png` (script `fig_intervention.py`).

**Why.** Every baseline changed the NN's command (|applied − NN| > 0.05) on
~75% of steps, ours on ~5%. That gap was suspect: our OCP prices deviation
over the whole horizon with soft path-D penalties, which is not W&Z's (6a)
(first input only). A reviewer would call it a strawman.

**The probe (`TTTCARV_GP_PIN=1`, off by default).** Before the normal solve,
pin the OCP's stage-0 input to the NN's command at the solve state
(lbu = ubu = clip(NN)), solve, and run the unchanged acceptance test. If it
passes, that plan is used and the NN command goes out unchanged; otherwise the
normal solve runs. This is what (6a) with a converged solver returns whenever
the NN input admits a certified continuation, so it is each baseline's
best-case intervention rate, independent of our cost design.

Sanity: G0 with exact state + pin has 0 collisions and 0 invalid certificates
(the probe is sound); the unpinned controls reproduce rev 4 exactly (G0 75.33%,
55 failed solves).

**Result 1 — the 75% was our implementation, not the method.**

| arm | interv. unpinned | interv. pinned | mean \|Δu\| pinned | collisions pinned |
|---|---|---|---|---|
| G0 δ=0, uniform | 75.3% | 7.8% | 0.068 | **52** |
| R2 δ=0.1, uniform | 74.7% | 12.6% | 0.112 | 0 |

**Result 2 — point-anchored PSFs that intervene minimally are unsafe under
noise.** Their rev 3–5 near-zero collisions came from the implicit margin of an
over-intervening cost, not from the certificate.

| arm (pinned) | uniform: interv. / coll. | trunc: interv. / coll. | exact state: coll. |
|---|---|---|---|
| G0 textbook, δ=0 | 7.8% / **52** | 7.9% / **93** | 0 |
| G2 compensated, δ=0.1 | 11.3% / **38** | 11.3% / **82** | 0 |
| G3 box-tightened, δ=0 | 8.2% / 0 | — | — |
| R1 W&Z tube, δ=0 | 8.9% / 0 | 11.7% / 0 | — |
| R2 W&Z tube + comp., δ=0.1 | 12.6% / 0 | 15.4% / 0 | — |
| **ours, pipelined** | **4.9% / 0** | **7.4% / 0** | — |
| ours, ideal | 4.8% / 0 | 7.3% / 0 | — |

(G2/R2 at δ=0.2 are identical to δ=0.1: the control period is 0.2 s and
ALIGN rounds both up to one tick.)

**Result 3 — every zero-collision baseline intervenes more than ours.** Paired
over the same 500 scenes, bootstrap 95% CI of (baseline − ours, pipelined):

| vs | noise | Δ interv. rate | Δ mean \|Δu\| | baseline higher in |
|---|---|---|---|---|
| R2 (deployable) | uniform | +7.7 pp [7.3, 8.2] | +0.053 [0.048, 0.057] | 96% of scenes |
| R2 | trunc | +8.0 pp [7.5, 8.5] | +0.058 [0.053, 0.063] | 93% |
| R1 (δ=0, not deployable) | uniform | +4.0 pp [3.7, 4.4] | +0.020 [0.016, 0.025] | 82% |
| R1 | trunc | +4.4 pp [3.9, 4.9] | +0.025 [0.020, 0.031] | 78% |
| G3 (δ=0, not deployable) | uniform | +3.3 pp [2.9, 3.7] | +0.014 [0.009, 0.018] | 75% |

**How to read it, and its limits.**
- Against the deployable robust baseline (R2) ours changes the NN about
  2.6× less often (4.9% vs 12.6%) at the same zero collisions; against the
  δ=0 idealisations R1/G3 the gap is ~1.7–1.8×. That is the measured
  advantage — not a 15× one, and not a safety one.
- The tables above compare ours at > 0 with the baselines at > 0.05. That
  mismatch is removed by the matched-threshold rerun below; the conclusion
  holds.
- Part of R2's excess is structural, not tuning: a pinned plan certifies the
  input nn(x̂_pred), while the command that lands is compared with nn(x̂) at the
  landing tick; under noise these differ. Ours certifies the NN *policy* over
  the set, so it can apply nn(x̂) itself. This is a real property of the method,
  but say it is.
- Intervention is a proxy. Task progress (goal reached, path length) is not
  measured yet.
- With pinning, R1/R2 loiter more (27–42 loiter steps vs 8 unpinned); ours
  never loitered.

### Rev 6b: matched intervention threshold (`~/runs/alg35_thr`)

Both modules now record `u_diff_gt` = count of steps with |applied − NN| > t
for t ∈ {0, 0.01, 0.05, 0.1}, from the same per-step list (bookkeeping only).
Rerun: array 25859486 (15 baseline arms) and exclusive job 25859485 (ours, 4
arms), mit_preemptable, all rc=0. Pinned: `alg29_generic_psf` 9466e26a…,
`alg28_sound` f332d364…, `mpc_safety_filter_acados` 378fa856…. Collisions and
the baselines' > 0.05 rates reproduce rev 6 exactly (deterministic arms).

| arm | coll. | > 0 | > 0.01 | > 0.05 | > 0.1 | mean \|Δu\| |
|---|---|---|---|---|---|---|
| **ours pipelined, uniform** | 0 | 4.90% | 4.87% | 4.81% | 4.79% | 0.060 |
| ours ideal, uniform | 0 | 4.84 | 4.80 | 4.74 | 4.72 | 0.059 |
| G0 pin, uniform | 52 | 96.95 | 8.09 | 7.75 | 7.27 | 0.068 |
| G2 pin δ=0.1, uniform | 38 | 100 | 17.78 | 11.31 | 9.35 | 0.100 |
| G3 pin δ=0, uniform | 0 | 96.95 | 8.56 | 8.20 | 7.66 | 0.073 |
| R1 pin δ=0, uniform | 0 | 97.00 | 9.58 | 8.93 | 8.38 | 0.080 |
| R2 pin δ=0.1, uniform | 0 | 100 | 19.16 | 12.62 | 10.59 | 0.112 |
| **ours pipelined, trunc** | 0 | 6.67 | 6.62 | 6.56 | 6.53 | 0.083 |
| R1 pin δ=0, trunc | 0 | 96.98 | 12.62 | 11.73 | 10.94 | 0.107 |
| R2 pin δ=0.1, trunc | 0 | 100 | 27.64 | 15.35 | 12.79 | 0.140 |
| G0 pin, exact state | 0 | 97.20 | 8.16 | 7.81 | 7.34 | 0.069 |

- **Do not use the > 0 column for baselines.** Even G0 with exact state and a
  pinned first input reports ~97%: acados returns the pinned input to solver
  tolerance, not bit-exact, so > 0 counts float noise. Ours passes the NN
  through bit-exact, so > 0 would flatter ours by ~90 pp — an artefact.
  Report > 0.01 … > 0.1 and mean |Δu|.
- Ours is flat in the threshold (4.90 → 4.79%); its interventions are large
  when they happen. The baselines fall from 0.01 to 0.1, most for G2/R2 — the
  predicted-vs-landing NN mismatch (they certify nn(x̂_pred), compared against
  nn(x̂) at landing).
- Paired, ours pipelined vs R2 at > 0.05: +7.8 pp [7.4, 8.2], R2 higher in 97%
  of scenes (uniform); +8.8 pp [8.3, 9.3], 95% (trunc). Vs R1: +4.1 pp
  [3.7, 4.5] (uniform), +5.2 pp [4.7, 5.7] (trunc). Vs G3: +3.4 pp [3.0, 3.8].
- **Ours is not deterministic run to run** (wall-clock priced): trunc
  pipelined was 7.37% in rev 5 and 6.67% here; uniform 4.90% both times.
  Quote ours with that spread. The baseline arms are deterministic on a
  given node, but not bit-identical across nodes (see rev 6c control).

### Rev 6c: textbook stage-0 cost for the baselines (`~/runs/alg36_firststep`)

New knob `TTTCARV_GP_FIRSTSTEP=1` gives cost |u₀ − u_NN|², with weight
`TTTCARV_GP_TAILW` = 1e-3 on stages 1..N−1. The tail weight is not 0 because
Gauss–Newton without LM makes zero-weight blocks singular. This is the
`first_step_cost` option dropped on 2026-08-31; it was measured then on the
re-simulation path, never on the one-shot path the baselines now use.

Run details:
- Array 25888867, 6 arms × 500 scenes, seed 20260824, mit_preemptable, all rc=0.
- Pinned sources: `alg29_generic_psf` a7c42a37… and `mpc_safety_filter_acados`
  378fa856….
- Baselines only; ours is unchanged, and its rev 6b numbers are used.

| arm (uniform unless noted) | coll. | > 0.05 old → new | mean \|Δu\| | failed solves | shrink recovered | cached-tail / loiter steps |
|---|---|---|---|---|---|---|
| R2 pin, control (fs=0) | 0 → 0 | 12.62 → 12.58 | 0.112 → 0.112 | 108 → 103 | 325 → 333 | 81/27 → 76/27 |
| R2 pin | 0 → 0 | 12.62 → **11.90** | 0.112 → 0.104 | 108 → 107 | 325/406 → 89/196 | 81/27 → 107/0 |
| R2 pin, trunc | 0 → 0 | 15.35 → **14.78** | 0.140 → 0.139 | 145 → 215 | 811/929 → 104/319 | 118/27 → 213/0 |
| R1 pin | 0 → 0 | 8.93 → **8.24** | 0.080 → 0.071 | 108 → 116 | 318/391 → 76/192 | 73/35 → 116/0 |
| G0 pin (point) | 52 → 63 | 7.75 → 7.25 | 0.068 → 0.062 | 63 → 118 | 55/118 → 34/152 | 63/0 → 118/0 |
| R2 **unpinned** | 0 → 0 | 74.73 → **71.93** | 0.278 → 0.310 | 99 → 79 | 445 → 85 | 91/8 → 76/3 |

- **The textbook cost lowers the baselines' intervention slightly, by
  0.5–0.7 pp.** Paired R2: −0.72 pp [−1.16, −0.25]. The verdict against ours
  holds. Paired at > 0.05 vs ours pipelined:
  - R2: +7.1 pp [6.6, 7.6] (uniform), higher in 94% of scenes;
    +8.2 pp [7.6, 8.9] (trunc), 95%.
  - R1: +3.4 pp [3.0, 3.9].
  - So R2 is about 2.5× ours, previously 2.6×.
- **No instability from the cost itself.** Solver failures are unchanged for
  R2/R1 uniform, and up for trunc (+48%) and G0 (+87%). The shrinking re-solve
  recovers far less (its padding steers through the cost), so more steps fly
  the cached tail. Loiter drops to 0, and the sound arms stay at 0 collisions
  and 0 invalid certificates. This is a feasibility cost, not a safety one,
  which is the direction the 2026-08-31 note found but much smaller.
- **A single solve does not return u_NN even with the textbook cost.** Unpinned
  R2 with fs=1 intervenes on 71.9% of steps against 11.9% pinned, and on 59.5%
  above 0.1, so these are not tolerance effects. The OCP still prices the soft
  path constraints: D slack 1e1 and raw-r slack 1e3. The likely cause is that
  this pulls u₀ away from u_NN even when the pinned plan passes the hard
  acceptance test; not verified. Non-convexity (local solutions) is the other
  candidate. So the pin is required for the baselines to reach W&Z's optimum;
  it is not just a shortcut.
- **Determinism is per node.** The control matches rev 6b in 485 / 500 scenes,
  −0.04 pp [−0.10, +0.02]. It ran on node1383; rev 6b ran on node1705 or 2506.
  This is likely CPU-dependent floating point, and it is negligible next to
  the effects above.
- Wall times are not comparable here: each arm ran on a different node.

### Rev 6d: textbook stage-0 cost for ours too (`~/runs/alg37_ours_fs`)

Ours already had the knob: `TTTCARV_FIRST_STEP_COST=1`, with
`TTTCARV_TAIL_COST_WEIGHT=1e-3`. Once ours engages it re-solves every step and
applies u₀ (RECEDING=1), which is the case the stage-0 cost is meant for.

Run details:
- Job 25934264, one exclusive preemptable node (node2805), 8 workers, seed
  20260824.
- Pipelined only. Each arm has a same-node control, because ours is
  wall-clock priced.
- All arms rc=0, 500/500 safe.
- `alg28_sound` 005fb2d4… This is the version with the default-off
  REANCHOR_REARM knob from another session; REARM=0 here.

| arm | coll. | > 0.05 | mean \|Δu\| | re-solves ok / failed / no budget | deadline gates | max wall/step |
|---|---|---|---|---|---|---|
| ours pipe, uniform, control | 0 | 4.89 | 0.060 | 397 / 33 / 618 | 22 | 0.150 s |
| ours pipe, uniform, **stage-0** | 0 | 4.87 | 0.059 | 422 / 31 / 600 | 34 | 0.141 s |
| ours pipe, trunc, control | 0 | 6.66 | 0.085 | 494 / 16 / 781 | 30 | 0.158 s |
| ours pipe, trunc, **stage-0** | 0 | 6.67 | 0.083 | 510 / 21 / 760 | 43 | 0.151 s |

- **No effect on ours.** Paired stage-0 − control at > 0.05: −0.02 pp
  [−0.15, +0.10] (uniform) and +0.01 pp [−0.12, +0.15] (trunc).
- Sound audits are 0 and uncertified fallbacks are 0. Every step is within the
  0.20 s budget. Deadline gates rise a little; these are solves the budget
  stopped cleanly, not overruns. As expected, when ours engages is decided by
  the policy certificate, not by the cost.
- The controls match rev 6b within noise: +0.08 pp [−0.08, +0.24] and
  +0.09 pp [−0.09, +0.27]. The earlier trunc 7.37% (rev 5) looks like the
  outlier.
- Paired vs R2 with the stage-0 cost (rev 6c): R2 − ours = +7.0 pp [6.6, 7.5]
  (uniform), higher in 95% of scenes; +8.1 pp [7.5, 8.8] (trunc), 94%. Mean
  |Δu| is 0.059 vs 0.104 (uniform) and 0.083 vs 0.139 (trunc).
- Decision: the stage-0 cost is used on both sides from rev 7 on
  (`~/runs/alg38_textbook`). This is the literature's cost, and it costs
  neither side safety.

## Rev 5 (2026-10-13): noise laws — bounded Gaussian and unbounded (superseded by rev 6)

Run: `~/runs/alg33_trunc`, jobs 25777362 (array, 22 fixed-δ arms) and 25777363
(exclusive node, 12 wall-clock-priced arms). Pinned: `alg29_generic_psf`
1d9160e0…, `alg28_sound` 456d44ce…, `mpc_safety_filter_acados` 378fa856….
Every arm carries rev 4's `TTTCARV_GP_SHRINK=1`. Same scenes, seed 20260824.

alg28 already had `TTTCARV_NOISE=trunc`; alg29 rejected anything but
`none|uniform`, so no baseline could be run under it. alg29 now draws from the
same sampler (one `rng.uniform` of identical shape, so trunc/uniform arms are
seed-aligned and see identical scenes):

| law | draw | bound |
|---|---|---|
| `uniform` | U[−vmax, vmax] | respected |
| `trunc` | N(0, (vmax/r)²) conditioned on \|v\| ≤ vmax, inverse CDF, r = 3 | respected |
| `gauss` | N(0, (vmax/r)²), unbounded | **violated** |

Sampler checks: max\|v\| = 0.149992 against the 0.15 bound, σ = 0.0493, RNG in
the identical state after either draw. `trunc` keeps the certificates' bound
but makes the typical error half as large (mean \|v\| 0.040 vs 0.075; 4.3% of
draws beyond 2σ vs 33%), so consecutive measurement boxes nearly coincide and
set-membership intersections cut much less — the pessimistic case, as
`results_alg28/error_box_lemma.md:134` already recorded for interventions.

**The peaked law doubles the rate at which a point-anchored certificate goes
stale, and leaves box-anchored ones at zero.** Share of steps on which the
filter's active plan, re-simulated from the TRUE state, no longer passes the
test it was accepted under:

| arm | certifies on | uniform | trunc | trials affected (unif → trunc) |
|---|---|---|---|---|
| G0 textbook, δ=0 | point | 0.160% | **0.353%** | 38 → 70 of 500 |
| G2 δ=0.1, compensated | point | 0.166% | **0.353%** | 41 → 74 of 500 |
| G3 δ=0, box-tightened | box | 0.000% | 0.000% | 0 → 0 |
| R1 tube, δ=0 | tube | 0.000% | 0.000% | 0 → 0 |
| ours (ideal and pipelined) | certified set | 0 audits | 0 audits | 0 → 0 |

**Collisions under `trunc` (bound respected), 500 paired scenes:**

| arm | collisions | note |
|---|---|---|
| ours ideal / pipelined | **0 / 0** | every audit 0, truth never left the set |
| G0 textbook, δ=0 | **1** | seed 1707289359, step 36: no failed solve, no fallback, no loiter, truth in box — 13 steps with the active plan invalid from the true state, 12 of them `path_raw`. The certificate was anchored at x̂ and the robot was elsewhere inside the box. |
| G2 / G3 / R1 δ=0 / R2 all δ | 0 | |
| G1 δ=0.1 / δ=0.2 / measured | 9 / 4 / 8 | latency, law-insensitive (uniform: 10 / 2 / 2) |
| R1 δ=0.1 / δ=0.15 / measured | 7 / 128 / 4 | tube broken on 32% / 44% / 13% of steps |

**Collisions under `gauss` (bound violated).** The assumption every certificate
here rests on is false, and it shows — stated rather than hidden:

| arm | collisions | certified set lost the truth |
|---|---|---|
| ours ideal | 1 | 2 588 / 30 000 steps (8.6%) |
| ours pipelined | 3 | 2 441 / 30 000 steps |
| G0 / G2 δ=0.1 | 0 / 1 | box unsound on ~1.2% of steps, but G0/G2 never use the box |
| R1 δ=0 / R2 | 0 / 0 | tube held anyway (tube ≫ the Gaussian tail) |

In the colliding trials ours' set lost the truth on 25–43 of 60 steps.
*(Corrected 2026-10-14.)* Every set-based method here assumes the bound — G3,
R1, R2 and ours alike; robust MPC always does. R1/R2 survived the Gaussian tail
because their open-loop tube is very wide (0.385 m at node 12), i.e. by
conservatism, not by robustness to unbounded noise. Ours' set-membership set
intersects measurement boxes over time, so it is tighter, and one out-of-bound
measurement can cut the truth out of it (8.6% of steps vs ~1.2% for the
baselines' box). G0 makes no robustness claim, so a broken bound breaks
nothing it promised. *(Rev 6: with minimal intervention, G0 collides 52/500
under uniform and 93/500 under trunc, so the rev 5 G0 0-vs-1 comparison below
is superseded.)*

**What is and is not established.** The *rate* statistics are solid (30 000
steps/arm): the peaked law doubles point-certificate staleness, box and tube
certificates stay at exactly 0, and ours' audits stay at 0. The single G0
collision is one trial — 1/500 vs 0/500 does not separate the arms at this n
(95% CIs overlap); what it demonstrates is the *mechanism*, in an arm whose
solver never failed once. Rev 1's "point certificates are void under noise"
claim is restored in that form, after rev 3 had weakened it.

## Rev 4 (2026-09-25): shrinking-horizon re-solve (Wabersich & Zeilinger Algorithm 2, line 6)

Run: `~/runs/alg32_shrink`, jobs 23982473 (array) and 23982474 (exclusive).
Pinned: `alg29_generic_psf` 09f24737… + the shrink patch, `alg28_sound`
456d44ce…, `mpc_safety_filter_acados` 378fa856…. DI arms unaffected (the knob
is unicycle-only); ours unchanged and rerun only for node parity.

**The fidelity gap.** Algorithm 2 does not replay the cached plan when the
full-horizon problem is infeasible: it re-solves at horizon N − (k − k̄), i.e.
the steps left before the deadline the last good plan promised. The cached tail
is only the feasible *point* their recursive-feasibility argument exhibits
(eq. 11, via the ancillary policy π). Flying it is the weakest reading of
their fallback, and it was what alg29 did.

**Implementation.** acados fixes the horizon at build time and bakes the
obstacles in, so a solver per horizon would cost 12 × 1.4 s per *trial*.
Instead the shorter problem is posed on the existing solver: the path-D slack
is priced at 1e7 from stage m on, and the acceptance test certifies nodes
1..m only. Since the terminal set is control-invariant (the max-turn loiter is
the witness), "land in S_t at m" and "land at m and stay" have the same
feasible set, so the padding only steers the solver — the certificate is the
paper's. Off by default (`TTTCARV_GP_SHRINK=1`), so rev 3 reproduces.

**Effect: it recovers most failures and changes no verdict.**

| arm | failed solves (rev 3 → 4) | recovered | collisions (3 → 4) |
|---|---|---|---|
| G0 textbook, δ=0 | 180 → 55 | 124/179 | 0 → 0 |
| G2 δ=0.1 | 181 → 60 | 122/182 | 0 → 0 |
| R1 tube, δ=0 | 543 → 99 | 441/524 | 0 → 0 |
| R1 δ=0.1 | 669 → 165 | 473/618 | 10 → 7 |
| R1 δ=0.15 | 1500 → 1018 | 478/1469 | 129 → 128 |
| R2 compensated | 537 → 100 | 446/538 | 0 → 0 |
| G1 δ=0.15 / δ=0.3 | 908 → 748 / 518 → 463 | 163 / 58 | 128 → 128 / 148 → 148 |

The re-solve fixes `term_D` rejections ("cannot reach the safe set in time":
40 → 6 in a 40-trial probe) and not `path_raw` ones ("the next node already
hits an obstacle": 90 → 88).

**It costs compute exactly when the filter is in trouble.** Up to 3 solve
passes per timestep (one-shot, then 4 multi-start warm starts, then the same
again shrunk = up to 10 acados solves). On measured-δ arms:

| arm | median solve | p99 | max | deadline misses |
|---|---|---|---|---|
| G1 measured, rev 3 → 4 | 23.7 → 31.2 ms | 29.6 → 37.9 | 97 → 245 ms | 0 → 0 |
| R1 measured, rev 3 → 4 | 23.8 → 30.1 ms | 35.4 → 55.9 | 89 → 222 ms | 0 → 0 |
| G2 measured, rev 3 → 4 | 23.7 → 30.1 ms | 29.8 → 37.8 | 139 → 231 ms | 0 → **16** |
| R2 measured, rev 3 → 4 | 23.8 → 30.2 ms | 34.5 → 55.1 | 98 → 224 ms | 0 → **14** |

+30% median; the worst case now exceeds the 0.20 s assumed budget, so G2/R2
begin missing deadlines where they never did, and measured-δ collisions rose
slightly (G1 1 → 2, R1 1 → 4). Being faithful to Algorithm 2 improves the
baseline's recovery on paper and worsens its timing in wall clock.

**Not applied to ours, deliberately.** Ours' cached backup is certified for
every state in the set S, and the truth is inside S (`truth_out = 0` in every
run), so the certificate already covers where the robot is — there is nothing
to re-anchor. The baselines' tail was certified for one predicted point, which
the truth has left; that is the hole the re-solve patches. Ours also had
nothing to recover (0 audit failures, loiter never used) and is priced against
a 0.20 s budget that a third solve pass would break.

## Fidelity audit against Wabersich & Zeilinger (2021), Automatica

Checked R1/R2 against the paper's equations (text extracted from
`waberish.pdf`, 17 pp.).

| paper | ours | verdict |
|---|---|---|
| (6a) cost on the **first** input only | ‖ω_i − ω_NN,i‖² over the horizon | deviation; shared by every arm (`TTTCARV_FIRST_STEP_COST=0`), affects plan choice, not soundness |
| (6c) tightened state constraint | per-stage obstacle inflation by e_k, pushed into acados `p` | faithful |
| (6d) tightened input set | not tightened | **vacuous**: (9b) exists because π(x,µ,v) ≠ v; with no ancillary law u = v exactly, so the required tightening is 0 |
| (6e) model-confidence map | dropped | N/A: exact model, so their Assumption 4.1 holds globally (Z_c = X × U) |
| (6f) tightened terminal set | terminal check runs on the tube box at N | faithful |
| (6g) µ₀ = x(k), **true state** | centre of the set-membership box, box folded into e_0 | extension; the paper has no observer |
| Alg. 2 line 6, shrinking re-solve | implemented in rev 4 | faithful (surrogate posing, same certificate) |
| Alg. 2 line 10, terminal filter | loiter | faithful |
| p_S chance level | worst-case bounded noise, p_S = 1 | the paper calls this "a robust version" (§4.3) |

**The tightening law.** The paper's ε_i = ε(1−√ρⁱ)/(1−√ρ) *converges*, because
Assumption 4.3 posits an ancillary controller contracting the error at rate
ρ < 1. That assumption fails here: at constant speed with only ω, along-track
error is uncontrollable (two unicycles on one path at one speed keep their gap
whatever they steer), so ρ = 1 is forced — and their recursion at ρ = 1 gives
ε_i = i·ε, which is exactly our open-loop Lipschitz tube
e_k = e_p0 + k·v·dt·e_θ. The open loop is their law evaluated at the contraction
rate this plant has, not a cheaper substitute. By node 12 the tube is 0.385 m
against obstacles of radius 0.5 m, which is what costs R1/R2 ~3× G0's failed
solves. Adding a tracker would thin the tube cross-track only (the binding
direction is the one that cannot shrink), would cost ω authority via (9b), and
would still not repair the latency hole: invariance assumes you *start* inside
the set, and a command landing a tick late does not.

**The paper never mentions computation delay.** x(k) goes in and u(k) comes out
at the same index; the only timing remark is that the quadrotor OCP "was solved
in real-time using Ipopt". *(Corrected 2026-10-14.)* This is the standard
sampled-data idealisation, not a flaw specific to W&Z: computation delay in MPC
has been handled since at least Findeisen & Allgöwer (ADCHEM 2004, prediction
over the solve time with stability conditions), advanced-step NMPC and
real-time iteration, and for CBF safety filters by predictor feedback (Molnar,
Kiss, Ames & Orosz, TCST 2023). Delay-aware robust tube MPC used as a safety
guard also exists (arXiv 2109.07101). G1/R1 therefore demonstrate a known
failure; they motivate, they are not a result. Compensation (G2/R2) fixes the
collisions, and R2 matches ours on every safety metric measured. Our measured
advantage is intervention (rev 6), not safety.

## Rev 3 (2026-09-22): loiter-invariant terminal set (superseded by rev 5)

Run: `~/runs/alg31_term`, jobs 23481544 (array) and 23481545 (exclusive node).
Pinned sources: `alg29_generic_psf` 09f24737…, `alg28_sound` 456d44ce…,
`mpc_safety_filter_acados` 378fa856…. Same scenes and seed as before; rev 2
settings (|ω| bound enforced, ALIGN) plus `TTTCARV_TERMINAL=loiter` on every
unicycle arm, ours included. Rev 2's figures and CSV are archived in `rev2/`.

**The fix.**
- D = √(r² + 2rR) is invariant for one obstacle only, and only if the loiter
  turns away from it. Rev 2's only collision in the sound arms was a loiter
  blocked on both sides.
- The new terminal test: the max-turn loiter on the Euler plant keeps every
  tick on one circle of radius ρ = v·dt / (2 sin(ω·dt/2)) = 1.0017, and every
  chord between ticks stays at least h = ρ cos(ω·dt/2) from its centre. A
  terminal box is accepted iff some side s has every obstacle wholly outside
  that annulus, for every state in the box. The centre's uncertainty is bounded
  by the xy half-diagonal plus 2ρ sin(hw_θ/2).
- This set is control-invariant under the loiter for any number of obstacles.
  The loiter commits to the certified side. Ours' fixed +1 loiter now uses the
  certified side too.
- Validation: the geometry is exact to 3e-14. In a Monte Carlo of 20 000
  random boxes and scenes, 15 621 were accepted; 78 105 loiters sampled from
  inside them over 70 steps each hit 0 obstacles.
- The test is also *less* conservative than D in open space: near an obstacle
  but heading away is accepted. G0's failed solves dropped from 3 132 to 180.

| arm | collisions | invalid-backup steps (trials) | tube broken |
|---|---|---|---|
| **ours ideal / pipelined** | **0 / 0** | every audit 0 | — |
| G0, no noise / noise | 0 / 0 | 0 / 42 (32) | — |
| G1 nn δ = 0.05 / 0.1 / 0.15 / 0.2 / 0.3 / 0.4 | 3 / 10 / 128 / 6 / 148 / 37 | 96 / 325 / 1323 / 648 / 2710 / 1615 | — |
| G1 nn δ = measured, no noise / noise | 0 / 1 | 77 (48) / 118 (68); 20–21% of fallbacks | — |
| G1 prev δ = 0.1 / 0.2 | 3 / 6 | 115 / 648 | — |
| G2 (aligned), any δ, no noise | 0 | 0 | — |
| G2 (aligned) δ = 0.1 / 0.2 / measured, noise | 0 / 0 / 0 | 57 / 57 / 53 (~42 trials) | — |
| G3 δ = 0 / 0.2 | 0 / 0 | 0 / 1 | — |
| R1 δ = 0 | 0 | 0 | 0 |
| R1 δ = 0.05 / 0.1 / 0.15 / 0.2 / 0.3 | 3 / 10 / 129 / 3 / 144 | 32 / 206 / 1196 / 302 / 2344 | 48 / 57 / 64 / 100 / 100% of steps, all 500 trials |
| R1 δ = measured | 1 | 36 (14) | 35% of steps, 486 / 500 trials |
| R2 (aligned) δ = 0.1 / 0.2 / 0.4 / measured | 0 | 0 | 0 |
| DI (terminal set unchanged) | identical to rev 1/2 | | |

- An uncertified loiter (no side certified from the estimate) happened only
  in naive-latency arms (G1 δ ≥ 0.15, R1 δ = 0.15 / 0.3), in 1–17 steps per
  arm. Every certified loiter step was also certified from the TRUE state.
- Ours: 992 / 1 048 activations (ideal / pipelined). It never needed the
  loiter. Median wall per timestep was 0.090 s.

**What rev 3 changes in the story.**
1. **Every sound method now has 0 collisions**: ours, G0, G2, G3, R1 at δ = 0,
   and R2. The rev 1–2 collisions in the sound arms were the terminal set.
2. **Noise alone barely matters any more.** With a correct terminal set, point
   certificates under noise fail on only ~0.15–0.2% of steps (32–43 trials of
   500), down from ~1.5% (~155 trials), and never collide. G3 is at 0–1. The
   rev 1 claim that point certificates are void under noise is now weak; do
   not headline it.
3. **Latency without compensation is still the clear failure**, for the
   textbook PSF and the robust tube PSF alike:
   - collisions of 2–30% at δ = 0.1–0.3 s;
   - at the measured solve time, 0–1 collisions but 20% of fallbacks invalid,
     and R1's tube broken on 35% of steps in 97% of trials.
4. **Compensation fixes it** (G2, R2: 0 everything). The defensible claim:
   PSFs from the literature assume zero computation time and their
   certificates break without it; making them sound needs the set pushed
   through the delay. Ours does that and pipelines at zero actuation latency,
   inside a certified ≤ 0.20 s budget.

## Rev 2 (2026-09-22): |ω| ≤ 1 enforced, and the literature robust (tube) PSF (superseded by rev 3)

**Current numbers.** The tables further down are rev 1 (`~/runs/alg29_main`);
their figures and CSV are archived in `rev1/`. Rev 2 run: `~/runs/alg29_ub`,
jobs 23450455 (array) and 23450456 (exclusive node). Pinned sources:
`alg29_generic_psf` 317af5d8…, `alg28_sound` acd2420f…,
`mpc_safety_filter_acados` 2120cf4a…. Same scenes and seed as rev 1.

### What changed

- **The |ω| ≤ 1 bound now holds on every flown filter input.**
  - `TTTCARV_SINGLE_UBOUND=reject` (new; default `count`, which leaves
    behaviour unchanged): the single-start solve now raises
    InputBoundViolation, as multi-start already did, instead of returning an
    over-limit iterate.
  - `TTTCARV_GP_ALIGN=1` (new): G2, G3 and R2 apply the new plan at the next
    whole tick, so no converted first input exceeds the bound. This costs up
    to one step of extra latency.
  - The NN itself stays unclipped: in the traces it reaches up to 1.75, above
    1 on about 10% of its steps. The bound is the filter's restriction on its
    own backups (D = √(r² + 2rR) needs it), not a plant limit.
- **The overshoots were rare.** Measured in a count-only rerun of rev 1
  (`~/runs/alg29_ubcount`, `ubcount_rev1.tsv`):
  - Baselines: 12–38 of 30 000 single-start solves overshot. These are solver
    blow-ups (peak 14–18), not near-misses. After the acceptance test, 1–22
    over-limit plans per 500 trials were flown.
  - G2 δ = 0.1: the converted first input exceeded 1 on about 1 200 landings.
  - **Ours:** 3–7 single-start overshoots out of about 430 solves (peak 1.4),
    and 0 accepted, so ours' rev 1 results are unaffected. The double
    integrator had none.
  - The deterministic fixed-δ arms reproduced rev 1's collision counts exactly,
    except G2 δ = 0.4 (2 vs 1 collisions), probably solver warm starts carrying
    over between trials.
- **New arms R1 and R2: the literature robust PSF**, an output-feedback tube
  PSF (Wabersich & Zeilinger style).
  - Start set: the set-membership box S.
  - Tube: the open-loop Lipschitz tube of the unicycle,
    e_k = e_p0 + k·v·dt·e_θ. The same controls keep the heading error at e_θ.
  - The OCP tightens stage k by e_k, using per-stage parameters on the same
    acados solver. The acceptance test runs on the tube boxes.
  - Audit `tube_out`: is the true state outside the active plan's tube?
  - R1 uses G1's timing: the literature assumption that the input lands at
    the tick it was planned for, then δ > 0 applied naively.
  - R2 pushes S through the held controls to the tick the plan starts
    (delay-compensated), with ALIGN.

### Unicycle, rev 2 (500 paired scenes / arm)

| arm | collisions | 95% CI | invalid-backup steps | trials ≥1 | tube broken (steps / trials) |
|---|---|---|---|---|---|
| G0, no noise | 0 | [0.0, 0.8]% | 0 | 0 | — |
| G0, noise | 1 | [0.0, 1.1]% | 419 (1.4%) | 158 | — |
| G1 nn δ = 0.05 / 0.1 / 0.15 / 0.2 / 0.3 / 0.4 | 3 / 14 / 155 / 6 / 185 / 63 | | 4.8 / 7.0 / 11.4 / 12.5 / 17.2 / 16.9% | ~410 | — |
| G1 nn δ = measured, no noise / noise | 4 / 4 | [0.3, 2.0]% | 4.9 / 5.1% | 392 / 383 | — |
| G2 (aligned) δ = 0.1 / 0.2 / 0.4, no noise | 0 / 0 / 2 | | 0 / 0 / 0 | 0 | — |
| G2 (aligned) δ = 0.1 / 0.2 / measured, noise | 1 / 1 / 1 | [0.0, 1.1]% | 1.6% | 154 | — |
| G3 δ = 0 / 0.2, noise | 0 / 0 | [0.0, 0.8]% | 1.0 / 0.9% | 77 / 69 | — |
| **R1 δ = 0 (literature)** | 1 | [0.0, 1.1]% | **0** | 0 | **0 / 0** |
| R1 δ = 0.05 | 3 | [0.2, 1.7]% | 434 (1.5%) | 184 | 44% / 500 |
| R1 δ = 0.1 | 20 | [2.6, 6.1]% | 1177 (4.0%) | 334 | 52% / 500 |
| R1 δ = 0.15 | **164** | [28.8, 37.0]% | 2616 (8.8%) | 388 | 58% / 500 |
| R1 δ = 0.2 | 2 | [0.1, 1.4]% | 1528 (5.2%) | 236 | **99.7% / 500** |
| R1 δ = 0.3 | **196** | [35.0, 43.5]% | 3504 (12.1%) | 374 | 99.7% / 500 |
| **R1 δ = measured** (median 25 ms) | **10** | [1.1, 3.6]% | 553 (1.9%) | 187 | 42% / 499 |
| **R2 δ = 0.1 / 0.2 / 0.4 / measured** | 1 / 1 / 1 / 1 | [0.0, 1.1]% | **0** | 0 | **0 / 0** |
| **ours (ideal / pipelined)** | **0 / 0** | [0.0, 0.8]% | every audit 0 | 0 | — |

- Every flown filter input satisfied |ω| ≤ 1 in every arm (`ub_flown` = 0).
- Ours: 901 / 939 activations (ideal / pipelined). Median wall per timestep
  was 0.126 s, higher than rev 1's 0.081 s; this run landed on a different
  exclusive node.
- R1/R2 fail more solves than G0 with noise (R1 δ = 0: 3 925 vs 3 132, +25%).
  That is the conservatism of the tube: e_12 reaches about 0.4 m.

### What rev 2 shows

1. **The literature's robust PSF is sound only under zero computation time.**
   At δ = 0 its tube is never broken and no backup is invalid. At the solve
   time we actually measure (median 25 ms), the true state leaves the planned
   tube on 42% of steps in 499 of 500 trials. It collides 10 / 500 against
   ours' 0 (R1 δ = 0 collides 1; see point 3).
   This is the gap the user asked about: robust PSFs handle noise and ignore
   latency.
2. **Delay compensation closes it (R2).** Pushing the start set through the
   held controls gives 0 broken tubes and 0 invalid backups at every δ. So
   the defensible claim is not "robust PSFs fail under latency". It is:
   - the standard formulation does not account for latency;
   - compensating needs the set pushed through the delay, which is what ours
     does, and ours also pipelines at zero actuation latency;
   - ours does it inside a certified ≤ 0.20 s budget.
3. **The one collision in the sound arms is the terminal set, not the tube.**
   R1 δ = 0 and every R2 arm collide in scene 889232576 (G0 and G2 with noise
   too). It happens after 8 loiter steps: the cached plan ran out while every
   re-solve was rejected. D = √(r² + 2rR) guarantees that the away-side circle
   clears ONE obstacle. With obstacles on both sides and loiter radius R = 1,
   both circles can be blocked. The trace is
   `~/runs/alg29_diag/r2_889232576.json`.
   - Ours shares this gap: its loiter always turns +1, and its own comment
     says repeated loiter steps are not covered. It loitered 13 steps in
     5 / 500 trials here, with 0 collisions.
   - The fix is a terminal test that one side's discrete loiter clears every
     obstacle from the whole terminal box.
   - Separately, the baselines re-pick the loiter side every step, so the
     loiter can flip sides. The side should be committed once the loiter
     starts. Not fixed yet.

## The baselines (`src/alg29_generic_psf.py`)

All arms share alg28's pieces, so only the filter logic differs:
- the same NN, reading the estimate
- the same bounded noise (|v| ≤ 0.15, uniform)
- the same set-membership estimator
- the same acados OCP: N = 12, path nodes clear raw r, terminal node clears
  D = √(r² + 2rR), |ω| ≤ 1, one-shot, multi-start
- the same acceptance test (`_rollout_verdict`)

| arm | what it is |
|---|---|
| G0 | textbook PSF, δ = 0 (computation is free) |
| G1 | textbook PSF with latency δ: solves from x̂_t, applies u₀ at t+δ as if no time had passed; a failed solve falls back to the cached plan. HOLD = what runs during the solve: `nn` (NN keeps control until the filter answers) or `prev` (zero-order hold) |
| G2 | delay-compensated: holds the cached backup during the solve, predicts x̂ through it, plans from the prediction, and converts u₀ so the step's heading increment is exact |
| G3 | G2 + constraint tightening by the estimate box (constant-size plan boxes, the pre-alg28 acceptance) |
| ours | alg28 final config (ESTCTRL=J, JSPLIT 2,2,16,1,1,4, EBOX=S, REANCHOR=1, SOUND_MPC=1, SETMEM=1) |
| ours pipelined | `TTTCARV_MEAS_DELAY=1` (new, default off): tick t sees only z_{t−1}, pushed through the known u_{t−1}, so the whole ≤ 0.20 s tick runs during the previous step and u_t lands exactly on the tick — no actuation latency |

**Timing model.** The plant integrates heading exactly under piecewise-constant
ω (θ' = θ + ∫ω). A command that switches inside a step is therefore modelled
exactly, and with switches on ticks this reduces to the discrete model every
OCP uses. The DI plant is exact LTI, so it is exact in continuous time.

**Certificate audit.** After every step on which a plan is active, the rest of
the filter's current backup is re-simulated from the **true** state and put
through the same acceptance test it passed when built. Nominally (δ = 0, exact
state) this is 0 by construction; G0 noise-free measures 0.

## Unicycle (500 paired scenes / arm)

| arm | collisions | 95% CI | steps with invalid backup | trials with ≥1 | fallbacks invalid |
|---|---|---|---|---|---|
| G0, no noise | 0 | [0.0, 0.8]% | 0 / 29 976 | 0 | 0 / 3072 |
| G0, noise | 1 | [0.0, 1.1]% | 419 (1.4%) | 159 | 164 / 3098 |
| G1 nn δ=0.05 | 3 | [0.2, 1.7]% | 1440 (4.8%) | 397 | 409 / 3076 |
| G1 nn δ=0.10 | 14 | [1.7, 4.6]% | 2089 (7.0%) | 418 | 487 / 3073 |
| G1 nn δ=0.15 | **155** | [27.1, 35.2]% | 3410 (11.4%) | 422 | 1380 / 3918 |
| G1 nn δ=0.20 | 6 | [0.6, 2.6]% | 3683 (12.5%) | 415 | **2835 / 3262 (87%)** |
| G1 nn δ=0.30 | **184** | [32.7, 41.1]% | 5065 (17.2%) | 412 | 1238 / 2010 |
| G1 nn δ=0.40 | 63 | [10.0, 15.8]% | 4895 (16.9%) | 415 | 1519 / 1693 |
| G1 prev δ=0.10 | 0 | [0.0, 0.8]% | 708 (2.4%) | 411 | 128 / 3165 |
| G1 prev δ=0.20 | 6 | [0.6, 2.6]% | 3687 (12.5%) | 415 | 2834 / 3265 |
| G1 nn δ=measured, no noise | 1 | [0.0, 1.1]% | 1074 (3.6%) | 356 | 336 / 3078 |
| G1 nn δ=measured, noise | 2 | [0.1, 1.4]% | 1187 (4.0%) | 336 | 426 / 3099 |
| G1 nn δ=0.10, noise | 17 | [2.1, 5.4]% | 2246 (7.5%) | 409 | 617 / 3100 |
| G1 nn δ=0.20, noise | 2 | [0.1, 1.4]% | 3447 (11.7%) | 397 | 2593 / 3226 |
| G2 δ=0.10 / 0.20 / 0.40, no noise | 0 / 0 / 1 | ≤ [0.0, 1.1]% | 0 / 0 / 0 | 0 | 0 |
| G2 δ=0.10, noise | 1 | [0.0, 1.1]% | 413 (1.4%) | 156 | 160 / 3102 |
| G2 δ=0.20, noise | 1 | [0.0, 1.1]% | 471 (1.6%) | 154 | 214 / 2996 |
| G2 δ=measured (plans for 0.2), noise | 1 | [0.0, 1.1]% | 467 (1.6%) | 154 | 211 / 2992 |
| G3 δ=0 / 0.20, noise | 0 / 0 | [0.0, 0.8]% | 301 / 277 (~1.0%) | 77 / 69 | 0 / 52, 0 / 54 |
| **ours (ideal timing)** | **0** | [0.0, 0.8]% | every audit 0 | 0 | — |
| **ours pipelined** | **0** | [0.0, 0.8]% | every audit 0 | 0 | — |

- The "ours" audits cover the lookahead, plan boxes, plan nodes and
  set-membership (tight / meas / post), plus the delayed box in the pipelined
  arm. Ours had 961 activations with ideal timing and 1002 pipelined (+4%).
  Median wall/timestep was 0.081 vs 0.082 s.
- Measured solve time (multi-start): median 25 ms, p99 ≈ 50 ms, max 0.10 s.
  G2 with measured latency had 0 deadline misses against its 0.2 s plan.

Figures: `fig_unicycle_latency.png` (collision rate and invalid-certificate
rate vs δ) and `fig_unicycle_trajectory.png`. The trajectory figure is scene
579323432 under G0 / G1 δ=0.1 / G2 δ=0.1. In G1 the cached backup still
swerves away, but when the filter falls back on it at t=41 the robot is
already at the obstacle.

## Double integrator (tight start, 500 trials / arm, exact state)

| arm | violations | steps with invalid backup | trials with ≥1 |
|---|---|---|---|
| G0 | 0 | 0 / 20 000 | 0 |
| G1 δ=0.05 | 0 | 1478 (7.4%) | 499 |
| G1 δ=0.10 | 19 (3.8%, [2.4, 5.9]) | 2983 (14.9%) | 500 |
| G1 δ=0.15 | **330 (66%, [61.7, 70.0])** | 6539 (33.9%) | 500 |
| G1 δ=0.20 | 0 | 0 | 0 |
| G2 δ=0.10 / 0.20 | 0 / 0 | 1 / 0 | 1 / 0 |
| alg22_di (ours, exact state) | 0 | — | — |

The DI has no set-membership estimator, so it runs noise-free. With exact
state, the one-tick-old measurement pushed through the known control IS the
exact state, so a pipelined alg22_di is identical to the ideal run; it was not
run separately.

## What this shows — and what it does not

1. **Your claim, measured directly.** When the naive PSF falls back on its
   cached backup, that backup is often no longer valid from where the robot
   really is: 87% of fallbacks at δ = 0.2 s and 11–14% at the real measured
   latency. Invalid-certificate steps grow roughly linearly with δ (4.8% at
   0.05 s, 17% at 0.3 s). Ours: 0 on every audit, in both the ideal and the
   pipelined arm.
2. **Collisions follow, but not monotonically in δ.** The dangerous regime is
   the NN keeping control during every solve with a FRACTIONAL δ: at 0.15 s the
   NN keeps 75% of every step (31% collisions; 66% on the DI). At δ = dt the
   late plan still gets whole steps and collisions drop to ~1%. So the
   architecture of the naive PSF matters as much as δ does.
3. **Latency alone is fixable.** G2 (predict through the held backup, convert
   the first input) is exact nominally: 0 invalid certificates and ~0
   collisions at every δ tested. A reviewer will say this, and the data
   agrees. The honest separation from ours is therefore not "generic PSFs
   collide under latency" but:
   - **point certificates are void under state noise even when
     delay-compensated.** G2 and G0 with noise have ~1.5% of steps flying an
     invalid backup, in ~31% of trials. Constant-box tightening (G3) roughly
     halves that (~1%, ~15% of trials) but does not remove it.
   - **ours is sound in both.** It is set-based, and it pipelines at zero
     actuation latency (MEAS_DELAY=1) at +4% activations, which it can do
     because every tick is budgeted to ≤ 0.20 s.
4. **Collision counts under noise are small** (0–2 / 500 for G0/G2/G3) on these
   scenes, because the D terminal set and receding re-solves leave margin.
   Quote certificate validity as the headline and collisions as supporting
   evidence. Do not claim that generic PSFs crash under noise.

## Caveats

- G1/G2 get alg28's multi-start solver. With single-start, G0 alone had 4/96
  collisions, from RTI local minima. The single-start textbook PSF is
  therefore strictly weaker than shown here.
- The textbook terminal controller (a max-turn loiter once the cached plan
  runs out) turns to the side whose DISCRETE loiter keeps more clearance. The
  first version always turned left, and the continuous-circle rule fails
  under the Euler plant (it cuts ~0.1 m inside the circle); both produced
  artificial G0 collisions.
- (Rev 1 only.) Applied turn rates could exceed |ω| = 1: from G2's converted
  first input, and from single-start solves that skipped `_input_bounds_ok`.
  Rev 2 closes both; see above.
- Scenes are the random-obstacle benchmark, not adversarial.
- n = 500 per arm; the CIs are in the table.
- "ours" per-step budget maxima were not logged in this run (no worker logs).
  The alg28 10k run measured max 151 ms.
