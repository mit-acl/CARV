# Archive (2026-10-14)

Superseded, untracked work moved out of `src/`, `nfl_robustness_training/` and the
repo root. Nothing here is imported by the live code: alg28_sound, alg29_generic_psf,
alg22_di, Reach_PSF_FINAL_NL and REAL_integrated_sim. These files are not in git, so
this folder is their only copy.

Scripts still `import REAL_integrated_sim` etc., so to run one, copy it back into
`src/` or put `src/` on `PYTHONPATH`. Sbatch scripts in `~/scripts` that pin these
paths by hash will fail closed until the file is back in place.

## src/ — the lineage that led to alg28_sound

| file | what it was |
|---|---|
| alg23_tail_recheck.py | Reach_PSF_FINAL + a tail re-check on the fallback path; the fix for uncertified fallbacks, parked unless reviewers flag it |
| alg24_kf_noreset.py | removes the sim's reset-to-truth of the EKF |
| alg25_nokf.py | estimator box removed from the certified bounds |
| alg25_di_nokf.py, alg25_di_figs.py, alg25_di_anim.py | the DI version of alg25, its figures and animation |
| alg26_setmem.py | sound set-membership box replaces the EKF box (bounded noise) |
| alg26b_noreset.py | alg26 + no estimator reset |
| alg26c_repair.py, alg26d_mbfix.py | alg26b repairs |
| alg26g_gauss.py | alg26 under Gaussian noise |
| alg26v_verify.py | alg26 with verification audits |
| alg27_estctrl.py | the NN reads the estimate (estimate feedback); alg28 = alg27 made sound |
| alg14_di_kfbox_figs.py | alg14_di figures with the raw Kalman 3σ box |
| exp28*.py | standalone state-uncertainty studies behind alg28 (lookahead looseness, splitting, symbolic/α-CROWN, error-box rule S, DI port). Findings are in memory (exp28 state-uncertainty findings) and results_alg28/ |
| alg19_default.gif/.mp4 | alg19 animation |
| *.bak_20260911, utils/*.bak_20260911 | pre-edit copies from 2026-09-11 |

## results/

`results_di_kfbox`, `results_di_nokf` and `results_di_prebuffer` are DI result folders
from the alg14/alg25 era.

## root/

These are from the repo root:
- alg19/alg21 animations; `alg19_psf_buffer.gif` was empty
- the 2026-08-28 trial logs (~40 MB)
- `run_trials.sh`, an alg21 sbatch
- stale default outputs `trials_fingerprint.json` and `trials_di.png`

## 2026-11-02 cleanup (reach-psf branch)

`src/pre_alg28/` holds every module from `src/` that none of the live code
imports any more: alg8–alg21 experiments, their animation scripts, the old
`run_trials.py` / `tester.py` pipeline and the deprecated timing scripts. To
run one, copy it back into `src/`.

`root/` also got the repo-root clutter: `diag_alg15_reasons.py`,
`repro_alg15.py`, `timing_analysis/`, and stray images and gifs.

The `.bak_*` snapshots of alg28_sound / alg29_generic_psf / anim_alg28 were
moved out of the repo to `~/runs/_bak_src_20261102/` on Engaging.
