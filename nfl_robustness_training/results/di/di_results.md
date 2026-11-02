# alg14_di — split-terminal PSF on the double integrator

100 seeds, PSF vs. unfiltered NN baseline. **PSF: 0 real violations, 0 RSOA
violations.** Nominal: 21 seeds with real violations (42 total), 60 seeds with
RSOA violations (214 total).

Provenance: commit `ae8b11d` + uncommitted per-stage-buffer patch to
`utils/di_mpc_acados.py` (see Caveats). SLURM job 21151603, `mit_normal`,
7 min 17 s. Raw log `run.log`, per-seed JSON `di_results.json`.

## Scenario and constraint conditions

**There are no obstacles in this scenario.** The DI demonstration constrains
the state directly with two half-planes rather than with obstacle geometry;
`HalfPlaneConstraints(pos_min, vel_min)` replaces the `Obstacles` object on the
tester (`alg14_di.py:419-421`).

| | value | source |
|---|---|---|
| system | double integrator, `dt = 0.2 s` | `setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')` |
| state | `x = (p, v)` | |
| **position constraint** | **`p >= 0.0`** | `POS_MIN` in `alg14_di_figs.py:198` |
| **velocity constraint** | **`v >= -0.90`** | `VEL_MIN` in `alg14_di_figs.py:197` |
| control bound | `\|u\| <= 1.0` | `u_max` in `alg14_di.py:429` |
| initial set | `p in [2.5, 3.0]`, `v in [-0.25, 0.25]` | analyzer default, `REAL_integrated_sim.py:1288` |
| concrete start | drawn uniform from that box, `RandomState(seed)` | `REAL_integrated_sim.py:191-199` |
| episode length | 40 timesteps (8 s) | `MAX_TIME` |
| MPC horizon | 12 steps, acados SQP_RTI | `n_horizon` |
| MPC margin | `buffer = 0.05`, applied to per-stage `p`, per-stage `v`, and terminal `p` | `di_mpc_acados.py:197,211-212` |
| MPC terminal set | parabolic max-brake invariant set | `di_mpc_acados.py` |
| process noise | `sigma = 0.01` | `ReachabilityTester` default |
| measurement noise | `sigma = 0.05`, Kalman estimator | `ReachabilityTester` default |
| compute budget | 0.20 s per timestep, costs calibrated at start of each trial | `TimeBudget` |
| seeds | `1401830092` then 99 from `np.random.default_rng(0)` | `alg14_di_figs.py:202-203` |

`vel_min = -0.90` is deliberately tight — chosen so the unfiltered controller's
reachable set violates it on a majority of seeds, which is what makes the
comparison informative. The nominal NN controller is not trying to respect
these half-planes; it was trained to drive `p` to 0.

**Violation definitions.** A *real* violation is the true simulated state
crossing a half-plane: `p <= pos_min or v <= vel_min`. An *RSOA* violation is
the lower corner of the certified reachable-set box crossing it:
`bounds[0,0] <= pos_min or bounds[1,0] <= vel_min` (`alg14_di.py:118-120`).
RSOA violations are the stricter test — the certificate can fail while the true
state stays safe.

## Per-seed results

| seed | PSF real | PSF RSOA | nominal real | nominal RSOA | mpc_calls | no_backup | queue_empty | aborted | max &#124;du&#124; |
|---|---|---|---|---|---|---|---|---|---|
| 1401830092 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | no | 0.668 |
| 1826701614 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | no | 0.000 |
| 1367864806 | 0 | 0 | 2 | 5 | 2 | 0 | 0 | no | 0.546 |
| 1097657231 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.629 |
| 579362555 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | no | 0.892 |
| 661058651 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 87989972 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 161576974 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | no | 0.639 |
| 35492826 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 376383645 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 1746484539 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | no | 0.334 |
| 1394609702 | 0 | 0 | 1 | 4 | 1 | 0 | 0 | no | 0.512 |
| 1960127675 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.766 |
| 1081530686 | 0 | 0 | 2 | 6 | 2 | 0 | 0 | no | 0.499 |
| 1302740407 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 2084654369 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | no | 0.886 |
| 1566581934 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.629 |
| 1357791198 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.651 |
| 1167425778 | 0 | 0 | 2 | 5 | 1 | 0 | 0 | no | 0.446 |
| 1202413557 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | no | 0.270 |
| 2008052738 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | no | 0.000 |
| 595598245 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.569 |
| 1752032165 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 1440696407 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 5880883 | 0 | 0 | 2 | 6 | 1 | 0 | 0 | no | 0.539 |
| 846428919 | 0 | 0 | 2 | 5 | 1 | 0 | 0 | no | 0.610 |
| 1841261662 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | no | 0.000 |
| 1190382222 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 72124473 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | no | 0.289 |
| 1642588603 | 0 | 0 | 2 | 5 | 2 | 0 | 0 | no | 0.590 |
| 1566923138 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | no | 0.358 |
| 1818006482 | 0 | 0 | 1 | 4 | 1 | 0 | 0 | no | 0.795 |
| 377217572 | 0 | 0 | 0 | 5 | 1 | 0 | 0 | no | 0.453 |
| 191741830 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 1853662620 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | no | 0.510 |
| 47463574 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | no | 0.000 |
| 1162779115 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | no | 0.000 |
| 172656923 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | no | 0.845 |
| 643626383 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 1033071480 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.613 |
| 907713895 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 865948037 | 0 | 0 | 2 | 4 | 2 | 0 | 0 | no | 0.532 |
| 60816030 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | no | 0.000 |
| 11494707 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | no | 0.854 |
| 266896303 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | no | 0.000 |
| 17790419 | 0 | 0 | 3 | 6 | 2 | 0 | 0 | no | 0.807 |
| 1440154963 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.571 |
| 1128755548 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 1389828892 | 0 | 0 | 3 | 5 | 1 | 0 | 0 | no | 0.575 |
| 552547096 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.814 |
| 1321529463 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 1640795441 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | no | 0.000 |
| 823941273 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | no | 0.412 |
| 989821580 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | no | 0.441 |
| 2141492029 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | no | 0.000 |
| 1728701086 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | no | 0.588 |
| 2106327850 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.824 |
| 815020182 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | no | 0.000 |
| 1472190200 | 0 | 0 | 3 | 5 | 1 | 0 | 0 | no | 0.492 |
| 2040324894 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.622 |
| 1396850658 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | no | 0.370 |
| 1804554973 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | no | 0.646 |
| 1478428095 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.657 |
| 1511830742 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 835202397 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | no | 0.000 |
| 1879383516 | 0 | 0 | 2 | 5 | 1 | 0 | 0 | no | 0.488 |
| 290117534 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.657 |
| 1243185673 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 1549384411 | 0 | 0 | 3 | 6 | 1 | 0 | 0 | no | 0.522 |
| 1815655708 | 0 | 0 | 1 | 4 | 1 | 0 | 0 | no | 0.483 |
| 1128189815 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.831 |
| 806201099 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.666 |
| 666239354 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 908297867 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | no | 0.000 |
| 1043323488 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.684 |
| 1543657888 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 1910160578 | 0 | 0 | 1 | 4 | 2 | 0 | 0 | no | 0.645 |
| 156652360 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | no | 0.566 |
| 2005843176 | 0 | 0 | 3 | 6 | 2 | 0 | 0 | no | 0.827 |
| 1141051469 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | no | 0.265 |
| 768359333 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | no | 0.000 |
| 1444696807 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | no | 0.354 |
| 1227350964 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | no | 0.725 |
| 546762614 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 691209253 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | no | 0.000 |
| 1545136976 | 0 | 0 | 0 | 5 | 1 | 0 | 0 | no | 0.606 |
| 1276249595 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 1083230752 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 725658830 | 0 | 0 | 3 | 6 | 2 | 0 | 0 | no | 0.625 |
| 1633503581 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.663 |
| 840995399 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 704750836 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | no | 0.700 |
| 1911849612 | 0 | 0 | 2 | 4 | 1 | 0 | 0 | no | 0.606 |
| 566779217 | 0 | 0 | 1 | 5 | 1 | 0 | 0 | no | 0.658 |
| 487817217 | 0 | 0 | 0 | 5 | 1 | 0 | 0 | no | 0.558 |
| 1533673821 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | no | 0.437 |
| 1338284201 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 104252791 | 0 | 0 | 1 | 5 | 1 | 0 | 0 | no | 0.529 |
| 180421576 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | no | 0.000 |
| 809899613 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |

## Aggregate

- `n_seeds` = 100
- `pos_min` = 0.0
- `vel_min` = -0.9
- `max_time` = 40
- `psf_seeds_with_real_viol` = 0
- `psf_seeds_with_rsoa_viol` = 0
- `nom_seeds_with_real_viol` = 21
- `nom_seeds_with_rsoa_viol` = 60
- `psf_aborted` = 0
- `psf_no_backup_total` = 0
- `psf_queue_empty_total` = 0


## Caveats

1. **The per-stage buffer is uncommitted and is a mitigation, not a proof.**
   The previous run (preserved at `../results_di_prebuffer/`) had 8 RSOA
   violations, all landing exactly one step after a PSF activation. Root cause:
   `buffer` was applied only to the terminal position, so the per-stage corner
   was free to graze `vel_min` exactly, while `hw_p`/`hw_v` are held constant
   across the MPC horizon even though the analyzer's true RSOA widens (e.g.
   0.1399 -> 0.1537 in `v` over one step). A plan verified at the boundary
   therefore lands just below it at `tau+1`. Extending `buffer` to the
   per-stage constraints clears all 8. The sound fix is per-stage half-widths
   that track the analyzer's propagation, not a constant margin.

2. **Part of the RSOA gap is bound tightness, not filtering.** On 15 of the 100
   seeds the PSF never intervened at all (`mpc_calls = 0`, `max|du| = 0`) yet
   still reported 0 RSOA violations where the nominal baseline reported 1-3.
   With identical controls and identical seeds the trajectories are identical,
   so that difference cannot come from the filter. It comes from `run_nominal`
   propagating only the empirical set while `test` also runs symbolic and
   concrete refinement, producing a tighter certificate at the same state.
   `run_nominal`'s docstring claim of "the SAME estimator / RSOA machinery" is
   accurate for the estimator but not for the refinement. The *real*-violation
   comparison (0 vs 42) is unaffected by this, since real states depend only on
   the applied control.

3. **The run is not bit-reproducible.** Admission to the 0.20 s budget depends
   on wall-clock, so intervention counts move between runs on identical seeds:
   total `mpc_calls` was 131 before the patch and 70 after, with 58 of 100
   seeds changing. Some of that is the patch, some is timing jitter; the two
   are not separated here.
