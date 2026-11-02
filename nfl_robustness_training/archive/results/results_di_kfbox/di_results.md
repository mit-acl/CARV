# alg14_di — split-terminal PSF on the double integrator

`p >= 0.0`, `v >= -0.9`, horizon 40 steps, 8 seeds

| seed | PSF real | PSF RSOA | nominal real | nominal RSOA | mpc_calls | no_backup | queue_empty | aborted | max &#124;du&#124; |
|---|---|---|---|---|---|---|---|---|---|
| 1401830092 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | no | 0.668 |
| 1826701614 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | no | 0.857 |
| 1367864806 | 0 | 0 | 2 | 5 | 2 | 0 | 0 | no | 0.579 |
| 1097657231 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 579362555 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | no | 0.000 |
| 661058651 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 87989972 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no | 0.000 |
| 161576974 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | no | 0.000 |

## Aggregate

- `n_seeds` = 8
- `pos_min` = 0.0
- `vel_min` = -0.9
- `max_time` = 40
- `psf_seeds_with_real_viol` = 0
- `psf_seeds_with_rsoa_viol` = 0
- `nom_seeds_with_real_viol` = 1
- `nom_seeds_with_rsoa_viol` = 5
- `psf_aborted` = 0
- `psf_no_backup_total` = 0
- `psf_queue_empty_total` = 0
