# alg14_di — split-terminal PSF on the double integrator

`p >= 0.0`, `v >= -0.9`, horizon 40 steps, 8 seeds

| seed | PSF real | PSF RSOA | nominal real | nominal RSOA | mpc_calls | no_backup | queue_empty | aborted | max &#124;du&#124; |
|---|---|---|---|---|---|---|---|---|---|
| 1401830092 | 0 | 23 | 0 | 37 | 4 | 0 | 0 | no | 0.841 |
| 1826701614 | 0 | 23 | 0 | 37 | 4 | 0 | 0 | no | 1.174 |
| 1367864806 | 0 | 23 | 2 | 37 | 4 | 0 | 0 | no | 1.057 |
| 1097657231 | 0 | 23 | 0 | 37 | 4 | 0 | 0 | no | 0.875 |
| 579362555 | 0 | 23 | 0 | 37 | 4 | 0 | 0 | no | 1.142 |
| 661058651 | 0 | 23 | 0 | 37 | 4 | 0 | 0 | no | 0.880 |
| 87989972 | 0 | 23 | 0 | 37 | 4 | 0 | 0 | no | 0.879 |
| 161576974 | 0 | 23 | 0 | 37 | 4 | 0 | 0 | no | 0.874 |

## Aggregate

- `n_seeds` = 8
- `pos_min` = 0.0
- `vel_min` = -0.9
- `max_time` = 40
- `psf_seeds_with_real_viol` = 0
- `psf_seeds_with_rsoa_viol` = 8
- `nom_seeds_with_real_viol` = 1
- `nom_seeds_with_rsoa_viol` = 8
- `psf_aborted` = 0
- `psf_no_backup_total` = 0
- `psf_queue_empty_total` = 0
