#!/bin/bash
#SBATCH -J a28wall
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 06:00:00
#SBATCH -o /home/ryosei_t/runs/alg28_wallcensus/job_%j.log
#SBATCH -e /home/ryosei_t/runs/alg28_wallcensus/job_%j.err
# Wall census + k sweep. Counts, per trial, how many timesteps with a live
# conflict actually leave a wall P3b could not pass (rc_wall_steps) and which
# gate refuses the re-anchor (rc_blk_*). rc_blk_rearm under k=0 is exactly how
# many extra re-walks a fire-on-every-wall policy would buy, so it settles
# whether the once-per-conflict threshold is doing anything at all.
# k in {0,1,2,4} x noise {uniform,trunc}, ALL EIGHT ARMS SEQUENTIALLY ON ONE
# EXCLUSIVE NODE -- activations are priced against wall clock.
set -o pipefail
export PATH=/home/ryosei_t/.conda/envs/carv/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
export ACADOS_SOURCE_DIR=/home/ryosei_t/acados ACADOS_LIB_DIR=/home/ryosei_t/acados/lib
export LD_LIBRARY_PATH=/home/ryosei_t/acados/lib:${LD_LIBRARY_PATH:-}
export TTTCARV_WORKERS=8 TTTCARV_TRIALS=${TRIALS:-200} TTTCARV_GOAL_KEEPOUT=0
export TTTCARV_SEED=20260824
export TTTCARV_ALG=alg28_sound
# alg35_thr arm u_ours_ideal -- the current evaluation config.
export TTTCARV_RECEDING=1 TTTCARV_ONESHOT=1 TTTCARV_ONESHOT_VERIFY=1 TTTCARV_ONESHOT_RTI=10
export TTTCARV_ONESHOT_REFINE=2 TTTCARV_BOOTSTRAP_RESIM=1 TTTCARV_PSF_LOITER=1 TTTCARV_PSF_TAIL=1
export TTTCARV_FIRST_STEP_COST=0 TTTCARV_TAIL_COST_WEIGHT=1e-3 TTTCARV_RESOLVE_RESERVE=0
export TTTCARV_SINGLE_START_FRAC=0.50 TTTCARV_MPC_PRICING=1 TTTCARV_BUDGET_MARGIN=0.003
export TTTCARV_GC_PER_TRIAL=1
export TTTCARV_SETMEM=1 TTTCARV_KF_RESET=0 TTTCARV_SOUND_MPC=1
export TTTCARV_ESTCTRL=J TTTCARV_JSPLIT=2,2,16,1,1,4
export TTTCARV_VMAX_SIGMA=3.0 TTTCARV_EBOX=S TTTCARV_MEAS_DELAY=0
export TTTCARV_SINGLE_UBOUND=reject TTTCARV_TERMINAL=loiter
export TTTCARV_REANCHOR=1 TTTCARV_PLAN_CELLS=1
OUT=/home/ryosei_t/runs/alg28_wallcensus
cd /home/ryosei_t/CARV
echo "node=$(hostname) start=$(date '+%F %T') commit=$(git rev-parse --short HEAD)"
sha256sum -c --quiet $OUT/expected.sha256 || { echo "!! SOURCE DIFFERS -- not running"; exit 3; }
rc=0
for noise in uniform trunc; do
for k in 0 1 2 4; do
  arm=${noise}_k${k}
  export TTTCARV_NOISE=$noise TTTCARV_REANCHOR_REARM=$k
  if [ -s $OUT/fp_$arm.json ]; then
    echo "=== arm=$arm already done (fingerprint present) -- skipping ==="
    continue
  fi
  echo "=== arm=$arm noise=$noise k=$k $(date '+%T') ==="
  LOCAL=${TMPDIR:-/tmp}/carv_a28wall_${SLURM_JOB_ID}_$arm
  rm -rf $LOCAL; mkdir -p $LOCAL/wlog
  export TTTCARV_WORKER_LOG_DIR=$LOCAL/wlog
  TTTCARV_FINGERPRINT=$OUT/fp_$arm.json python nfl_robustness_training/src/run_trials_random_obs.py \
    > $LOCAL/chunk.log 2>&1
  a=$?; [ $a -ne 0 ] && rc=$a
  gzip -c $LOCAL/chunk.log > $OUT/chunk_$arm.log.gz
  grep -a -E "Safety record|trials FAILED|UNSAFE|AUDIT MISS" $LOCAL/chunk.log | tail -4
  rm -rf $LOCAL
  echo "=== arm=$arm exit=$a end=$(date '+%T') ==="
done
done
echo "ALL DONE rc=$rc $(date '+%F %T')"
exit $rc
