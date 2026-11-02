#!/bin/bash
#SBATCH -J a28noise2
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 06:00:00
#SBATCH -o /home/ryosei_t/runs/alg28_noise2/job_%j.log
#SBATCH -e /home/ryosei_t/runs/alg28_noise2/job_%j.err
# Is the eboxS benefit distribution-robust, or an artefact of uniform noise?
#
# The certified set shrinks by intersecting consecutive measurement boxes, so
# what drives the collapse of hw_theta(S) is the OFFSET between consecutive
# boxes, i.e. the spread of v_k - v_{k-1}. Measured at sigma=0.05, vmax=3sigma:
#   mean |offset|   uniform 0.1003   truncated-normal 0.0558   (1.8x smaller)
# So a peaked law inside the SAME bound is the pessimistic case for
# set-membership. Same assumption satisfied, possibly much less benefit.
# This is NOT a violation test -- every draw still obeys |v| <= vmax exactly.
#
# 3 estimator arms x 2 noise laws. "% of excess over the floor removed" is only
# defined against the truth arm, so truth is run under BOTH laws.
# ALL SIX ARMS RUN SEQUENTIALLY ON ONE EXCLUSIVE NODE: activations are priced
# against wall-clock, and the original ebox batch (job 23301747) put its arms on
# separate nodes. Same seed as that batch, so scenes are paired with it.
set -o pipefail
export PATH=/home/ryosei_t/.conda/envs/carv/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
export ACADOS_SOURCE_DIR=/home/ryosei_t/acados ACADOS_LIB_DIR=/home/ryosei_t/acados/lib
export LD_LIBRARY_PATH=/home/ryosei_t/acados/lib:${LD_LIBRARY_PATH:-}
export TTTCARV_WORKERS=8 TTTCARV_TRIALS=${TRIALS:-200} TTTCARV_GOAL_KEEPOUT=0
export TTTCARV_ALG=alg28_sound
export TTTCARV_RECEDING=1 TTTCARV_ONESHOT=1 TTTCARV_ONESHOT_REFINE=2 TTTCARV_ONESHOT_VERIFY=1
export TTTCARV_BOOTSTRAP_RESIM=1 TTTCARV_PSF_LOITER=1 TTTCARV_PSF_TAIL=1 TTTCARV_FIRST_STEP_COST=0
export TTTCARV_TAIL_COST_WEIGHT=1e-3 TTTCARV_RESOLVE_RESERVE=0 TTTCARV_SINGLE_START_FRAC=0.50
export TTTCARV_MPC_PRICING=1 TTTCARV_ONESHOT_RTI=10 TTTCARV_BUDGET_MARGIN=0.003 TTTCARV_GC_PER_TRIAL=1
export TTTCARV_SETMEM=1 TTTCARV_KF_RESET=0
export TTTCARV_SEED=20260824
export TTTCARV_VMAX_SIGMA=3.0 TTTCARV_SOUND_MPC=1
export TTTCARV_ESTCTRL=J TTTCARV_JSPLIT=2,2,16,1,1,4
export TTTCARV_PLAN_CELLS=1 TTTCARV_REANCHOR=0
OUT=/home/ryosei_t/runs/alg28_noise2
cd /home/ryosei_t/CARV
echo "node=$(hostname) start=$(date '+%F %T') commit=$(git rev-parse --short HEAD)"
sha256sum -c --quiet $OUT/expected.sha256 || { echo "!! SOURCE DIFFERS -- not running"; exit 3; }
rc=0
for arm in unif_X unif_S unif_truth trunc_X trunc_S trunc_truth; do
  [ -s $OUT/fp_$arm.json ] && { echo "=== $arm already done -- skipping ==="; continue; }
  case $arm in
    unif_*)  export TTTCARV_NOISE=uniform ;;
    trunc_*) export TTTCARV_NOISE=trunc ;;
  esac
  export TTTCARV_ESTCTRL=J TTTCARV_EBOX=X
  case $arm in
    *_X)     export TTTCARV_EBOX=X ;;
    *_S)     export TTTCARV_EBOX=S ;;
    *_truth) export TTTCARV_ESTCTRL=0 ;;
  esac
  echo "=== arm=$arm noise=$TTTCARV_NOISE ebox=$TTTCARV_EBOX estctrl=$TTTCARV_ESTCTRL $(date '+%T') ==="
  LOCAL=${TMPDIR:-/tmp}/carv_a28n_${SLURM_JOB_ID}_$arm
  rm -rf $LOCAL; mkdir -p $LOCAL/wlog
  export TTTCARV_WORKER_LOG_DIR=$LOCAL/wlog
  TTTCARV_FINGERPRINT=$OUT/fp_$arm.json python nfl_robustness_training/src/run_trials_random_obs.py \
    > $LOCAL/chunk.log 2>&1
  a=$?; [ $a -ne 0 ] && rc=$a
  gzip -c $LOCAL/chunk.log > $OUT/chunk_$arm.log.gz
  tar -C $LOCAL -czf $OUT/wlog_$arm.tar.gz wlog
  grep -a -E "Safety record|trials FAILED|UNSAFE|AUDIT MISS|truth_out" $LOCAL/chunk.log | tail -6
  rm -rf $LOCAL
  echo "=== arm=$arm exit=$a end=$(date '+%T') ==="
done
echo "ALL DONE rc=$rc $(date '+%F %T')"
exit $rc
