#!/bin/bash
#SBATCH -J a28ra
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -o /home/ryosei_t/runs/alg28_ra/job_%j.log
#SBATCH -e /home/ryosei_t/runs/alg28_ra/job_%j.err
# 2x2: TTTCARV_REANCHOR x TTTCARV_PLAN_CELLS, paired scenes, ALL FOUR ARMS RUN
# SEQUENTIALLY ON ONE EXCLUSIVE NODE. The earlier split batch put each arm on a
# separate node and the control landed on a busier one; since activations are
# priced against wall-clock, that alone can move the count. Same node, one arm
# at a time, nothing else on it.
set -o pipefail
export PATH=/home/ryosei_t/.conda/envs/carv/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
export ACADOS_SOURCE_DIR=/home/ryosei_t/acados ACADOS_LIB_DIR=/home/ryosei_t/acados/lib
export LD_LIBRARY_PATH=/home/ryosei_t/acados/lib:${LD_LIBRARY_PATH:-}
export TTTCARV_WORKERS=8 TTTCARV_TRIALS=${TRIALS:-200} TTTCARV_GOAL_KEEPOUT=0
export TTTCARV_SEED=20260824
export TTTCARV_ALG=alg28_sound
export TTTCARV_RECEDING=1 TTTCARV_ONESHOT=1 TTTCARV_ONESHOT_VERIFY=1 TTTCARV_ONESHOT_RTI=10
export TTTCARV_ONESHOT_REFINE=2 TTTCARV_BOOTSTRAP_RESIM=1 TTTCARV_PSF_LOITER=1 TTTCARV_PSF_TAIL=1
export TTTCARV_FIRST_STEP_COST=0 TTTCARV_TAIL_COST_WEIGHT=1e-3 TTTCARV_RESOLVE_RESERVE=0
export TTTCARV_SINGLE_START_FRAC=0.50 TTTCARV_MPC_PRICING=1 TTTCARV_BUDGET_MARGIN=0.003
export TTTCARV_GC_PER_TRIAL=1
export TTTCARV_SETMEM=1 TTTCARV_KF_RESET=0 TTTCARV_SOUND_MPC=1
export TTTCARV_ESTCTRL=J TTTCARV_JSPLIT=2,2,16,1,1,4
export TTTCARV_VMAX_SIGMA=3.0 TTTCARV_NOISE=uniform
OUT=/home/ryosei_t/runs/alg28_ra
cd /home/ryosei_t/CARV
echo "node=$(hostname) start=$(date '+%F %T') commit=$(git rev-parse --short HEAD)"
sha256sum -c --quiet $OUT/expected.sha256 || { echo "!! SOURCE DIFFERS -- not running"; exit 3; }
rc=0
for arm in pc1_ra0 pc1_ra1 pc16_ra0 pc16_ra1; do
  case $arm in
    pc1_ra0)  export TTTCARV_PLAN_CELLS=1  TTTCARV_REANCHOR=0 ;;
    pc1_ra1)  export TTTCARV_PLAN_CELLS=1  TTTCARV_REANCHOR=1 ;;
    pc16_ra0) export TTTCARV_PLAN_CELLS=16 TTTCARV_REANCHOR=0 ;;
    pc16_ra1) export TTTCARV_PLAN_CELLS=16 TTTCARV_REANCHOR=1 ;;
  esac
  # Preemptable + --requeue restarts this script from the top. Arms that already
  # produced a fingerprint are skipped so a requeue resumes instead of looping.
  if [ -s $OUT/fp_$arm.json ]; then
    echo "=== arm=$arm already done (fingerprint present) -- skipping ==="
    continue
  fi
  echo "=== arm=$arm cells=$TTTCARV_PLAN_CELLS reanchor=$TTTCARV_REANCHOR $(date '+%T') ==="
  LOCAL=${TMPDIR:-/tmp}/carv_a28ra_${SLURM_JOB_ID}_$arm
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
echo "ALL DONE rc=$rc $(date '+%F %T')"
exit $rc
