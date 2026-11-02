#!/bin/bash
#SBATCH -J a28split
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -c 10
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH --array=1-3
#SBATCH -o /home/ryosei_t/runs/alg28_split/job_%A_%a.log
#SBATCH -e /home/ryosei_t/runs/alg28_split/job_%A_%a.err
# Paired arms for the plan-box theta split. Identical scenes and config to the
# 10k headline arm; the ONLY difference is TTTCARV_PLAN_CELLS.
#   1  cells=1   control -- must reproduce the current numbers exactly
#   2  cells=4
#   3  cells=16
set -o pipefail
export PATH=/home/ryosei_t/.conda/envs/carv/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
export ACADOS_SOURCE_DIR=/home/ryosei_t/acados ACADOS_LIB_DIR=/home/ryosei_t/acados/lib
export LD_LIBRARY_PATH=/home/ryosei_t/acados/lib:$LD_LIBRARY_PATH
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
case $SLURM_ARRAY_TASK_ID in
  1) ARM=c1;  export TTTCARV_PLAN_CELLS=1 ;;
  2) ARM=c4;  export TTTCARV_PLAN_CELLS=4 ;;
  3) ARM=c16; export TTTCARV_PLAN_CELLS=16 ;;
esac
OUT=/home/ryosei_t/runs/alg28_split
cd /home/ryosei_t/CARV
echo "arm=$ARM node=$(hostname) start=$(date '+%F %T') commit=$(git rev-parse --short HEAD)"
sha256sum -c --quiet $OUT/expected.sha256 || { echo "!! SOURCE DIFFERS -- not running"; exit 3; }
env | grep '^TTTCARV_' | sort | sed 's/^/  /'
LOCAL=${TMPDIR:-/tmp}/carv_a28s_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
rm -rf $LOCAL; mkdir -p $LOCAL/wlog
export TTTCARV_WORKER_LOG_DIR=$LOCAL/wlog
TTTCARV_FINGERPRINT=$OUT/fp_$ARM.json python nfl_robustness_training/src/run_trials_random_obs.py \
  > $LOCAL/chunk.log 2>&1
cstatus=$?
gzip -c $LOCAL/chunk.log > $OUT/chunk_$ARM.log.gz
grep -a -E "Safety record|trials FAILED|UNSAFE|AUDIT MISS" $LOCAL/chunk.log | tail -4
rm -rf $LOCAL
echo "=== $ARM done python exit=$cstatus end=$(date '+%T') ==="
exit $cstatus
