#!/bin/bash
#SBATCH -J a28j
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -c 10
#SBATCH --mem=40G
#SBATCH -t 01:30:00
#SBATCH --array=1-3
#SBATCH -o /home/ryosei_t/runs/alg28j_batch/job_%A_%a.log
#SBATCH -e /home/ryosei_t/runs/alg28j_batch/job_%A_%a.err
# alg28 lookahead comparison, paired seeds, sound plan boxes + uniform 3 sigma in all arms:
#   1 B      mode B lookahead (decoupled; current default), rerun on the edited source
#   2 J32    mode J, joint one-step CROWN split (1,1,8,1,1,4) = 32 cells
#   3 J256   mode J, split (2,2,16,1,1,4) = 256 cells
#   4 truth  ESTCTRL=0: NN reads the TRUE state (the floor; not implementable)
set -o pipefail
export PATH=/home/ryosei_t/.conda/envs/carv/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
export ACADOS_SOURCE_DIR=/home/ryosei_t/acados ACADOS_LIB_DIR=/home/ryosei_t/acados/lib
export LD_LIBRARY_PATH=/home/ryosei_t/acados/lib:$LD_LIBRARY_PATH
export TTTCARV_WORKERS=8 TTTCARV_TRIALS=${TRIALS:-200} TTTCARV_GOAL_KEEPOUT=0
export TTTCARV_ALG=alg28_sound
export TTTCARV_RECEDING=1 TTTCARV_ONESHOT=1 TTTCARV_ONESHOT_REFINE=2 TTTCARV_ONESHOT_VERIFY=1
export TTTCARV_BOOTSTRAP_RESIM=1 TTTCARV_PSF_LOITER=1 TTTCARV_PSF_TAIL=1 TTTCARV_FIRST_STEP_COST=0
export TTTCARV_TAIL_COST_WEIGHT=1e-3 TTTCARV_RESOLVE_RESERVE=0 TTTCARV_SINGLE_START_FRAC=0.50
export TTTCARV_MPC_PRICING=1 TTTCARV_ONESHOT_RTI=10 TTTCARV_BUDGET_MARGIN=0.003 TTTCARV_GC_PER_TRIAL=1
export TTTCARV_SETMEM=1 TTTCARV_KF_RESET=0 TTTCARV_ESTCTRL=B
export TTTCARV_SEED=20260824
k=$SLURM_ARRAY_TASK_ID
export TTTCARV_VMAX_SIGMA=3.0 TTTCARV_NOISE=uniform TTTCARV_SOUND_MPC=1
case $k in
  1) ARM=B;    export TTTCARV_ESTCTRL=B ;;
  2) ARM=J32;  export TTTCARV_ESTCTRL=J TTTCARV_JSPLIT=1,1,8,1,1,4 ;;
  3) ARM=J256; export TTTCARV_ESTCTRL=J TTTCARV_JSPLIT=2,2,16,1,1,4 ;;
  4) ARM=truth; export TTTCARV_ESTCTRL=0 ;;
esac
OUT=/home/ryosei_t/runs/alg28j_batch
cd /home/ryosei_t/CARV
echo "arm=$ARM node=$(hostname) start=$(date '+%F %T') commit=$(git rev-parse --short HEAD)"
sha256sum -c --quiet $OUT/expected.sha256 || { echo "!! SOURCE DIFFERS -- not running"; exit 3; }
env | grep '^TTTCARV_' | sort | sed 's/^/  /'
LOCAL=${TMPDIR:-/tmp}/carv_a28j_${SLURM_JOB_ID}; rm -rf $LOCAL; mkdir -p $LOCAL/wlog
export TTTCARV_WORKER_LOG_DIR=$LOCAL/wlog
TTTCARV_FINGERPRINT=$OUT/fp_$ARM.json python nfl_robustness_training/src/run_trials_random_obs.py \
  > $LOCAL/chunk.log 2>&1
cstatus=$?
gzip -c $LOCAL/chunk.log > $OUT/chunk_$ARM.log.gz
tar -C $LOCAL -czf $OUT/wlog_$ARM.tar.gz wlog
grep -a -E "Safety record|trials FAILED|UNSAFE" $LOCAL/chunk.log | tail -4
rm -rf $LOCAL
echo "=== $ARM done python exit=$cstatus end=$(date '+%T') ==="
exit $cstatus
