#!/bin/bash
#SBATCH -J a28_10kra
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH -t 12:00:00
#SBATCH -a 1-4
#SBATCH -o /home/ryosei_t/runs/alg28_10k_ra/chunk%a.log
#SBATCH -e /home/ryosei_t/runs/alg28_10k_ra/chunk%a.err
# alg28 headline 10k run WITH THE WALL RE-ANCHOR (TTTCARV_REANCHOR).
#
# Each array task runs BOTH arms back to back on its own exclusive node, 2500
# trials each, same seed. That pairs the arms per node: activations are priced
# against wall-clock, so arms on different nodes are not comparable (this is what
# spoiled the first split batch, job 23300811).
#
# SCENES ARE THE alg20/alg22 SET. seed = 20260823 + chunk with GOAL_KEEPOUT=0
# regenerates the same 10,000 scenes as results_alg20 / results_alg22, so this
# is a PAIRED comparison against the alg22 headline run, not a new sample.
#
# Configuration under test (alg28 = alg27 + sound plan boxes + audits):
#   SETMEM=1      certified box = propagated INTERSECT measurement box
#   KF_RESET=0    estimator is never re-anchored at the true state
#   ESTCTRL=J     plant applies NN(centre of certified box); lookahead is the
#                 joint one-step CROWN over (x,e), JSPLIT 2,2,16,1,1,4 = 256 cells
#   SOUND_MPC=1   backup-plan boxes by interval propagation + per-stage fan margin
#   NOISE=uniform |v| <= 3 sigma, a hard bound (the containment claim)
#
# LOW LOAD IS PART OF THE CONFIGURATION: 8 workers on an exclusive node, as in
# the alg22 headline run. The 0.20 s timestep budget only means what it claims
# if each trial gets the compute it was calibrated against.
set -o pipefail
export PATH=/home/ryosei_t/.conda/envs/carv/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
export ACADOS_SOURCE_DIR=/home/ryosei_t/acados ACADOS_LIB_DIR=/home/ryosei_t/acados/lib
export LD_LIBRARY_PATH=/home/ryosei_t/acados/lib:$LD_LIBRARY_PATH
cd /home/ryosei_t/CARV

OUT=/home/ryosei_t/runs/alg28_10k_ra
sha256sum -c --quiet $OUT/expected.sha256 || { echo "!! SOURCE DIFFERS -- not running"; exit 3; }

export TTTCARV_WORKERS=8 TTTCARV_TRIALS=2500 TTTCARV_GOAL_KEEPOUT=0
export TTTCARV_SEED=$((20260823 + SLURM_ARRAY_TASK_ID))
export TTTCARV_ALG=alg28_sound
# alg22 final configuration, carried over unchanged
export TTTCARV_RECEDING=1 TTTCARV_ONESHOT=1 TTTCARV_ONESHOT_VERIFY=1 TTTCARV_ONESHOT_RTI=10
export TTTCARV_ONESHOT_REFINE=2 TTTCARV_BOOTSTRAP_RESIM=1 TTTCARV_PSF_LOITER=1 TTTCARV_PSF_TAIL=1
export TTTCARV_FIRST_STEP_COST=0 TTTCARV_TAIL_COST_WEIGHT=1e-3 TTTCARV_RESOLVE_RESERVE=0
export TTTCARV_SINGLE_START_FRAC=0.50 TTTCARV_MPC_PRICING=1 TTTCARV_BUDGET_MARGIN=0.003
export TTTCARV_GC_PER_TRIAL=1
# alg28 state-uncertainty configuration
export TTTCARV_SETMEM=1 TTTCARV_KF_RESET=0 TTTCARV_SOUND_MPC=1
export TTTCARV_ESTCTRL=J TTTCARV_JSPLIT=2,2,16,1,1,4
export TTTCARV_VMAX_SIGMA=3.0 TTTCARV_NOISE=uniform
export TTTCARV_PLAN_CELLS=1

echo "chunk=$SLURM_ARRAY_TASK_ID seed=$TTTCARV_SEED node=$(hostname) workers=$TTTCARV_WORKERS"
echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)  cpus=$(nproc)"
echo "commit=$(git rev-parse --short HEAD)  dirty=$(git status --porcelain | wc -l) files"
env | grep '^TTTCARV_' | sort | sed 's/^/  /'

cstatus=0
for arm in ra0 ra1; do
  # --requeue restarts this script from the top; skip arms already finished.
  if [ -s $OUT/fp_${arm}_$SLURM_ARRAY_TASK_ID.json ]; then
    echo "=== arm=$arm chunk=$SLURM_ARRAY_TASK_ID already done -- skipping ==="
    continue
  fi
  case $arm in
    ra0) export TTTCARV_REANCHOR=0 ;;
    ra1) export TTTCARV_REANCHOR=1 ;;
  esac
  echo "=== arm=$arm reanchor=$TTTCARV_REANCHOR chunk=$SLURM_ARRAY_TASK_ID start=$(date '+%T') ==="
  LOCAL=${TMPDIR:-/tmp}/carv_a28ra_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$arm
  rm -rf $LOCAL; mkdir -p $LOCAL/wlog
  export TTTCARV_WORKER_LOG_DIR=$LOCAL/wlog

  TTTCARV_FINGERPRINT=$OUT/fp_${arm}_$SLURM_ARRAY_TASK_ID.json \
  python nfl_robustness_training/src/run_trials_random_obs.py > $LOCAL/chunk.log 2>&1
  a=$?; [ $a -ne 0 ] && cstatus=$a

  gzip -c $LOCAL/chunk.log > $OUT/chunk_${arm}_$SLURM_ARRAY_TASK_ID.log.gz
  tar -C $LOCAL -czf $OUT/wlog_${arm}_$SLURM_ARRAY_TASK_ID.tar.gz wlog
  grep -a -E "Safety record|trials FAILED|UNSAFE|AUDIT MISS|truth_out" $LOCAL/chunk.log | tail -8
  rm -rf $LOCAL
  echo "=== arm=$arm chunk=$SLURM_ARRAY_TASK_ID exit=$a end=$(date '+%F %T') ==="
done
echo "=== chunk $SLURM_ARRAY_TASK_ID ALL DONE rc=$cstatus $(date '+%F %T') ==="
exit $cstatus
