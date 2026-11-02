#!/bin/bash
# Run alg29 arms SEQUENTIALLY on one allocation (arms compared must share a node).
# usage: carv_alg29_arms.sh OUTDIR TRIALS SEED ARMFILE
# ARMFILE: one arm per line: NAME VAR=VAL VAR=VAL ...
set -o pipefail
OUT=$1; TRIALS=$2; SEED=$3; ARMS=$4
export PATH=/home/ryosei_t/.conda/envs/carv/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
export ACADOS_SOURCE_DIR=/home/ryosei_t/acados ACADOS_LIB_DIR=/home/ryosei_t/acados/lib
export LD_LIBRARY_PATH=/home/ryosei_t/acados/lib:$LD_LIBRARY_PATH
cd /home/ryosei_t/CARV
mkdir -p $OUT
# Pin sources: snapshot the algorithm modules into $OUT/snap under *_snap names
# (the runner's own dir precedes PYTHONPATH, so the names must differ) and
# rewrite TTTCARV_ALG to them. Later edits in the checkout cannot leak in.
SNAP=$OUT/snap_$$; mkdir -p $SNAP
# CARV_SRC_OVERRIDE=DIR: take a module from DIR when DIR/$m.py exists (for arms
# that must not touch a checkout file another job has pinned by hash).
for m in alg29_generic_psf alg28_sound; do  # alg45 copy: SCENE_RUNNER override
  src=nfl_robustness_training/src/$m.py
  [ -n "$CARV_SRC_OVERRIDE" ] && [ -f "$CARV_SRC_OVERRIDE/$m.py" ] && src=$CARV_SRC_OVERRIDE/$m.py
  echo "source $m <- $src"
  cp $src $SNAP/${m}_snap.py
done
( cd $SNAP && sha256sum *_snap.py; sha256sum /home/ryosei_t/CARV/nfl_robustness_training/src/mpc_safety_filter_acados.py ) > $OUT/source_$$.sha256
export PYTHONPATH=$SNAP:/home/ryosei_t/CARV/nfl_robustness_training/src:$PYTHONPATH
NW=${TTTCARV_WORKERS:-$(( $(nproc) ))}
echo "node=$(hostname) workers=$NW trials=$TRIALS seed=$SEED commit=$(git rev-parse --short HEAD)"
worst=0
while read -r NAME REST; do
  [ -z "$NAME" ] && continue; [[ "$NAME" == \#* ]] && continue
  echo "=== arm $NAME  $REST  start=$(date '+%T')"
  RUNNER=run_trials_random_obs.py
  [[ "$REST" == *TTTCARV_GP_SYSTEM=di* || "$REST" == *TTTCARV_ALG=alg22_di* ]] && RUNNER=run_trials_di.py
  REST=$(echo "$REST" | sed -E 's/TTTCARV_ALG=(alg29_generic_psf|alg28_sound)\b/TTTCARV_ALG=\1_snap/')
  env $REST TTTCARV_WORKERS=$NW TTTCARV_TRIALS=$TRIALS TTTCARV_SEED=$SEED \
      TTTCARV_GOAL_KEEPOUT=0 TTTCARV_FINGERPRINT=$OUT/fp_$NAME.json \
      python ${SCENE_RUNNER:-nfl_robustness_training/src/$RUNNER} > $OUT/log_$NAME.txt 2>&1
  rc=$?
  [ $rc -ne 0 ] && worst=$rc
  grep -a -E "Safety record|trials FAILED" $OUT/log_$NAME.txt | tail -2
  gzip -f $OUT/log_$NAME.txt
  echo "=== arm $NAME rc=$rc end=$(date '+%T')"
done < $ARMS
exit $worst
