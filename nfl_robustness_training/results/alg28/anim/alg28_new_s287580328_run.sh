#!/bin/bash
# Newest "ours" config (alg35_thr arm u_ours_ideal), scenario of reanchor_ruleS_s287580328.
set -o pipefail
export PATH=/home/ryosei_t/.conda/envs/carv/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
export ACADOS_SOURCE_DIR=/home/ryosei_t/acados ACADOS_LIB_DIR=/home/ryosei_t/acados/lib
export LD_LIBRARY_PATH=/home/ryosei_t/acados/lib:${LD_LIBRARY_PATH:-}
export TTTCARV_RECEDING=1 TTTCARV_ONESHOT=1 TTTCARV_ONESHOT_VERIFY=1 TTTCARV_ONESHOT_RTI=10 TTTCARV_ONESHOT_REFINE=2
export TTTCARV_BOOTSTRAP_RESIM=1 TTTCARV_PSF_LOITER=1 TTTCARV_PSF_TAIL=1 TTTCARV_FIRST_STEP_COST=0
export TTTCARV_TAIL_COST_WEIGHT=1e-3 TTTCARV_RESOLVE_RESERVE=0 TTTCARV_SINGLE_START_FRAC=0.50
export TTTCARV_MPC_PRICING=1 TTTCARV_BUDGET_MARGIN=0.003 TTTCARV_GC_PER_TRIAL=1
export TTTCARV_SETMEM=1 TTTCARV_KF_RESET=0 TTTCARV_SOUND_MPC=1 TTTCARV_ESTCTRL=J TTTCARV_JSPLIT=2,2,16,1,1,4
export TTTCARV_VMAX_SIGMA=3.0 TTTCARV_NOISE=uniform TTTCARV_EBOX=S TTTCARV_REANCHOR=1 TTTCARV_MEAS_DELAY=0
export TTTCARV_SINGLE_UBOUND=reject TTTCARV_TERMINAL=loiter
SEED=287580328
OBS="-3.63,0.57,0.5 -2.71,1.41,0.5 -1.40,1.04,0.5"
O=/home/ryosei_t/runs/alg28_anim3
cd /home/ryosei_t/CARV
echo "node=$(hostname) seed=$SEED config=alg35_thr:u_ours_ideal" > $O/provenance.txt
sha256sum nfl_robustness_training/src/alg28_sound.py >> $O/provenance.txt
TTTCARV_RSOA_DUMP=$O/dump_new.json python nfl_robustness_training/src/alg28_sound.py $SEED $OBS > $O/new.log 2>&1
rc=$?; echo "exit=$rc"; exit $rc
