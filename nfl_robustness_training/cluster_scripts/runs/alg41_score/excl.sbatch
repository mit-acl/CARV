#!/bin/bash
#SBATCH -J a41_ours
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH -t 04:00:00
#SBATCH -o /home/ryosei_t/runs/alg41_score/excl.log
# alg41 scoring check, ours (wall-clock priced): sequential on ONE exclusive node,
# USAT=1 (NN saturated at +-1, as the baselines' pin) vs USAT=0, same node.
export CARV_SRC_OVERRIDE=/home/ryosei_t/runs/alg41_score/src
TTTCARV_WORKERS=8 /home/ryosei_t/scripts/carv_alg29_arms.sh /home/ryosei_t/runs/alg41_score 500 20260824 /home/ryosei_t/runs/alg41_score/arms_ours.txt
exit $?
