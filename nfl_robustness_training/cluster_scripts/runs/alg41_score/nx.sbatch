#!/bin/bash
#SBATCH -J a41_ours_nx
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH -t 04:00:00
#SBATCH -o /home/ryosei_t/runs/alg41_score/nx/nx.log
# NON-EXCLUSIVE duplicate of excl.sbatch (keep whichever starts first), 16 CPUs for 8 workers.
# USAT=1 (NN saturated at +-1, as the baselines' pin) vs USAT=0, same node.
export CARV_SRC_OVERRIDE=/home/ryosei_t/runs/alg41_score/src
TTTCARV_WORKERS=8 /home/ryosei_t/scripts/carv_alg29_arms.sh /home/ryosei_t/runs/alg41_score/nx 500 20260824 /home/ryosei_t/runs/alg41_score/arms_ours.txt
exit $?
