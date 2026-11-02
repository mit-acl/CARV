#!/bin/bash
#SBATCH -J a42_ours
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH -t 04:00:00
#SBATCH -o /home/ryosei_t/runs/alg42_modemetric/ours.log
# Ours is wall-clock priced, so both arms run sequentially on ONE allocation.
# 16 CPUs for 8 workers (one worker per physical core).
export CARV_SRC_OVERRIDE=/home/ryosei_t/runs/alg42_modemetric/src
TTTCARV_WORKERS=8 /home/ryosei_t/scripts/carv_alg29_arms.sh /home/ryosei_t/runs/alg42_modemetric/ours 500 20260824 /home/ryosei_t/runs/alg42_modemetric/arms_ours.txt
exit $?
