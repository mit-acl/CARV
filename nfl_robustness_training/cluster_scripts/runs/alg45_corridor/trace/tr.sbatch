#!/bin/bash
#SBATCH -J a45_tr
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /home/ryosei_t/runs/alg45_corridor/%x_%j.log
# alg45 corridor: ours / ours+NOMINAL_PASS / R2 on one node (arms compared share a node)
A45_OUT=trace A45_N=8 A45_ARMS=trace/arms.txt bash /home/ryosei_t/scripts/a45_run.sh
exit $?
