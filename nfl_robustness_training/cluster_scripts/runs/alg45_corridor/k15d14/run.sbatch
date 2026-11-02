#!/bin/bash
#SBATCH -J a45_d14
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /home/ryosei_t/runs/alg45_corridor/%x_%j.log
# alg45 corridor: ours / ours+NOMINAL_PASS / R2 on one node (arms compared share a node)
bash /home/ryosei_t/scripts/a45_run.sh
exit $?
