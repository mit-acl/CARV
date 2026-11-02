#!/bin/bash
#SBATCH -J a42_smoke
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 00:40:00
#SBATCH -o /home/ryosei_t/runs/alg42_modemetric/smoke.log
exec bash /home/ryosei_t/scripts/carv_modemetric_smoke.sh /home/ryosei_t/runs/alg42_modemetric/smoke
