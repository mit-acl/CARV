#!/bin/bash
#SBATCH -J a41_base
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH -t 04:00:00
# alg41 scoring check, baselines: one fixed-delay arm per array task.
# alg29_generic_psf comes from alg41_score/src (scoring-only patch: clip / pred references).
export CARV_SRC_OVERRIDE=/home/ryosei_t/runs/alg41_score/src
exec bash /home/ryosei_t/scripts/carv_alg29_array.sbatch /home/ryosei_t/runs/alg41_score 500 20260824 /home/ryosei_t/runs/alg41_score/arms_base.txt
