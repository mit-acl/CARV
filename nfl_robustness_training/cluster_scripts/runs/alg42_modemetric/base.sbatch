#!/bin/bash
#SBATCH -J a42_base
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH -t 04:00:00
# alg42 mode-based intervention metric, baselines: one fixed-delay arm per array
# task. alg29_generic_psf comes from alg42_modemetric/src (bookkeeping-only
# patch: eng_steps / nn_over / u_diff_clip_gt).
export CARV_SRC_OVERRIDE=/home/ryosei_t/runs/alg42_modemetric/src
exec bash /home/ryosei_t/scripts/carv_alg29_array.sbatch /home/ryosei_t/runs/alg42_modemetric 500 20260824 /home/ryosei_t/runs/alg42_modemetric/arms_base.txt
