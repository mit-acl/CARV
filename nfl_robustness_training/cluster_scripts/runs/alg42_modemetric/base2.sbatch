#!/bin/bash
#SBATCH -J a42_base2
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH -t 04:00:00
# alg42 rerun after the NOISE='none' box-propagation fix in alg29_generic_psf.
# m2_r2_0.1_none is the exact-state R2 arm; m2_r2_0.1_unif is the control that
# must still reproduce 12.11 (the fix touches only the NOISE='none' path);
# m2_g2_0.1_none is the reference the fixed R2 arm should land near.
export CARV_SRC_OVERRIDE=/home/ryosei_t/runs/alg42_modemetric/src
exec bash /home/ryosei_t/scripts/carv_alg29_array.sbatch /home/ryosei_t/runs/alg42_modemetric 500 20260824 /home/ryosei_t/runs/alg42_modemetric/arms_base2.txt
