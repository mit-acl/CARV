#!/bin/bash
#SBATCH -J a43_nosat
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH -t 04:00:00
#SBATCH -o /home/ryosei_t/runs/alg43_nosat/arr_%a.log
# alg43: neither side saturates. R1/R2 pin the RAW nn(ctr) at stage 0 (TTTCARV_GP_PIN_RAW=1);
# *_clip arms = rev 7 clipped pin, same batch, as control. Ours = alg41 nx USAT=0 (unchanged, sound).
export CARV_SRC_OVERRIDE=/home/ryosei_t/runs/alg43_nosat/src
exec bash /home/ryosei_t/scripts/carv_alg29_array.sbatch /home/ryosei_t/runs/alg43_nosat 500 20260824 /home/ryosei_t/runs/alg43_nosat/arms.txt
