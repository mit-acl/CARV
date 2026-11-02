#!/bin/bash
#SBATCH -J a29
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
# One alg29 arm per array task: line $SLURM_ARRAY_TASK_ID of $ARMS.
# Only for arms whose result does not depend on node speed (fixed-delay
# generic PSF). Budget-priced arms (alg28) and GP_DELAY=measured must run
# sequentially on one node via carv_alg29_arms.sh.
# usage: sbatch -a 1-N -o OUT/arr_%a.log carv_alg29_array.sbatch OUT TRIALS SEED ARMFILE
OUT=$1; TRIALS=$2; SEED=$3; ARMS=$4
LINE=$(grep -v '^\s*#' $ARMS | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
TMPA=$(mktemp); echo "$LINE" > $TMPA
TTTCARV_WORKERS=8 /home/ryosei_t/scripts/carv_alg29_arms.sh $OUT $TRIALS $SEED $TMPA
rc=$?
rm -f $TMPA
exit $rc
