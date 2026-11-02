#!/bin/bash
#SBATCH --job-name=carv_trials
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=carv_trials_%j.log
#SBATCH --partition=mit_preemptable
#SBATCH --exclusive

# 1. Navigate to the repository root
cd ~/CARV

# 2. Initialize and activate environment
module load miniforge
conda activate carv

# 3. Environment variables (Acados + Threading Limits)
source carv_env.sh
export ACADOS_SOURCE_DIR=/orcd/home/002/ryosei_t/acados
export ACADOS_LIB_DIR=/orcd/home/002/ryosei_t/acados/lib

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

# 4. Launch trials (matching alg20 baseline parameters)
TTTCARV_ALG=alg21_remove_D_req \
TTTCARV_TRIALS=2000 \
TTTCARV_WORKERS=64 \
TTTCARV_SEED=20260824 \
TTTCARV_GOAL_KEEPOUT=0 \
python nfl_robustness_training/src/run_trials_random_obs.py
