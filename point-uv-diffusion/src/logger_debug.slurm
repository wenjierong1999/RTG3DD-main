#!/bin/bash -l

#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100_debug
#SBATCH --account=intro_vsc37593
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=2
#SBATCH --time=00:10:00



module load CUDA/12.1.1
source ~/.bashrc
cd src
conda activate rtg3dd_env_test

torchrun --nproc_per_node=2  logger_debug.py