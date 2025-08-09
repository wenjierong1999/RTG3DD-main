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
conda activate rtg3dd_env_test

torchrun --nproc_per_node=1  src/train_debug.py \
experiment=cond_coarse_demo \
datamodule.batch_size=6 \
trainer.max_epochs=3 \
config_load=True \