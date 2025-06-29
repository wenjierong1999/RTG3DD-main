#!/bin/bash -l

#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=lp_emediaxr
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jierong.wen@student.kuleuven.be
#SBATCH --job-name=trainer_test

module load CUDA/12.1.1
source ~/.bashrc
conda activate rtg3dd_env_test

# accelerate launch finetune_trainer.py

accelerate launch --num_processes=1 --mixed_precision=no finetune_trainer.py

# accelerate launch finetune_trainer.py