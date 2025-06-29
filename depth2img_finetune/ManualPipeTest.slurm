#!/bin/bash -l

#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --account=intro_vsc37593
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jierong.wen@student.kuleuven.be
#SBATCH --job-name=manual_pipe_test

module load CUDA/12.1.1
source ~/.bashrc
conda activate rtg3dd_env_test

# cd Paint3D-main/Depth2Img_finetune

python Depth2ImgManualPipe.py