#!/bin/bash -l

#SBATCH --clusters=genius
#SBATCH --partition=gpu_v100
#SBATCH --account=intro_vsc37593 
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jierong.wen@student.kuleuven.be

module load CUDA/12.1.1
source ~/.bashrc
conda activate rtg3dd_env_test

#cd point-uv-diffusion-dev
# python3 -m nvitop

# python src/test.py \
# test_batch_size=8 \
# test_samples=100 \
# mode=coarse \
# coarse_ckpt=/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-dev/results/cond_coarse_train_3001627/3001627/static_timestamped \
# #fine_ckpt=/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-dev/results/cond_fine_train_table/demo-Table/static_timestamped

# python src/test.py \
# test_batch_size=16 \
# test_samples=100 \
# mode=joint \
# coarse_ckpt=/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-dev/results/cond_coarse_train_3001627/3001627/static_timestamped \
# fine_ckpt=/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-dev/results/cond_fine_train_3001627/3001627/static_timestamped


python src/test.py \
test_batch_size=16 \
test_samples=100 \
mode=joint \
coarse_ckpt=/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-dev/results/pretrain_3001627_coarse_cond \
fine_ckpt=/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-dev/results/pretrain_3001627_fine_cond



