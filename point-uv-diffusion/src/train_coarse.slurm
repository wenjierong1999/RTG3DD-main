#!/bin/bash -l

#SBATCH --clusters=geinus
#SBATCH --partition=gpu_v100
#SBATCH --account=intro_vsc37593
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00

#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jierong.wen@student.kuleuven.be
#SBATCH --job-name=cond_fine_train

module load CUDA/12.1.1
source ~/.bashrc
conda activate rtg3dd_env_test

# torchrun --nproc_per_node=2  src/train.py \
# experiment=cond_coarse_train_table \
# datamodule.batch_size=8 \
# trainer.max_epochs=500 \
# ckpt_resume=/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-dev/results/cond_coarse_train_table/demo-Table/static_timestamped \

# torchrun --nproc_per_node=1  src/train.py \
# experiment=cond_coarse_train_table \
# datamodule.batch_size=4 \
# trainer.max_epochs=1 \
# config_load=False \


# torchrun --nproc_per_node=2  src/train.py \
# experiment=cond_coarse_train_3001627 \
# datamodule.batch_size=40 \
# trainer.max_epochs=500 \
# config_load=False \

torchrun --nproc_per_node=1  src/train.py \
experiment=cond_fine_train_3001627 \
datamodule.batch_size=8 \
trainer.max_epochs=1 \
config_load=False \