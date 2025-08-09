#!/bin/bash -l

#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --account=lp_emediaxr
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00

#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jierong.wen@student.kuleuven.be
#SBATCH --job-name=gen_gt_views

# Activate your environment
module load CUDA/12.1.1
source ~/.bashrc
conda activate kaolin_p100_env 


python multiview_render.py \
    --category Sofa \
    --raw_data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
    --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
    --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
    --output_root /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8 \
    --num_views 8 \