#!/bin/bash -l

#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --account=lp_emediaxr
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00

#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jierong.wen@student.kuleuven.be
#SBATCH --job-name=paint3d_baseline_sofa

# Activate your environment
module load CUDA/12.1.1
source ~/.bashrc
conda activate kaolin_p100_env 

# conda activate rtg3dd_env_test

#  

  # python paint3d_main_launch.py \
  # --category Chair \
  # --data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
  # --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline_v2 \
  # --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
  # --sd_config_stage1 controlnet/config/depth_based_inpaint_template.yaml \
  # --sd_config_stage2 controlnet/config/UV_based_inpaint_base.yaml \
  # --render_config paint3d/config/train_config_paint3d_base.py \
  # --use_stage_2 \
  # --max_sample_number 100 \
  # --model_id 0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d


  #   python paint3d_main_launch.py \
  # --category Table \
  # --data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
  # --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline_4views_ipplus \
  # --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
  # --sd_config_stage1 controlnet/config/depth_based_inpaint_template.yaml \
  # --sd_config_stage2 controlnet/config/UV_based_inpaint_base.yaml \
  # --render_config paint3d/config/train_config_paint3d_base.py \
  # --use_stage_2 \
  # --max_sample_number 100 \


#   python paint3d_main_launch.py \
# --category Bed \
# --data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
# --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline_4views \
# --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
# --sd_config_stage1 controlnet/config/depth_based_inpaint_template.yaml \
# --sd_config_stage2 controlnet/config/UV_based_inpaint_base.yaml \
# --render_config paint3d/config/train_config_paint3d_base.py \
# --use_stage_2 \
# --max_sample_number 100 \


  python paint3d_main_launch.py \
--category Sofa \
--data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
--output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline_4views \
--split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
--sd_config_stage1 controlnet/config/depth_based_inpaint_template.yaml \
--sd_config_stage2 controlnet/config/UV_based_inpaint_base.yaml \
--render_config paint3d/config/train_config_paint3d_base.py \
--use_stage_2 \
--max_sample_number 100 \

