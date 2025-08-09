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
#SBATCH --job-name=paint3d_finetuned_chair

# Activate your environment
module load CUDA/12.1.1
source ~/.bashrc
conda activate kaolin_p100_env 

python paint3d_main_launch.py \
  --category Table \
  --data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
  --output_root /scratch/leuven/375/vsc37593/paint3d_res_strict/finetuned_lora_15000_4initviews \
  --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
  --sd_config_stage1 controlnet/config/finetuned_depth_based_cnet.yaml \
  --sd_config_stage2 controlnet/config/UV_based_inpaint_template.yaml \
  --render_config paint3d/config/train_config_paint3d_table.py \
  --use_finetuned_depth2img \
  --use_stage_2 \
  --max_sample_number 100 \