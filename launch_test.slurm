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
#SBATCH --job-name=kaolin_compile

# Activate your environment
module load CUDA/12.1.1
source ~/.bashrc
conda activate kaolin_p100_env  # or your env name

#  

python paint3d_main_launch.py \
  --category Chair \
  --data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
  --output_root /data/leuven/375/vsc37593/my_py_projects/Paint3D-main/test_outputs \
  --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
  --sd_config_stage1 controlnet/config/finetuned_depth_based_cnet.yaml \
  --sd_config_stage2 controlnet/config/UV_based_inpaint_template.yaml \
  --render_config paint3d/config/train_config_paint3d.py \
  --use_finetuned_depth2img \
  --use_stage_2 \
  --max_sample_number 3 \
  # --dry_run