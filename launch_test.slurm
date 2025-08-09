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

# # conda activate rtg3dd_env_test



python paint3d_main_launch.py \
  --category Chair \
  --data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
  --output_root demo_output \
  --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
  --sd_config_stage1 controlnet/config/finetuned_depth_based_cnet.yaml \
  --sd_config_stage2 controlnet/config/UV_based_inpaint_template.yaml \
  --render_config paint3d/config/train_config_paint3d.py \
  --use_finetuned_depth2img \
  --model_id 0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d \
  --use_stage_2 \

  python paint3d_main_launch.py \
  --category Table \
  --data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
  --output_root demo_output \
  --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
  --sd_config_stage1 controlnet/config/finetuned_depth_based_cnet.yaml \
  --sd_config_stage2 controlnet/config/UV_based_inpaint_template.yaml \
  --render_config paint3d/config/train_config_paint3d_table.py \
  --use_finetuned_depth2img \
  --model_id 8b64cd69-2f4d-4b26-98e1-96dbd1e370fb\
  --use_stage_2 \

# finetuned

  python paint3d_main_launch.py \
  --category Sofa \
  --data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
  --output_root demo_output \
  --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
  --sd_config_stage1 controlnet/config/finetuned_depth_based_cnet.yaml \
  --sd_config_stage2 controlnet/config/UV_based_inpaint_template.yaml \
  --render_config paint3d/config/train_config_paint3d_sofa.py \
  --use_finetuned_depth2img \
  --model_id 0034935b-20a0-45e6-b2a5-9bff89a94198\
  --use_stage_2 \

# baseline bed

  python paint3d_main_launch.py \
  --category Bed \
  --data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
  --output_root demo_output_bl \
  --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
  --sd_config_stage1 controlnet/config/depth_based_inpaint_template.yaml \
  --sd_config_stage2 controlnet/config/UV_based_inpaint_template.yaml \
  --render_config paint3d/config/train_config_paint3d_sofa.py \
  --model_id 003eea79-6ec0-4731-b5fb-acc09dc59898\
  --use_stage_2 \


# finetuned bed
  python paint3d_main_launch.py \
  --category Bed \
  --data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
  --output_root demo_output \
  --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
  --sd_config_stage1 controlnet/config/finetuned_depth_based_cnet.yaml \
  --sd_config_stage2 controlnet/config/UV_based_inpaint_template.yaml \
  --render_config paint3d/config/train_config_paint3d.py \
  --use_finetuned_depth2img \
  --model_id 003eea79-6ec0-4731-b5fb-acc09dc59898\
  --use_stage_2 \



python paint3d_main_launch.py \
  --category Chair \
  --data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
  --output_root demo_output \
  --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
  --sd_config_stage1 controlnet/config/depth_based_inpaint_template.yaml \
  --sd_config_stage2 controlnet/config/UV_based_inpaint_template.yaml \
  --render_config paint3d/config/train_config_paint3d.py \
  --model_id 0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d \
  --use_stage_2 \


  python paint3d_main_launch.py \
  --category Table \
  --data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
  --output_root /scratch/leuven/375/vsc37593/paint3d_res/finetuned_lora_13000_2initviews_65 \
  --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
  --sd_config_stage1 controlnet/config/finetuned_depth_based_cnet.yaml \
  --sd_config_stage2 controlnet/config/UV_based_inpaint_template.yaml \
  --render_config paint3d/config/train_config_paint3d.py \
  --use_finetuned_depth2img \
  --use_stage_2 \
  --max_sample_number 100 \
  # --model_id 0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d

#DEBUG
  # python paint3d_main_launch.py \
  # --category Chair \
  # --data_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
  # --output_root test_outputs \
  # --split_file_root /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files \
  # --sd_config_stage1 controlnet/config/finetuned_depth_based_cnet.yaml \
  # --sd_config_stage2 controlnet/config/UV_based_inpaint_template.yaml \
  # --render_config paint3d/config/train_config_paint3d.py \
  # --use_finetuned_depth2img \
  # --use_stage_2 \
  # --max_sample_number 1 \
  # --model_id 0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d


#   python paint3d_launch_stage1.py \
#  --sd_config controlnet/config/depth_based_inpaint_template.yaml \
#  --render_config paint3d/config/train_config_paint3d_base.py \
#  --mesh_path /data/leuven/375/vsc37593/my_py_projects/Paint3D-main/demo/objs/0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d/normalized_model.obj \
#  --prompt " " \
#  --ip_adapter_image_path /data/leuven/375/vsc37593/my_py_projects/Paint3D-main/demo/objs/0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d/image.jpg \
#  --outdir test_outputs/baseline/0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d/img_stage1 \


#  python paint3d_launch_stage2.py \
# --sd_config controlnet/config/UV_based_inpaint_template.yaml \
# --render_config paint3d/config/train_config_paint3d.py \
# --mesh_path /data/leuven/375/vsc37593/my_py_projects/Paint3D-main/demo/objs/0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d/normalized_model.obj \
# --texture_path test_outputs/0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d/img_stage1/res-0/albedo.png \
# --prompt " " \
# --ip_adapter_image_path /data/leuven/375/vsc37593/my_py_projects/Paint3D-main/demo/objs/0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d/image.jpg \
# --outdir test_outputs/0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d/img_stage2