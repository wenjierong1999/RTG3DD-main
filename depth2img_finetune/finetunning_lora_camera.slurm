#!/bin/bash -l

#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_emediaxr
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jierong.wen@student.kuleuven.be
#SBATCH --job-name=finetuning_lora_with_camera

module load CUDA/12.1.1
source ~/.bashrc
conda activate rtg3dd_env_test

PORT=$((29500 + RANDOM % 1000))
echo "[INFO] Using port $PORT for torch.distributed"

accelerate launch --num_processes=1 --mixed_precision=no --main_process_port $PORT finetune_trainer.py \
  --output_dir /scratch/leuven/375/vsc37593/finetune_expr_v2/lora_with_camera \
  --fid_gt_dir /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/combined \
  --num_epochs 5 \
  --use_customized_unet \
  --use_lora \
  --use_camera_label \
  --use_directional_prompts \
  --lr 5e-7 \
  --max_steps 20000 \
  --eval_interval 1000 \
  --log_interval 20 \