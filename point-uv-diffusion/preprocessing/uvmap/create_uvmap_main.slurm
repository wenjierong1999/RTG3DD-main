#!/bin/bash -l

#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --account=intro_vsc37593
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00



module load CUDA/12.1.1
source ~/.bashrc
cd preprocessing/uvmap
conda activate rtg3dd_env_test

python create_uvmap_main.py \
  --model_filename raw_model.obj \
  --view_num 24 \
  --render_res 512 \
  --uv_resolution 512 \
  --save_folder /scratch/leuven/375/vsc37593/3D-FUTURE-Preprocessed-24Views \
  --dataset_folder /scratch/leuven/375/vsc37593/3D-FUTURE-model/3D-FUTURE-model \
  --flag_csv_path /data/leuven/375/vsc37593/my_py_projects/Future3d-24views-flag.csv \
  --category_list Table


python create_uvmap_shapenet.py \
  --model_filename model_normalized.obj \
  --view_num 24 \
  --render_res 512 \
  --uv_resolution 512 \
  --save_folder /data/leuven/375/vsc37593/my_py_projects/test_preprocess_output_shapenet \
  --dataset_folder /data/leuven/375/vsc37593/my_py_projects/ShapeNet_toy \
  --flag_csv_path None \
  --category_list '02828884'
