#!/bin/bash -l

#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100_debug
#SBATCH --account=intro_vsc37593
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00



module load CUDA/12.1.1
source ~/.bashrc
cd preprocessing/render
conda activate rtg3dd_env_test

# python run_render_cmd_test.py --dataset_folder '/data/leuven/375/vsc37593/my_py_projects/ShapeNet_toy/02828884' \
# --view_num 24 --model_filename 'model_normalized.obj' --resolution 512

python run_render_cmd_shapenet.py --dataset_folder '/data/leuven/375/vsc37593/my_py_projects/ShapeNet_toy' \
--synset '02828884' \
--view_num 24 --model_filename 'model_normalized.obj' --resolution 512
