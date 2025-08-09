import os
import json
import glob
import argparse
import torch
from PIL import Image
from pathlib import Path
import sys
from tqdm import tqdm
import gc
from cleanfid import fid

ROOT_DIR = Path(__file__).resolve().parents[1]  # 即 RTG3DD-main
sys.path.insert(0, str(ROOT_DIR))

from paint3d.models.textured_mesh import TexturedMeshModel
from paint3d.utils import save_tensor_image, color_with_shade, tensor2numpy

def load_render_cfg_from_pyfile(config_path: str):
    """
    Dynamically import a Python file and return TrainConfig instance.
    :param config_path: path to Python file (e.g., paint3d/config/train_config_paint3d.py)
    :return: an instance of TrainConfig
    """
    config_path = Path(config_path)
    module_dir = str(config_path.parent.resolve())
    module_name = config_path.stem  # without ".py"

    sys.path.insert(0, module_dir)  # temporarily add to import path
    module = __import__(module_name)
    sys.path.pop(0)  # clean up sys.path

    if hasattr(module, "TrainConfig"):
        return module.TrainConfig()
    else:
        raise ValueError(f"No TrainConfig found in {config_path}")


def mv_dr(model_id,index,theta,phi,mesh_model,render_cfg,save_result_dir):

    outputs = mesh_model.render(theta=theta,
                                phi=phi,
                                radius=render_cfg.render.radius,
                                dims=(512, 512))
    
    z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
    mask, uncolored_masks = outputs['mask'], outputs['uncolored_mask']
    color_with_shade_img = color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals, light_coef=0.3)
    rgb_render = outputs['image'] * (1 - uncolored_masks) + color_with_shade_img * uncolored_masks

    save_path = Path(save_result_dir) / f"{model_id}_view_{index}.png"
    save_tensor_image(rgb_render, save_path=save_path)


def render_multi_views_gt_views(model_id, 
                                save_root,
                                render_cfg,
                                raw_dataset_root):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    phis = [0, 90, 180, 270]
    thetas = [render_cfg.render.base_theta] * len(phis)

    gt_shape_path = os.path.join(raw_dataset_root, model_id, "normalized_model.obj")
    gt_tex_path = os.path.join(raw_dataset_root, model_id, "texture.png")

    if not os.path.exists(gt_shape_path) or not os.path.exists(gt_tex_path):
        print(f"[WARNING] No GT mesh/texture for {model_id}")
        return

    render_cfg.guide.shape_path = gt_shape_path
    render_cfg.guide.initial_texture = gt_tex_path
    model_gt = TexturedMeshModel(cfg=render_cfg, device=device, flip_texture=True)

    for i, (phi, theta) in enumerate(zip(phis, thetas)):
        try:
            # save_path = Path(save_root) / f"{model_id}_view_{i}.png"
            mv_dr(model_id, i, theta * torch.pi / 180, phi * torch.pi / 180,
                  model_gt, render_cfg, save_result_dir=save_root)
        except Exception as e:
            print(f"[ERROR] Failed rendering GT view {i} of {model_id}: {e}")

    del model_gt
    torch.cuda.empty_cache()
    gc.collect()

def collect_split_ids(split_root = '/data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/future3d-preprocess/final_split_files',
                     max_test_per_category=40):

    categories = ['Chair', 'Table', 'Sofa', 'Bed']
    train_ids, test_ids = [], []

    for cat in categories:
        cat_dir = os.path.join(split_root, cat)
        train_lst_path = os.path.join(cat_dir, 'train.lst')
        test_lst_path = os.path.join(cat_dir, 'test.lst')

        if os.path.exists(train_lst_path):
            with open(train_lst_path, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]
                train_ids.extend(ids)

        if os.path.exists(test_lst_path):
            with open(test_lst_path, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]
                test_ids.extend(ids[:max_test_per_category])
    
    return train_ids, test_ids


def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_ids', nargs='+', required=True, help='List of model IDs to render')
    parser.add_argument('--save_root', type=str, required=True, help='Directory to save rendered views')
    parser.add_argument('--raw_dataset_root', type=str, required=True, help='Root directory of raw GT data')
    parser.add_argument('--config_path', type=str, required=True, help='Path to render config .py file')
    args = parser.parse_args()

    render_cfg = load_render_cfg_from_pyfile(args.config_path)

    train_ids, test_ids = collect_split_ids(max_test_per_category=40)

    for model_id in tqdm(test_ids):
        render_multi_views_gt_views(model_id,
                                    save_root=args.save_root,
                                    render_cfg=render_cfg,
                                    raw_dataset_root=args.raw_dataset_root)


if __name__ == "__main__":

    main()

'''
python mv_render_gtviews.py \
  --save_root /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/combined \
  --raw_dataset_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
  --config_path /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py
'''