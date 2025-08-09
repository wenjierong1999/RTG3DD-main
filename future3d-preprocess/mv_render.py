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

CHAIR_OUTLIERS = [
    '1b1c2e2b-8d32-4768-8af3-0e7b82a148b7',
    '0e1517db-dfa3-485f-9ccb-45a2088b011a',
    '2c918c67-03a6-4451-94a0-d643d1abd95c',
    '8b89c0bf-0bcc-4cc7-adb5-c6fa4fcdf357',
    '45e38c28-5c62-427b-9171-d63ad55de8f9',
    '4838b05b-5f74-4e24-9cbd-3643042a4bd0',
    '8396f9e8-020d-3f12-aed8-f3c03f1afc25',
    '127406da-6400-4858-ba37-bb0ce96f773b',
    '9af69a41-21da-4d4d-b711-eeed465cf70a',
    '8396f9e8-020d-3f12-aed8-f3c03f1afc25'
]

TABLE_OUTLIERS = [
    '3fd0835e-40bd-3a0b-95b3-ffb0b9406952',
]

def parse_args():

    parser = argparse.ArgumentParser(description="Render multiview images for Paint3D models.")
    parser.add_argument("--category", type=str, required=True, help="Object category, e.g., Chair.")
    parser.add_argument("--output_root", type=str, required=True, help="Root folder containing output model folders.")
    parser.add_argument("--raw_data_folder", type=str, required=True,
                    help="Path to raw dataset (used to render GT views from mesh).")
    parser.add_argument("--render_config", type=str, required=True, help="Path to render config .py file.")
    parser.add_argument("--num_views", type=int, default=8, choices=[4, 8], help="Number of views to render per model.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max number of models to render (use -1 for all).")
    parser.add_argument(
        "--render_views",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
        help="Whether to render multiple views before computing FID/KID."
    )
    parser.add_argument(
        "--exclude_outliers",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="Whether to exclude known outlier models (e.g., CHAIR_OUTLIERS)."
    )
    return parser.parse_args()

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

def render_multi_views_from_output(model_id, model_folder, category_views_root,
                                   render_cfg, num_views=8, raw_dataset_root=None, gt_cache_root=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    if num_views == 4:
        phis = [0, 90, 180, 270]
    elif num_views == 8:
        phis = list(range(0, 360, 45))
    else:
        raise ValueError(f"Unsupported num_views={num_views}")

    thetas = [render_cfg.render.base_theta] * len(phis)
    os.makedirs(category_views_root, exist_ok=True)

    # === Predicted mesh ===
    shape_path = model_folder / "mesh.obj"
    tex_path = model_folder / "albedo.png"
    if not shape_path.exists() or not tex_path.exists():
        print(f"[WARNING] Skipping {model_id}: missing mesh or texture.")
        return

    render_cfg.guide.shape_path = str(shape_path)
    render_cfg.guide.initial_texture = str(tex_path)
    model = TexturedMeshModel(cfg=render_cfg, device=device, flip_texture=True)

    for i, (phi, theta) in enumerate(zip(phis, thetas)):
        try:
            mv_dr(model_id,i, theta * torch.pi / 180, phi * torch.pi / 180,
                  model, render_cfg, category_views_root)
        except Exception as e:
            print(f"[ERROR] Failed rendering pred view {i} of {model_id}: {e}")
    del model
    torch.cuda.empty_cache()

    # === GT mesh ===
    if raw_dataset_root is not None and gt_cache_root is not None:
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
                mv_dr(model_id,i, theta * torch.pi / 180, phi * torch.pi / 180,
                      model_gt, render_cfg, gt_cache_root)
            except Exception as e:
                print(f"[ERROR] Failed rendering GT view {i} of {model_id}: {e}")
        del model_gt
        torch.cuda.empty_cache()

    # torch.cuda.ipc_collect()
    # torch.cuda.synchronize()
    gc.collect()

def main():
    args = parse_args()
    render_cfg = load_render_cfg_from_pyfile(args.render_config)

    category = args.category
    pred_root = Path(args.output_root) / category
    # gt_root = Path(args.gt_folder) / f"{category}_views"
    pred_views_root = Path(args.output_root) / f"{category}_views"
    gt_temp_root = Path(args.output_root) / f"{category}_gt_cache"
    

    # breakpoint()

    pred_views_root.mkdir(parents=True, exist_ok=True)
    gt_temp_root.mkdir(parents=True, exist_ok=True)

    all_model_ids = sorted(os.listdir(pred_root))
    model_ids = [mid for mid in all_model_ids if (Path(pred_root) / mid / "mesh.obj").exists()]

    print(len(model_ids))

    if args.exclude_outliers:
        if category.lower() == "chair":
            model_ids = [m for m in model_ids if m not in CHAIR_OUTLIERS]
        elif category.lower() == "table":
            model_ids = [m for m in model_ids if m not in TABLE_OUTLIERS]
    
    if args.max_samples > 0:
        model_ids = model_ids[:args.max_samples]
    
    print(f"[INFO] Processing {len(model_ids)} models in category: {category}")
    
    # breakpoint()
    if args.render_views:
        for model_id in tqdm(model_ids, desc="Rendering"):
            model_folder = pred_root / model_id
            render_multi_views_from_output(model_id, model_folder,
                                           pred_views_root, render_cfg,
                                           num_views=args.num_views,
                                           raw_dataset_root=args.raw_data_folder,
                                           gt_cache_root=gt_temp_root)
            # breakpoint()

    print("[INFO] Calculating FID/KID...")
    fid_score = fid.compute_fid(str(gt_temp_root), str(pred_views_root), mode="clean")
    kid_score = fid.compute_kid(str(gt_temp_root), str(pred_views_root), mode="clean")
    print(f"\n[RESULT] FID: {fid_score}")
    print(f"[RESULT] KID: {kid_score}")

    # shutil.rmtree(gt_temp_root, ignore_errors=True)
    # print(f"[INFO] Deleted temporary GT cache: {gt_temp_root}")


if __name__ == "__main__":
    main()

    '''
    python mv_render.py \
    --category Chair \
    --output_root /scratch/leuven/375/vsc37593/paint3d_res_strict/finetuned_lora_13000_4initviews \
    --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
    --raw_data_folder /scratch/leuven/375/vsc37593/3D-FUTURE-model \
    --num_views 8 \
    --max_samples -1 \
    --render_views True \
    --exclude_outliers True
    '''
    '''
    python mv_render.py \
    --category Chair \
    --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline_4views \
    --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
    --raw_data_folder /scratch/leuven/375/vsc37593/3D-FUTURE-model \
    --num_views 4 \
    --max_samples -1 \
    --render_views True \
    --exclude_outliers True
    '''

    '''
    python mv_render.py \
    --category Chair \
    --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline \
    --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
    --raw_data_folder /scratch/leuven/375/vsc37593/3D-FUTURE-model \
    --num_views 8 \
    --max_samples -1 \
    --render_views True \
    --exclude_outliers True
    '''

    '''
    python mv_render.py \
    --category Table \
    --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline_4views \
    --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
    --raw_data_folder /scratch/leuven/375/vsc37593/3D-FUTURE-model \
    --num_views 8 \
    --max_samples -1 \
    --render_views True \
    --exclude_outliers True
    '''

    '''
    python mv_render.py \
    --category Bed \
    --output_root /scratch/leuven/375/vsc37593/paint3d_res_strict/finetuned_lora_13000_4initviews \
    --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
    --raw_data_folder /scratch/leuven/375/vsc37593/3D-FUTURE-model \
    --num_views 8 \
    --max_samples 90 \
    --render_views True \
    --exclude_outliers True
    '''

    '''
    python mv_render.py \
    --category Bed \
    --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline_4views \
    --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
    --raw_data_folder /scratch/leuven/375/vsc37593/3D-FUTURE-model \
    --num_views 8 \
    --max_samples 80 \
    --render_views True \
    --exclude_outliers True
    '''

#sofa

    '''
    python mv_render.py \
    --category Sofa \
    --output_root /scratch/leuven/375/vsc37593/paint3d_res_strict/finetuned_lora_15000_4initviews \
    --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
    --raw_data_folder /scratch/leuven/375/vsc37593/3D-FUTURE-model \
    --num_views 8 \
    --max_samples -1 \
    --render_views True \
    --exclude_outliers True
    '''

    '''
    [RESULT] FID: 50.25031714502563
    [RESULT] KID: 0.009919895547052426
    '''

    '''
    python mv_render.py \
    --category Sofa \
    --output_root /scratch/leuven/375/vsc37593/paint3d_res_strict/finetuned_lora_15000_4initviews \
    --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
    --raw_data_folder /scratch/leuven/375/vsc37593/3D-FUTURE-model \
    --num_views 8 \
    --max_samples -1 \
    --render_views True \
    --exclude_outliers True
    '''

    '''
    python mv_render.py \
    --category Sofa \
    --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline_4views \
    --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
    --raw_data_folder /scratch/leuven/375/vsc37593/3D-FUTURE-model \
    --num_views 8 \
    --max_samples 100 \
    --render_views True \
    --exclude_outliers True
    '''

# table 


    '''
    python mv_render.py \
    --category Bed \
    --output_root /scratch/leuven/375/vsc37593/paint3d_res_strict/finetuned_lora_15000_4initviews \
    --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
    --raw_data_folder /scratch/leuven/375/vsc37593/3D-FUTURE-model \
    --num_views 8 \
    --max_samples -1 \
    --render_views True \
    --exclude_outliers True
    '''