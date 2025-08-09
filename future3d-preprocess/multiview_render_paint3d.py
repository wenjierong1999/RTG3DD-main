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

BED_OUTLIERS = [
    '5f175677-3321-467a-9a82-1ede16807be6',
    '4d5da561-8adc-4c16-8eac-593740646701',
    '14a1c497-e2e5-41d1-94e1-9fbe63b3e964',
    '74d1f77d-3af6-47ab-a850-1c0a4d439e84',
    '7abbb840-1187-45ef-a42f-9912f1ec85d3',
    '14a1c497-e2e5-41d1-94e1-9fbe63b3e964',
    '0e25e454-e81d-4050-bfd1-d2d560b25777',
]

SOFA_OUTLIERS = [
    '006f9cee-42d9-49ae-bd2c-b790f96a404a'
]

def parse_args():

    parser = argparse.ArgumentParser(description="Render multiview images for Paint3D models.")
    parser.add_argument("--category", type=str, required=True, help="Object category, e.g., Chair.")
    parser.add_argument("--output_root", type=str, required=True, help="Root folder containing output model folders.")
    parser.add_argument("--render_config", type=str, required=True, help="Path to render config .py file.")
    parser.add_argument("--num_views", type=int, default=8, choices=[4, 8], help="Number of views to render per model.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max number of models to render (use -1 for all).")
    parser.add_argument("--gt_folder", type=str, required=True, help="Path to ground-truth multiview images folder.")
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


def mv_dr(model_id,theta,phi,mesh_model,render_cfg,save_result_dir):

    outputs = mesh_model.render(theta=theta,
                                phi=phi,
                                radius=render_cfg.render.radius,
                                dims=(512, 512))
    
    z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
    mask, uncolored_masks = outputs['mask'], outputs['uncolored_mask']
    color_with_shade_img = color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals, light_coef=0.3)
    rgb_render = outputs['image'] * (1 - uncolored_masks) + color_with_shade_img * uncolored_masks

    save_path = os.path.join(save_result_dir, f"{model_id}_view_{i}.png")
    save_tensor_image(tensor2numpy(rgb_render), save_path=save_path)

    # pass






def render_multi_views_from_output(model_id, 
                                   model_folder, 
                                   category_views_root, 
                                   render_cfg, 
                                   num_views=8,
                                   exclude_model_ids=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    shape_path = model_folder / "mesh.obj"
    tex_path = model_folder / "albedo.png"

    if not shape_path.exists() or not tex_path.exists():
        print(f"[WARNING] Cannot render views for {model_id}: missing mesh or texture.")
        return

    render_cfg.guide.shape_path = str(shape_path)
    render_cfg.guide.initial_texture = str(tex_path)

    model = TexturedMeshModel(cfg=render_cfg, device=device, flip_texture=True)

    if num_views == 4:
        phis = [0, 90, 180, 270]
    elif num_views == 8:
        phis = list(range(0, 360, 45))
    else:
        raise ValueError(f"Unsupported number of views: {num_views}. Must be 4 or 8.")

    thetas = [render_cfg.render.base_theta] * len(phis)
    os.makedirs(category_views_root, exist_ok=True)

    for i, (phi, theta) in enumerate(zip(phis, thetas)):
        phi_rad = torch.tensor(phi * torch.pi / 180)
        theta_rad = torch.tensor(theta * torch.pi / 180)

        try:
            render_out = model.render(
                theta=theta_rad,
                phi=phi_rad,
                radius=render_cfg.render.radius,
                dims=[512, 512]
            )
        except Exception as e:
            print(f"[ERROR] Rendering view {i} failed for {model_id}: {e}")
            continue

        image_tensor = render_out["image"].clamp(0, 1).cpu()
        image = (image_tensor * 255).byte().squeeze().permute(1, 2, 0).numpy()
        image_pil = Image.fromarray(image)

        view_path = category_views_root / f"{model_id}_view_{i}.png"
        image_pil.save(view_path)

    del model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    gc.collect()

def main():

    args = parse_args()
    render_cfg = load_render_cfg_from_pyfile(args.render_config)

    category_dir = Path(args.output_root) / args.category
    category_views_root = Path(args.output_root) / f"{args.category}_views"

    all_model_ids = [
        d.name for d in category_dir.iterdir()
        if d.is_dir() and not d.name.endswith("-F")
    ]

    if args.max_samples > 0:
        all_model_ids = all_model_ids[:args.max_samples]

    if args.render_views:

        # === Optional: Exclude CHAIR outliers ===
        if args.exclude_outliers and args.category.lower() == "chair":
            all_model_ids = [mid for mid in all_model_ids if mid not in CHAIR_OUTLIERS]
        elif args.exclude_outliers and args.category.lower() == "bed":
            all_model_ids = [mid for mid in all_model_ids if mid not in BED_OUTLIERS]

        print(f"[INFO] Found {len(all_model_ids)} models in '{category_dir}'")
        for model_id in tqdm(all_model_ids, desc="Rendering output models"):
            model_folder = category_dir / model_id
            try:
                render_multi_views_from_output(
                    model_id=model_id,
                    model_folder=model_folder,
                    category_views_root=category_views_root,
                    render_cfg=render_cfg,
                    num_views=args.num_views
                )
            except Exception as e:
                print(f"[FATAL] Failed rendering for model {model_id}: {e}")
                continue
        print("[INFO] All output models rendered.")
    
    print("[INFO] Computing FID and KID between rendered views and ground-truth...")

    fid_score = fid.compute_fid(
        str(category_views_root),  # your generated images
        args.gt_folder,
        mode="clean"
    )
    kid_score = fid.compute_kid(
        str(category_views_root),
        args.gt_folder,
        mode="clean"
    )

    print(f"[RESULT] FID score: {fid_score:.6f}")
    print(f"[RESULT] KID score: {kid_score:.6f}")


if __name__ == '__main__':

    main()

    '''
    python multiview_render_paint3d.py \
    --category Chair \
    --output_root /scratch/leuven/375/vsc37593/paint3d_res/finetuned_lora_14000_4initviews \
    --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
    --num_views 8 \
    --max_samples -1
    '''

    '''
    python multiview_render_paint3d.py \
        --category Chair \
        --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline \
        --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
        --num_views 8 \
        --max_samples -1 \
        --gt_folder /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/Chair_views \
        --render_views False
    '''


    '''
    python multiview_render_paint3d.py \
        --category Chair \
        --output_root /scratch/leuven/375/vsc37593/paint3d_res/finetuned_lora_14000_2initviews \
        --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
        --num_views 8 \
        --max_samples -1 \
        --gt_folder /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/Chair_views \
        --render_views True
    '''



    '''
    python multiview_render_paint3d.py \
        --category Chair \
        --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline_4views \
        --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
        --num_views 8 \
        --max_samples -1 \
        --gt_folder /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/Chair_views \
        --render_views True
        --
    '''

# Chair category


    '''
    python multiview_render_paint3d.py \
        --category Chair \
        --output_root /scratch/leuven/375/vsc37593/paint3d_res/finetuned_lora_10000_2initviews \
        --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
        --num_views 8 \
        --max_samples -1 \
        --gt_folder /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/Chair_views \
        --render_views True
    '''



    '''
    python multiview_render_paint3d.py \
        --category Chair \
        --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline \
        --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
        --num_views 8 \
        --max_samples -1 \
        --gt_folder /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/Chair_views \
        --render_views True
        --exclude_outliers True
    '''




    '''
    python multiview_render_paint3d.py \
        --category Table \
        --output_root /scratch/leuven/375/vsc37593/paint3d_res/finetuned_lora_13000_4initviews_65 \
        --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
        --num_views 8 \
        --max_samples -1 \
        --gt_folder /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/Table_views \
        --render_views True \
        --exclude_outliers True
    '''

#Table category

    '''
    python multiview_render_paint3d.py \
        --category Table \
        --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline_4views \
        --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
        --num_views 8 \
        --max_samples 90 \
        --gt_folder /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/Table_views \
        --render_views True \
        --exclude_outliers True
    '''


    '''
    python multiview_render_paint3d.py \
        --category Table \
        --output_root /scratch/leuven/375/vsc37593/paint3d_res/finetuned_lora_13000_2initviews_65 \
        --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
        --num_views 8 \
        --max_samples -1 \
        --gt_folder /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/Table_views \
        --render_views True \
        --exclude_outliers True
    '''

    '''
    python multiview_render_paint3d.py \
        --category Chair \
        --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline_4views \
        --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
        --num_views 8 \
        --max_samples -1 \
        --gt_folder /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/Chair_views \
        --render_views True \
        --exclude_outliers True
    '''


    '''
    python multiview_render_paint3d.py \
        --category Chair \
        --output_root /scratch/leuven/375/vsc37593/paint3d_res_strict/finetuned_lora_13000_4initviews \
        --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
        --num_views 8 \
        --max_samples -1 \
        --gt_folder /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/Chair_views \
        --render_views True \
        --exclude_outliers True
    '''


# bed

    '''
    python multiview_render_paint3d.py \
        --category Bed \
        --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline_4views \
        --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
        --num_views 8 \
        --max_samples 80 \
        --gt_folder /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/Bed_views \
        --render_views True \
        --exclude_outliers True
    '''


# sofa

    '''
    python multiview_render_paint3d.py \
        --category Sofa \
        --output_root /scratch/leuven/375/vsc37593/paint3d_res/baseline_4views \
        --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
        --num_views 8 \
        --max_samples -1 \
        --gt_folder /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/Sofa_views \
        --render_views False \
        --exclude_outliers True
    '''

    '''
    python multiview_render_paint3d.py \
        --category Sofa \
        --output_root /scratch/leuven/375/vsc37593/paint3d_res_strict/finetuned_lora_15000_4initviews \
        --render_config /data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py \
        --num_views 8 \
        --max_samples -1 \
        --gt_folder /scratch/leuven/375/vsc37593/3D-FUTURE-gt-view-8/Sofa_views \
        --render_views False \
        --exclude_outliers True
    '''
