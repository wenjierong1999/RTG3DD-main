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

ROOT_DIR = Path(__file__).resolve().parents[1]  # 即 RTG3DD-main
sys.path.insert(0, str(ROOT_DIR))

from paint3d.models.textured_mesh import TexturedMeshModel
from paint3d.utils import save_tensor_image

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

def get_model_ids(category, split_file_root):
    split_file_path = os.path.join(split_file_root, category, 'test.lst')
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"[ERROR] Split file not found: {split_file_path}")

    with open(split_file_path, 'r') as f:
        model_ids = [line.strip() for line in f if line.strip()]
    return model_ids


def render_multi_views(model_id, raw_data_root, category_views_root, render_cfg, num_views=4):
    """
    Render multiple RGB views for FID evaluation and save them into category_views_root.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

        # Setup shape + texture
    render_cfg.guide.shape_path = os.path.join(raw_data_root, model_id, 'normalized_model.obj')
    render_cfg.guide.initial_texture = os.path.join(raw_data_root, model_id, "texture.png")

        # Check existence
    if not os.path.exists(render_cfg.guide.shape_path) or not os.path.exists(render_cfg.guide.initial_texture):
        print(f"[WARNING] Cannot render views for {model_id}: missing mesh or texture.")
        return
    
        # Create model
    model = TexturedMeshModel(cfg=render_cfg, device=device, flip_texture=True)

    # Determine phi angles and corresponding thetas
    if num_views == 4:
        phis = [0, 90, 180, 270]
    elif num_views == 8:
        phis = list(range(0, 360, 45))  # [0, 45, 90, ..., 315]
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
            
        # Convert tensor to image and save
        image_tensor = render_out["image"].clamp(0, 1).cpu()
        image = (image_tensor * 255).byte().squeeze().permute(1, 2, 0).numpy()
        image_pil = Image.fromarray(image)

        view_path = os.path.join(category_views_root, f"{model_id}_view_{i}.png")
        image_pil.save(view_path)
        # print(f"[INFO] Saved view {i} for {model_id} → {view_path}")
    
    del model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--raw_data_root", type=str, required=True, help="Path to folder with GT mesh.obj + albedo.png")
    parser.add_argument("--split_file_root", type=str, required=True, help="Folder containing category/test.lst")
    parser.add_argument("--render_config", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True, help="Where to save rendered views")
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=-1)
    return parser.parse_args()

def main():

    args = parse_args()
    model_ids = get_model_ids(args.category, args.split_file_root)
    if args.max_samples > 0:
        model_ids = model_ids[:args.max_samples]
    
    print(f"[INFO] Rendering {len(model_ids)} models from category '{args.category}'")

    render_cfg = load_render_cfg_from_pyfile(args.render_config)
    category_views_root = os.path.join(args.output_root, args.category + "_views")

    for model_id in tqdm(model_ids, desc="Rendering multi-view"):
        try:
            render_multi_views(
                model_id=model_id,
                raw_data_root=args.raw_data_root,
                category_views_root=category_views_root,
                render_cfg=render_cfg,
                num_views=args.num_views
            )
        except Exception as e:
            print(f"[FATAL] Failed rendering for model {model_id}: {e}")
            continue

    print("[INFO] All models rendered.")


if __name__ == '__main__':

    main()