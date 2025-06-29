import os
import json
import glob
import argparse
import torch
import gc
import shutil
from pathlib import Path
import sys
import warnings
from PIL import Image
warnings.filterwarnings("ignore", message="Error importing kaolin.visualize.ipython")

from paint3d.models.textured_mesh import TexturedMeshModel
from paint3d.utils import save_tensor_image

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, required=True, help='Model category, e.g., chair')
    parser.add_argument('--data_root', type=str, required=True, help='Root dir containing model folders')
    parser.add_argument('--output_root', type=str, required=True, help='Where to store stage1 outputs')

    # configs for paint3d pipeline
    parser.add_argument('--sd_config_stage1', type=str, required=True, help='Path to sd_config yaml')
    parser.add_argument('--sd_config_stage2', type=str, required=True, help='Path to sd_config yaml')
    parser.add_argument('--render_config', type=str, required=True, help='Path to render_config python file')
    parser.add_argument('--use_finetuned_depth2img', action='store_true', help='Use fine-tuned depth2img')
    parser.add_argument('--use_stage_2', action='store_true', help='Enable Paint3D stage 2 pipeline after stage 1')

    # model ID control
    parser.add_argument('--split_file_root', type=str, required=True, help='Path to train/test split .lst file')
    parser.add_argument('--max_sample_number', type=int, default=-1, help='Max number of models to process (default: all)')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without executing')

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

def render_multi_views(model_id, final_outdir, category_views_root, render_cfg, num_views=4):
    """
    Render multiple RGB views for FID evaluation and save them into category_views_root.
    """
    device = torch.device("cuda")

        # Setup shape + texture
    render_cfg.guide.shape_path = os.path.join(final_outdir, "mesh.obj")
    render_cfg.guide.initial_texture = os.path.join(final_outdir, "albedo.png")

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
        print(f"[INFO] Saved view {i} for {model_id} → {view_path}")


    # raise NotImplementedError

def get_model_info(category, model_id, data_root, output_root, stage):

    mesh_path = os.path.join(data_root, model_id, 'normalized_model.obj')
    ip_adapter_image_path = os.path.join(data_root, model_id, 'image.jpg')
    outdir = os.path.join(output_root, category, model_id, stage)

    return mesh_path, ip_adapter_image_path, outdir

def get_model_ids(category, split_file_root):
    """Read model_ids from a .lst file"""

    if not os.path.exists(split_file_root):
        raise FileNotFoundError(f"[ERROR] Split file not found: {split_file_root}")
    
    split_file_path = os.path.join(split_file_root, category, 'test.lst')
    
    with open(split_file_path, 'r') as f:
        model_ids = [line.strip() for line in f if line.strip()]
    
    return model_ids

def main():

    args = parse_args()
    model_ids = get_model_ids(args.category, args.split_file_root)

    if args.max_sample_number > 0:
        model_ids = model_ids[:args.max_sample_number]
    
    print(f"[INFO] Processing {len(model_ids)} models from category '{args.category}'")

    for idx, model_id in enumerate(model_ids):

        print(f"[{idx + 1}/{len(model_ids)}] Processing model: {model_id}")

        mesh_path, img_path, outdir_stage1 = get_model_info(args.category, model_id, args.data_root, args.output_root, "img_stage1")

        if not os.path.exists(mesh_path) or not os.path.exists(img_path):
            print(f"[WARNING] Skipping {model_id}: missing .obj or image.jpg")
            continue

        # breakpoint()
        # === Stage 1 Command ===
        cmd_stage1 = f"""
        python paint3d_launch_stage1.py
            {"--use_finetuned_depth2img" if args.use_finetuned_depth2img else ""}
            --sd_config {args.sd_config_stage1}
            --render_config {args.render_config}
            --mesh_path "{mesh_path}"
            --prompt " "
            --ip_adapter_image_path "{img_path}"
            --outdir "{outdir_stage1}"
        """
        cmd_stage1 = " ".join(cmd_stage1.split())

        if args.dry_run:
            print("[DRY RUN - STAGE 1]", cmd_stage1)
        else:
            os.makedirs(outdir_stage1, exist_ok=True)
            os.system(cmd_stage1)
            torch.cuda.empty_cache()
            gc.collect()
        
        # breakpoint()
        # === Optional Stage 2 ===
        if args.use_stage_2:
            texture_path = os.path.join(outdir_stage1, 'res-0', 'albedo.png')
            outdir_stage2 = os.path.join(args.output_root, args.category, model_id, 'img_stage2')

            if not os.path.exists(texture_path):
                print(f"[WARNING] Stage 2 skipped for {model_id}: missing {texture_path}")
            
            else:
                cmd_stage2 = f"""
                python paint3d_launch_stage2.py
                    --sd_config {args.sd_config_stage2}
                    --render_config {args.render_config}
                    --mesh_path "{mesh_path}"
                    --texture_path "{texture_path}"
                    --prompt " "
                    --ip_adapter_image_path "{img_path}"
                    --outdir "{outdir_stage2}"
                """
                cmd_stage2 = " ".join(cmd_stage2.split())

                if args.dry_run:
                    print("[DRY RUN - STAGE 2]", cmd_stage2)
                
                else:
                    os.makedirs(outdir_stage2, exist_ok=True)
                    print(f"[INFO] Running Stage 2 for {model_id}")
                    os.system(cmd_stage2)
        
                # === Memory cleanup ===
            torch.cuda.empty_cache()
            gc.collect()
        print(f"[INFO] Finished {model_id} — memory cleared.")

        # breakpoint()

        final_outdir = os.path.join(args.output_root, args.category, model_id)
        os.makedirs(final_outdir, exist_ok=True)

        # 1. Copy mesh.obj and mesh.mtl from stage1
        mesh_obj = os.path.join(outdir_stage1,'res-0', 'mesh.obj')
        mesh_mtl = os.path.join(outdir_stage1, 'res-0', 'mesh.mtl')

        if os.path.exists(mesh_obj):
            shutil.copy(mesh_obj, os.path.join(final_outdir, 'mesh.obj'))
        else:
            print(f"[WARNING] Missing mesh.obj for {model_id}")

        if os.path.exists(mesh_mtl):
            shutil.copy(mesh_mtl, os.path.join(final_outdir, 'mesh.mtl'))
        else:
            print(f"[WARNING] Missing mesh.mtl for {model_id}")
        
        # 2. Copy albedo (prefer Stage2 if available)
        albedo_stage1 = os.path.join(outdir_stage1, 'res-0', 'albedo.png')

        if args.use_stage_2:
            outdir_stage2 = os.path.join(args.output_root, args.category, model_id, 'img_stage2')
            albedo_stage2 = os.path.join(outdir_stage2, 'tile_res_0', 'albedo.png')
            if os.path.exists(albedo_stage2):
                shutil.copy(albedo_stage2, os.path.join(final_outdir, 'albedo.png'))
            elif os.path.exists(albedo_stage1):
                shutil.copy(albedo_stage1, os.path.join(final_outdir, 'albedo.png'))
            else:
                print(f"[WARNING] No albedo texture found for {model_id}")
        else:
            if os.path.exists(albedo_stage1):
                shutil.copy(albedo_stage1, os.path.join(final_outdir, 'albedo.png'))
            else:
                print(f"[WARNING] No albedo texture found for {model_id}")
        
        # === Optional FID View Rendering ===
        render_cfg_obj = load_render_cfg_from_pyfile(args.render_config)
        category_views = os.path.join(args.output_root, args.category + "_views")
        os.makedirs(category_views, exist_ok=True)

        try:
            render_multi_views(
                model_id=model_id,
                final_outdir=final_outdir,
                category_views_root=category_views,
                render_cfg=render_cfg_obj,
                num_views=8
            )
        except Exception as e:
            print(f"[WARNING] View rendering failed for {model_id}: {e}")




    
    print("[INFO] All models processed.")

if __name__ == '__main__':

    main()
    # cfg = load_render_cfg_from_pyfile('paint3d/config/train_config_paint3d.py')

    # print(cfg)




'''
python paint3d_launch_stage1.py \
 --use_finetuned_depth2img \
 --sd_config controlnet/config/finetuned_depth_based_cnet.yaml \
 --render_config paint3d/config/train_config_paint3d.py \
 --mesh_path /data/leuven/375/vsc37593/my_py_projects/Paint3D-main/demo/objs/0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d/normalized_model.obj \
 --prompt " " \
 --ip_adapter_image_path /data/leuven/375/vsc37593/my_py_projects/Paint3D-main/demo/objs/0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d/image.jpg \
 --outdir outputs/0da9753a-d2c8-4b02-9e7a-9fb3881f3c1d/img_stage1
'''