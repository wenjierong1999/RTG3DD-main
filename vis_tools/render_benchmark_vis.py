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
import shutil

ROOT_DIR = Path(__file__).resolve().parents[1]  # 即 RTG3DD-main
sys.path.insert(0, str(ROOT_DIR))

from paint3d.models.textured_mesh import TexturedMeshModel
from paint3d.utils import save_tensor_image, color_with_shade, tensor2numpy

view_angles = [
    (60, 180),  
    (60, 45),    
    (60, 315),   
]

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


def render_single_view(mesh_model, theta, phi, radius, render_cfg):
    outputs = mesh_model.render(theta=theta, phi=phi, radius=radius, dims=(1024, 1024))
    z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
    mask, uncolored_masks = outputs['mask'], outputs['uncolored_mask']
    color_with_shade_img = color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals, light_coef=0.3)
    rgb_render = outputs['image'] * (1 - uncolored_masks) + color_with_shade_img * uncolored_masks
    img = Image.fromarray(tensor2numpy(rgb_render)).convert("RGB")
    return img

def render_comparison_overview(model_id, 
                               model_folder, 
                               reference_data_root, 
                               render_cfg, 
                               output_dir, 
                               device,
                               mode='point-uv-diff'):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == 'paint3d':
        shape_path = Path(model_folder) / "mesh.obj"
        texture_path = Path(model_folder) / "albedo.png"
    elif mode == 'point-uv-diff':
        shape_path = Path(model_folder) / f"{model_id}.obj"
        texture_path = Path(model_folder) / f"{model_id}.png"
    else:
        raise NotImplementedError()

    # === Step 0: Copy reference image ===
    ref_img_path = Path(reference_data_root) / model_id / "image.jpg"
    ref_img_dst = output_dir / f"{model_id}_ref.jpg"
    if ref_img_path.exists():
        shutil.copy(ref_img_path, ref_img_dst)
    else:
        print(f"[WARNING] Reference image not found for {model_id}: {ref_img_path}")

    # === Step 1: Render empty mesh (gray) ===
    render_cfg.guide.shape_path = str(shape_path)
    render_cfg.guide.initial_texture = None  # no texture
    mesh_gray = TexturedMeshModel(cfg=render_cfg, device=device, flip_texture=True)

    gray_img = render_single_view(mesh_gray,
                                  theta=torch.tensor(60.0 * torch.pi / 180),
                                  phi=torch.tensor(300.0 * torch.pi / 180),
                                  radius=render_cfg.render.radius,
                                  render_cfg=render_cfg)
    
    gray_img_path = output_dir / f"{model_id}_gray.png"
    gray_img.save(str(gray_img_path))
    del mesh_gray
    torch.cuda.empty_cache()

    # === Textured mesh rendering with 3 views (includes back) ===

    # render_cfg.guide.shape_path = str(shape_path)
    render_cfg.guide.initial_texture = str(texture_path)
    mesh_tex = TexturedMeshModel(cfg=render_cfg, device=device, flip_texture=True)

    # view_angles = [
    #     (60, 180),   # back
    #     (60, 45),    # left front
    #     (60, 315),   # right front
    # ]


    view_angles = [
        (60, 180),   # back
        (60, 45),    # left front
        (60, 315),   # right front
    ]
    view_images = []

    for theta_deg, phi_deg in view_angles:
        img = render_single_view(mesh_tex,
                                 theta=torch.tensor(theta_deg * torch.pi / 180),
                                 phi=torch.tensor(phi_deg * torch.pi / 180),
                                 radius=render_cfg.render.radius,
                                 render_cfg=render_cfg)
        view_images.append(img)
    
    del mesh_tex
    torch.cuda.empty_cache()

    # === Only combine the 3 view images ===
    composite_width = 1024 * len(view_images)
    final_img = Image.new("RGB", (composite_width, 1024), color=(255, 255, 255))
    for i, v in enumerate(view_images):
        final_img.paste(v, (i * 1024, 0))

    final_path = output_dir / f"{model_id}_overview.png"
    final_img.save(final_path)
    print(f"[INFO] Saved: {final_path}")

def process_one_model_name(model_name_root, 
                           reference_data_root, 
                           output_root, 
                           render_cfg, 
                           device=torch.device('cuda'), 
                           categories=None, 
                           max_models_per_category=-1,
                           mode='point-uv-diff'):

    model_name_root = Path(model_name_root)
    if categories is None:
        categories = [d.name for d in model_name_root.iterdir() if d.is_dir()]

    model_name = model_name_root.name
    output_model_root = Path(output_root) / model_name

    for category in categories:
        print(f"\n[INFO] Processing {model_name}/{category}")
        category_folder = model_name_root / category
        if not category_folder.exists():
            print(f"[WARNING] Category folder {category_folder} not found, skipping.")
            continue

        model_ids = sorted(os.listdir(category_folder))
        if max_models_per_category > 0:
            model_ids = model_ids[:max_models_per_category]

        for model_id in tqdm(model_ids, desc=f"Rendering {model_name}/{category}"):
            model_folder = category_folder / model_id
            # if not (model_folder / "mesh.obj").exists() or not (model_folder / "albedo.png").exists():
            #     print(f"[WARNING] Skipping {model_id}: missing mesh or texture.")
            #     continue

            out_dir = output_model_root / category / model_id
            out_dir.mkdir(parents=True, exist_ok=True)

            render_comparison_overview(
                model_id=model_id,
                model_folder=model_folder,
                reference_data_root=reference_data_root,
                render_cfg=render_cfg,
                output_dir=out_dir,
                device=device,
                mode=mode
            )


def main(demo_root, reference_data_root, output_root, render_cfg, device=torch.device('cuda')):

    demo_root = Path(demo_root)
    reference_data_root = Path(reference_data_root)
    output_root = Path(output_root)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_dirs = [p for p in sorted(demo_root.iterdir()) if p.is_dir()]

    for model_name_root in model_name_dirs:
        process_one_model_name(
            model_name_root=model_name_root,
            reference_data_root=reference_data_root,
            output_root=output_root,
            render_cfg=render_cfg,
            device=device
        )



if __name__ == "__main__":
    
    render_cfg = load_render_cfg_from_pyfile('/data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py')
    # render_comparison_overview(
    #     model_id='1bbfd20a-50da-44ea-936c-9518400ee666',
    #     model_folder='/scratch/leuven/375/vsc37593/demo_examples/pa-paint3d/Bed/1bbfd20a-50da-44ea-936c-9518400ee666',
    #     reference_data_root='/scratch/leuven/375/vsc37593/3D-FUTURE-model',
    #     render_cfg=render_cfg,
    #     output_dir='test_output',
    #     device=torch.device('cuda')
    # )

    main(
        demo_root='/scratch/leuven/375/vsc37593/point-uv-diffusion-demo',
        reference_data_root='/scratch/leuven/375/vsc37593/3D-FUTURE-model',
        output_root='test_output_pvd',
        render_cfg=render_cfg,
    )