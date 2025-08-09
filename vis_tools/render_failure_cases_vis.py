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
                               mode='paint3d'):

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

    view_angles = [
        (60, 180),   # back
        (60, 45),    # left front
        (60, 315),   # right front
    ]
    view_images = []

    # === Save each view separately ===
    for idx, (theta_deg, phi_deg) in enumerate(view_angles):
        img = render_single_view(mesh_tex,
                                 theta=torch.tensor(theta_deg * torch.pi / 180),
                                 phi=torch.tensor(phi_deg * torch.pi / 180),
                                 radius=render_cfg.render.radius,
                                 render_cfg=render_cfg)
        img_path = output_dir / f"{model_id}_view_{idx}.png"
        img.save(str(img_path))
        print(f"[INFO] Saved: {img_path}")
    
    del mesh_tex
    torch.cuda.empty_cache()


if __name__ == "__main__":

    render_cfg = load_render_cfg_from_pyfile('/data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py')
    model_id='6f87d8ec-3bdc-46ec-b09c-7370a8878fef'

    render_cfg.render.radius = 1.3

    render_comparison_overview(
        model_id=model_id,
        model_folder=os.path.join('/data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/vis_tools/failure_examples/Chair', model_id),
        reference_data_root='/scratch/leuven/375/vsc37593/3D-FUTURE-model',
        render_cfg=render_cfg,
        output_dir=os.path.join('failure_case_output',model_id),
        device=torch.device('cuda'),
    )