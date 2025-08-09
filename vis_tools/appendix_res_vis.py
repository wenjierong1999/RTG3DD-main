import os
import json
import glob
import argparse
import torch
from PIL import Image, ImageDraw
from pathlib import Path
import sys
from tqdm import tqdm
import gc
from cleanfid import fid
import shutil
import copy

ROOT_DIR = Path(__file__).resolve().parents[1]  # 即 RTG3DD-main
sys.path.insert(0, str(ROOT_DIR))

from paint3d.models.textured_mesh import TexturedMeshModel
from paint3d.utils import save_tensor_image, color_with_shade, tensor2numpy


view_angles = [
    (60, 0),
    (60, 72),
    (60, 144),
    (60, 216),
    (60, 288),
]

def resize_images(images, size=(1024, 1024)):
    return [im.resize(size, Image.LANCZOS) for im in images]

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


def load_reference_image(model_id,
                         raw_data_dir='/scratch/leuven/375/vsc37593/3D-FUTURE-model'):
    ref_img_path = Path(raw_data_dir) / model_id / "image.jpg"  # 假设命名是 image.jpg
    if not ref_img_path.exists():
        raise FileNotFoundError(f"Reference image for {model_id} not found at {ref_img_path}")
    return Image.open(ref_img_path).convert("RGB")


def add_red_border(img, border_width=10):
    draw = ImageDraw.Draw(img)
    for i in range(border_width):
        draw.rectangle(
            [i, i, img.width - 1 - i, img.height - 1 - i],
            outline=(255, 0, 0),
            width=1
        )
    return img

def concat_images_horizontally(images, highlight_first_n=0):
    for idx in range(min(highlight_first_n, len(images))):
        images[idx] = add_red_border(images[idx])

    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    x_offset = 0
    for im in images:
        new_img.paste(im, (x_offset, 0))
        x_offset += im.width
    return new_img

def concat_images_vertically(images):
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)
    new_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for im in images:
        new_img.paste(im, (0, y_offset))
        y_offset += im.height
    return new_img


def main(models_dir, save_path, model_list=None,
        max_models=3, device=torch.device('cuda')):

    models_dir = Path(models_dir)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    all_models = sorted([p.name for p in models_dir.iterdir() if p.is_dir()])

    if model_list is not None:
        selected_models = model_list
    else:
        selected_models = all_models[:max_models] if max_models else all_models

    row_images = []

    for model_id in tqdm(selected_models, desc="Processing models"):
        render_cfg = load_render_cfg_from_pyfile('/data/leuven/375/vsc37593/my_py_projects/RTG3DD-main/paint3d/config/train_config_paint3d.py')
        model_folder = models_dir / model_id
        mesh_path = model_folder / "mesh.obj"
        texture_path = model_folder / "albedo.png"

        if not mesh_path.exists() or not texture_path.exists():
            print(f"[WARN] Skipping {model_id}, missing mesh or texture")
            continue

        # --- Empty mesh ---
        render_cfg.guide.shape_path = str(mesh_path)
        render_cfg.guide.initial_texture = None  # no texture
        mesh_gray = TexturedMeshModel(cfg=render_cfg, device=device, flip_texture=True)

        gray_img = render_single_view(mesh_gray,
                                    theta=torch.tensor(60.0 * torch.pi / 180),
                                    phi=torch.tensor(300.0 * torch.pi / 180),
                                    radius=render_cfg.render.radius,
                                    render_cfg=render_cfg)
        
        # --- Reference image ---
        ref_img = load_reference_image(model_id)

        # ---Textured mesh ---
        render_cfg.guide.initial_texture = texture_path

        mesh_tex = TexturedMeshModel(cfg=render_cfg, device=device, flip_texture=True)

        rendered_views = []
        for theta_deg, phi_deg in view_angles:
            img = render_single_view(mesh_tex,
                                    theta=torch.tensor(theta_deg * torch.pi / 180),
                                    phi=torch.tensor(phi_deg * torch.pi / 180),
                                    radius=render_cfg.render.radius,
                                    render_cfg=render_cfg)
            rendered_views.append(img)
        images = [gray_img, ref_img] + rendered_views
        images = resize_images(images)
        row_img = concat_images_horizontally(images, highlight_first_n=0)
        row_images.append(row_img)

    if len(row_images) == 0:
        print("No valid models to process. Exiting.")
        return
    final_img = concat_images_vertically(row_images)

    max_width = 2000  
    w, h = final_img.size
    if w > max_width:
        new_h = int(h * max_width / w)
        final_img = final_img.resize((max_width, new_h), Image.LANCZOS)

    final_img.save(save_path)
    print(f"Saved concatenated result to {save_path}")

if __name__ == "__main__":

    chair_res = '/scratch/leuven/375/vsc37593/paint3d_res_strict/finetuned_lora_13000_4initviews/Chair'
    chair_model_ids = ['0cd0f02c-7bf9-46b2-ac69-680349b0d84a',
                      '3c2e5751-1e64-41a7-8625-991879d862dd',
                      '3fc6a3f3-2f14-46b3-b06b-ce80ec5c7742',
                      '4d3b74e2-45e5-4f2b-b2a1-7d49ddaed268',
                      '5ad2308d-0033-433e-a061-830b28f61ede',
                      '6f87d8ec-3bdc-46ec-b09c-7370a8878fef',
                      '86c99c80-ffde-3e4a-b530-716ea61ea8dd',
                      '069045d8-29ed-40cb-a014-3252241aa7aa' ]
    table_res = '/scratch/leuven/375/vsc37593/paint3d_res_strict/finetuned_lora_13000_4initviews/Table'
    table_model_ids = [
        '0fe0f46d-d3c7-428c-a0a0-e18f549279c7',
        '2d2c2199-9987-4480-b7f2-530b2a73791c',
        '7a819d08-0cf3-4801-a466-efd6b2833b7a',
        '8ce43f0e-5b75-49c6-b0a1-d88e4863b205',
        '24fd6551-39c7-4e39-ae1e-f6ea52447e1e',
        '9e3696ac-5f81-3648-b111-355bf479b22a',
        '51ee76e1-6037-4535-a8b9-9d989882aab8',
        '0207d0ca-ddda-3740-a866-8031de2f2ad8'
    ]

    bed_res = '/scratch/leuven/375/vsc37593/paint3d_res_strict/finetuned_lora_13000_4initviews/Bed'
    bed_model_ids = [
        '1bbfd20a-50da-44ea-936c-9518400ee666',
        '1b99f254-0d54-437f-a715-8d8351d99afa',
        '0ba987fb-1ff8-4bcd-9e6b-637a1ec78201',
        '03c4ad94-9507-4bcf-a90a-67f867db2111',
        '3cc05e45-1d67-425a-a99c-3e8cfe4f533a',
        '05b03925-70b7-4dc5-afaa-60567e89bb0a',
        '6e01d4b2-c436-4375-80c9-c6515b778fc1',
        '061e8cde-d140-431a-93fb-68b57754f2ae'
    ]

    sofa_res = '/scratch/leuven/375/vsc37593/paint3d_res_strict/finetuned_lora_15000_4initviews/Sofa'

    main(
        models_dir=sofa_res,
        save_path='appendix_output/concat_result_sofa.png',
        max_models = 8
        #model_list = bed_model_ids
        )