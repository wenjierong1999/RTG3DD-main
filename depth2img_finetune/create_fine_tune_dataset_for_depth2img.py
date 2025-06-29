import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import kaolin as kal

from PIL import Image
from paint3d.models.textured_mesh import TexturedMeshModel
from paint3d.utils import save_tensor_image

from paint3d.config.train_config_paint3d import TrainConfig

def get_model_ids_by_super_category(super_categories, 
                                    json_path='/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-dev/3D-Future-Demo/model_info.json'):
    model_ids = []
    with open(json_path, 'r') as f:
        data = json.load(f)  # data is a list of dicts

    for entry in data:
        if entry.get("super-category") in super_categories:
            model_ids.append(entry.get("model_id"))
    return model_ids

def get_camera_from_view(theta, phi, cfg):
    '''
    x = r * sinθ * sinφ
    y = r * cosθ
    z = r * sinθ * cosφ
    → camera position
    → look at (0, dy, 0)
    → up = [0, 1, 0]
    → camera transformation matrix
    
    '''
    radius = cfg.render.radius
    look_at_height = cfg.render.look_at_height

    x = radius * torch.sin(theta) * torch.sin(phi)
    y = radius * torch.cos(theta)
    z = radius * torch.sin(theta) * torch.cos(phi)

    pos = torch.tensor([x, y, z]).unsqueeze(0)  
    look_at = torch.zeros_like(pos)  
    look_at[:, 1] = look_at_height  
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)  
    camera_transform = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
    return camera_transform


def get_view_angles(cfg):
    
    size = cfg.render.n_views # Number of views
    # phis = [(index / size) * 360 for index in range(size)]
    # thetas = [cfg.render.base_theta for _ in range(size)] 

    # Define 4 canonical views: front, right, back, left
    phis = [0, 90, 180, 270]
    thetas = [cfg.render.base_theta] * 4  # same elevation

    if cfg.render.alternate_views: #  

        alternate_lists = lambda l: (
        [l[0]] +                                  # Start with the first element
        [i for j in zip(l[1:size // 2],           # Pair up elements from the front half (excluding first)
                    l[-1:size // 2:-1])        # with elements from the mirrored second half (reverse)
        for i in j] +                            # Flatten the pairs
        [l[size // 2]]                            # Append the middle element (center of the list)
        )
        phis = alternate_lists(phis)
        thetas = alternate_lists(thetas)
    
    # for phi, theta in cfg.render.views_before:
    #     phis = [phi] + phis
    #     thetas = [theta] + thetas
    # for phi, theta in cfg.render.views_after:
    #     phis = phis + [phi]
    #     thetas = thetas + [theta]
    
    return list(zip(phis, thetas))


def render_single_model(cfg, model_dir, output_root):


    model_id = os.path.basename(model_dir)
    shape_path = os.path.join(model_dir, "normalized_model.obj")
    texture_path = os.path.join(model_dir, "texture.png")

    cfg.guide.shape_path = shape_path
    cfg.guide.initial_texture = texture_path
    cfg.log.exp_path = os.path.join(output_root, model_id, "tmp")
    cfg.log.cache_path = os.path.join(output_root, model_id, "cache")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.log.exp_path, exist_ok=True)


    # model = TexturedMeshModel(cfg, device)


    try:
        model = TexturedMeshModel(cfg, device)
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_id}: {e}")
        # traceback.print_exc()
        return
    
    # os.makedirs(cfg.log.exp_path, exist_ok=True)
    angles = get_view_angles(cfg)
    save_root = os.path.join(output_root, model_id)

    for i, (phi, theta) in enumerate(angles):
        theta_rad = torch.tensor(theta * torch.pi / 180)
        phi_rad = torch.tensor(phi * torch.pi / 180)
        view_dir = os.path.join(save_root, f"view_{i:03d}")


        render_out = model.render(theta=theta_rad, 
                                  phi=phi_rad, 
                                  radius=cfg.render.radius,
                                  dims=[512,512])
        
        os.makedirs(view_dir, exist_ok=True)
        
        camera_transform = get_camera_from_view(theta_rad, phi_rad, cfg).squeeze(0)

        # print(camera_transform.squeeze(0).shape)

        view_names = {0: "front", 90: "right", 180: "back", 270: "left"}
        view_name = view_names.get(phi, "unknown")

        save_tensor_image(render_out["image"], os.path.join(view_dir, "rgb.png"))
        save_tensor_image(render_out["depth"], os.path.join(view_dir, "depth.png"))

        cam_info = {
            "azimuth": float(phi),
            "elevation": float(theta),
            "theta": float(theta_rad.item()),
            "phi": float(phi_rad.item()),
            "camera_transform": camera_transform.tolist(),
            "prompt": f"a photo of a modern furniture from {view_name} view"
        }
        with open(os.path.join(view_dir, "camera.json"), "w") as f:
            json.dump(cam_info, f, indent=2)
        
        torch.save(camera_transform, os.path.join(view_dir, "camera_transform.pt"))


if __name__ == "__main__":

    '''
    python create_fine_tune_dataset_for_depth2img.py \
    --input_root /scratch/leuven/375/vsc37593/3D-FUTURE-model \
    --output_root /data/leuven/375/vsc37593/my_py_projects/test_depth2img_finetune_dataset\
    --super_categories Chair,Table,Sofa,Bed\
    --max_models 5
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True, help="Root dir with 3D-Future model folders")
    parser.add_argument("--output_root", type=str, required=True, help="Output path for rendered views")
    parser.add_argument("--super_categories", type=str, default="", 
                help="Comma-separated list of super-categories to include (e.g., Chair,Table)")
    parser.add_argument("--max_models", type=int, default=-1, help="Optional limit")
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.render.alternate_views = False

    if args.super_categories:
        super_categories = [x.strip() for x in args.super_categories.split(",")]
        model_info_path = os.path.join(args.input_root, "model_info.json")
        filtered_model_ids = set(get_model_ids_by_super_category(super_categories, model_info_path))
        print(f"Using super-categories: {super_categories}")
    else:
        filtered_model_ids = None  # no filtering

    model_dirs = sorted([
        os.path.join(args.input_root, d)
        for d in os.listdir(args.input_root)
        if os.path.isdir(os.path.join(args.input_root, d)) and
           os.path.exists(os.path.join(args.input_root, d, "normalized_model.obj")) and
           (filtered_model_ids is None or d in filtered_model_ids)
    ])


    if args.max_models > 0:
        model_dirs = model_dirs[:args.max_models]
    
    print(f'collected {len(model_dirs)} meshes')

    for model_dir in tqdm(model_dirs):
        render_single_model(cfg, model_dir, args.output_root)
    

    # res = get_view_angles(cfg)

    # print(res)