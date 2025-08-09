import os
from pathlib import Path
import time
from tqdm import tqdm
import PIL
import torch
import imageio
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    look_at_view_transform,
    TexturesUV,
    PerspectiveCameras
)

def render_360_texture(mesh_path,
                       texture_path,
                       output_dir,
                       num_views=36,
                       render_resolution=1024,
                       device='cuda',
                       save_frames=True,
                       make_video=False):
    '''
    # TODO:
    render a 3d mesh with a given texture map from 360 degrees views
    '''
    device = torch.device(device)

    # Load and preprocess the texture image
    texture_image = PIL.Image.open(texture_path).convert('RGB')
    texture_image = torch.from_numpy(np.array(texture_image)).float() / 255.0
    texture_image = texture_image.unsqueeze(0).to(device) # (N, H, W, C)


    # Load the mesh from .obj file
    mesh = load_objs_as_meshes([mesh_path], device=device)

    # Replace the mesh's texture with the provided texture image
    verts_uvs = mesh.textures.verts_uvs_padded()
    faces_uvs = mesh.textures.faces_uvs_padded()
    mesh.textures = TexturesUV(maps=texture_image, faces_uvs=faces_uvs, verts_uvs=verts_uvs)

    # --------------- Setup Renderer ---------------
    raster_settings = RasterizationSettings(
        image_size=render_resolution,
        blur_radius=0.0,
        faces_per_pixel=1
    )
    cameras = PerspectiveCameras(device=device)
    lights = PointLights(device=device, location=[[0.0, 2.0, 2.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    os.makedirs(output_dir, exist_ok=True)
    frames = []

    for i in range(num_views):
        # Compute the camera rotation and translation for the current view
        angle = 360.0 * i / num_views
        R, T = look_at_view_transform(dist=0.7, elev=15.0, azim=angle)
        R = R.to(device)
        T = T.to(device)

        # Render the mesh
        images = renderer(mesh, R=R, T=T)
        
        # Convert the image to numpy array and save it
        image = images[0, ..., :3].cpu().numpy()
        image = (image * 255).astype(np.uint8)
        if save_frames:
            frame_path = os.path.join(output_dir, f"view_{i:03d}.png")
            PIL.Image.fromarray(image).save(frame_path)
        frames.append(image)

            # --------------- Optional: Make a Video ---------------
    if make_video:
        video_path = os.path.join(output_dir, f"renderings_360_f{num_views}.mp4")
        imageio.mimsave(video_path, frames, fps=24)
        #print(f"Video saved to {video_path}")

def generate_360_render_views_video(root_dir, num_views, render_resolution):

    model_folders = [
        f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))
    ]

    for folder in tqdm(model_folders):

        start_time = time.time()
        folder_path = os.path.join(root_dir, folder)
        texture_path = os.path.join(folder_path, f'{folder}.png')
        mesh_path = os.path.join(folder_path, f'{folder}.obj')

        if not (os.path.exists(texture_path) and os.path.exists(mesh_path)):
            print(f"Skipping {folder}: missing texture_map.png or mesh.obj.")
            continue
        
        #print(f'OK {folder}')
        try:
            render_360_texture(
             mesh_path=mesh_path,
             texture_path=texture_path,
             num_views=num_views,
             render_resolution=render_resolution,
             output_dir=folder_path,
             save_frames=False,
             make_video=True
            )
        except Exception as e:
            print(f"Failed to render {folder}: {e}")


if __name__ == "__main__":

    results_folder = '/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-dev/results/cond_fine_train_3001627/3001627/static_timestamped/fine_output'
    # mesh_id = 'c2989bf8-2122-4204-9fa0-5cfe7a054777'
    # mesh_path = os.path.join(results_folder, mesh_id, f'{mesh_id}.obj')
    # texture_path = os.path.join(results_folder, mesh_id, f'{mesh_id}.png')

    # render_360_texture(
    #     mesh_path=mesh_path,
    #     texture_path=texture_path,
    #     output_dir='/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-dev/src/evaluator/tmp',
    #     save_frames=False,
    #     make_video=True
    # )

    # print(mesh_path)

    # print(mesh_path)

    generate_360_render_views_video(
        root_dir=results_folder,
        num_views=36,
        render_resolution=1024
    )