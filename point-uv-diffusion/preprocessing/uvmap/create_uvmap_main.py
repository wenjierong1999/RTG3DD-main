import pdb
import point_cloud_utils as pcu
import os
import argparse
import time
import numpy as np
import logging
import datetime
from multiprocessing import Pool
import torch
import xatlas
from nvdiffrast import torch as dr
import trimesh
import torchvision
import cv2
import pandas as pd
import PIL
from torch_geometric.nn.unpool import knn_interpolate
from torch_cluster import fps
from geometry_tools.sample_camera_distribution import *
from geometry_tools.camera import *
from geometry_tools.rasterize_mesh import *
from dataset_utils import *
from tqdm import tqdm
from get_image_clip_cond import *
from datetime import datetime


'''
Basic configurations
'''


# configurations for paths

parser = argparse.ArgumentParser('data_process', add_help=False)
# parser.add_argument('--length', default=500, type=int)
parser.add_argument(
    '--save_folder', type=str, default='/data/leuven/375/vsc37593/my_py_projects/test_preprocess_output',
    help='path for saving rendered image')
parser.add_argument(
    '--dataset_folder', type=str, default='/scratch/leuven/375/vsc37593/3D-FUTURE-model/3D-FUTURE-model',
    help='path for downloaded 3d dataset folder')
parser.add_argument('--model_filename', type=str, default='raw_model.obj', 
                    help='file name of obj file - raw_model.obj or normalized_model.obj?')
parser.add_argument('--flag_csv_path', type=str, default='/data/leuven/375/vsc37593/my_py_projects/test_preprocess_output/flag.csv',
                    help='path for flag csv')

# configurations for uv mapping settings

parser.add_argument('--category_list', nargs='+', type=str, default=['Chair'])
parser.add_argument('--uv_resolution', type=int, default=512)
parser.add_argument('--render_res', type=int, default=512)
parser.add_argument('--mesh_scale', type=float, default=0.7)
parser.add_argument('--view_num', type=int, default=24)
parser.add_argument('--fps_num', type=int, default=4096)

args = parser.parse_args()


view_num = args.view_num
save_dir = args.save_folder # main saving folder
mesh_dir = args.dataset_folder
model_filename = args.model_filename # model id
flag_csv_path = args.flag_csv_path

category_list = args.category_list
mesh_scale = args.mesh_scale
fps_num = args.fps_num
uv_resolution = args.uv_resolution
render_res = args.render_res

#TODO: update the obj path to include the category folder 
#'/data/model.obj' → '/data/model_category/model.obj'"

#mesh_dir = dataset_folder
# camera_dir = os.path.join(renderviews_folder, 'camera') 
# img_dir = os.path.join(renderviews_folder, 'img') 
#save_dir = os.path.join(save_folder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ctx = dr.RasterizeCudaContext(device=device)
'''
Define some utils functions
'''
def make_file(path):

    if os.path.splitext(path)[1]:  
        dir_path = os.path.dirname(path)
    else:  
        dir_path = path

    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def set_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def normalize_mesh(mesh, mesh_scale=0.7):
    '''
    normalizing the mesh
    '''
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    #assert center.all() == 0
    scale = mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    mesh.vertices = vertices

    return mesh

def savemeshtes2(pointnp_px3, tcoords_px2, facenp_fx3, facetex_fx3, fname):
    import os
    fol, na = os.path.split(fname)
    na, _ = os.path.splitext(na)

    matname = '%s/%s.mtl' % (fol, na)
    fid = open(matname, 'w')
    fid.write('newmtl material_0\n')
    fid.write('Kd 1 1 1\n')
    fid.write('Ka 0 0 0\n')
    fid.write('Ks 0.4 0.4 0.4\n')
    fid.write('Ns 10\n')
    fid.write('illum 2\n')
    fid.write('map_Kd %s.png\n' % na)
    fid.close()
    ####

    fid = open(fname, 'w')
    fid.write('mtllib %s.mtl\n' % na)

    for pidx, p in enumerate(pointnp_px3):
        pp = p
        fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))

    for pidx, p in enumerate(tcoords_px2):
        pp = p
        fid.write('vt %f %f\n' % (pp[0], pp[1]))

    fid.write('usemtl material_0\n')
    for i, f in enumerate(facenp_fx3):
        f1 = f + 1
        f2 = facetex_fx3[i] + 1
        fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
    fid.close()


'''
Helper functions for uv mapping
'''


def interpolate(attr, rast, attr_idx, rast_db=None):
    '''
    :param attr: vertex attributes for interpolation
    :param rast: rast
    :param attr_idx: mesh face
    :return: attributes map for a view
    '''
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

def texture_sample(tex, uv):
    '''
    tex: Texture tensor with dtype `torch.float32`. For 2D textures, must have shape
             [minibatch_size, tex_height, tex_width, tex_channels]. For cube map textures,
             must have shape [minibatch_size, 6, tex_height, tex_width, tex_channels] where
             tex_width and tex_height are equal. Note that `boundary_mode` must also be set
             to 'cube' to enable cube map mode. Broadcasting is supported along the minibatch axis.
        uv: Tensor containing per-pixel texture coordinates. When sampling a 2D texture,
            must have shape [minibatch_size, height, width, 2]. When sampling a cube map
            texture, must have shape [minibatch_size, height, width, 3].

        Here uv is: given a view, we need to tell each pixel what its texture coordinate is
    :return: the texture image
    '''
    return dr.texture(tex, uv)

def rasterize(ctx, vertex_clip, mesh_f, resolution):
    '''
    :param ctx: self.ctx = dr.RasterizeGLContext(device=self.device)
    :param vertex_clip: vertex transformed in clip space
    :param mesh_f: mesh face Nx3
    :param resolution: map resolution
    :return:
    '''
    rast, _ = dr.rasterize(ctx, vertex_clip, mesh_f, (resolution, resolution))
    return rast

def xatlas_uvmap(ctx, mesh_v, mesh_f, resolution):
    '''
    :param ctx: self.ctx = dr.RasterizeGLContext(device=self.device)
    :param mesh_v: mesh vertex
    :param mesh_f: mesh face
    :param resolution:
    :return:
    '''
    # The parametrization potentially duplicates vertices.
    # `vmapping` contains the original vertex index for each new vertex (shape N, type uint32).
    # `indices` contains the vertex indices of the new triangles (shape Fx3, type uint32)
    # `uvs` contains texture coordinates of the new vertices (shape Nx2, type float32)
    # vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    vmapping, indices, uvs_ori = xatlas.parametrize(mesh_v.detach().cpu().numpy(), mesh_f.detach().cpu().numpy())

    # Convert to tensors
    indices_int64 = indices.astype(np.int64, casting='same_kind').view(np.int64)

    uvs = torch.tensor(uvs_ori, dtype=torch.float32, device=mesh_v.device)
    mesh_vt_f = torch.tensor(indices_int64, dtype=torch.int64, device=mesh_v.device)
    # mesh_v_tex. ture
    uv_clip = uvs[None, ...] * 2.0 - 1.0 # [0,1] -> [-1,1]

    # pad to four component coordinate 
    # uv_clip4: B*N_v*4
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[..., 0:1]), torch.ones_like(uv_clip[..., 0:1])), dim=-1)
    #pdb.set_trace()
    # rasterize
    rast = rasterize(ctx, uv_clip4, mesh_vt_f.int(), resolution) # rast: B*res*res*4
    # rast[...,0:3] : barycentrics(u,v)
    # rast[...,3:4]: triangle_id

    #pdb.set_trace()
    # Interpolate world space position
    gb_pos, _ = interpolate(mesh_v[None, ...], rast, mesh_f.int()) # gb_pos: B*res*res*3
    # Interpolate normal
    mesh_n = pcu.estimate_mesh_vertex_normals(mesh_v.detach().cpu().numpy(), mesh_f.int().detach().cpu().numpy())
    mesh_n = torch.from_numpy(mesh_n).to(device='cuda').float()
    gb_normal, _ = interpolate(mesh_n[None, ...], rast, mesh_f.int())
    #pdb.set_trace()
    gb_info = {}
    gb_info['pos'] = gb_pos
    gb_info['normal'] = gb_normal

    mask = rast[..., 3:4] > 0


    return uvs, mesh_vt_f, gb_info, mask, vmapping, indices, uvs_ori

'''
get pos textures
'''

def load_render_view(camera_path, image_path, device, view_num=24):
    rotation_camera = np.load(os.path.join(camera_path, 'rotation.npy'))
    elevation_camera = np.load(os.path.join(camera_path, 'elevation.npy'))
    assert len(rotation_camera) == len(elevation_camera)
    #indices = np.random.randint(rotation_camera.shape[0], size=view_num)

    data_out = {}
    images = []
    masks = []
    for i in range(view_num):
        image_path_file = os.path.join(image_path, '%03d.png' % i)
        image = PIL.Image.open(image_path_file)
        image = np.array(image) / 255
        color, mask = image[..., :3], image[..., 3:]
        images.append(color)
        masks.append(mask)
    data_out[None] = torch.from_numpy(np.stack(images, axis=0)).to(device)
    data_out['mask'] = torch.from_numpy(np.stack(masks, axis=0)).to(device)
    # data_out['rotation'] = torch.from_numpy(rotation_camera[indices]).to(device)
    # data_out['elevation'] = torch.from_numpy(elevation_camera[indices]).to(device)
    data_out['rotation'] = torch.from_numpy(rotation_camera).to(device)
    data_out['elevation'] = torch.from_numpy(elevation_camera).to(device)

    return data_out

def save_pointcloud(ply_filename, xyz: torch.Tensor, color: torch.Tensor, mask: torch.Tensor):
    xyz = xyz.view(-1, 3).cpu().numpy()
    color = color.view(-1, 3).cpu().numpy()
    mask = mask.view(-1).cpu().numpy()
    del_idx = np.where(mask==0)

    xyz = np.delete(xyz, del_idx[0], axis=0)
    color = np.delete(color, del_idx[0], axis=0)
    #pdb.set_trace()
    #idx = pcu.downsample_point_cloud_poisson_disk(xyz, num_samples=100000)
    #xyz = xyz[idx]
    #color = color[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(ply_filename, pcd)

    return xyz, color

def del_invalid_pc(xyz: torch.Tensor, color: torch.Tensor, mask: torch.Tensor):
    xyz = xyz.view(-1, 3).cpu().numpy()
    color = color.view(-1, 3).cpu().numpy()
    mask = mask.view(-1).cpu().numpy()
    del_idx = np.where(mask == 0)

    xyz = np.delete(xyz, del_idx[0], axis=0)
    color = np.delete(color, del_idx[0], axis=0)

    return xyz, color


def align_blender_imgs(rotation_camera, elevation_camera):
    assert len(rotation_camera) == len(elevation_camera)
    cam_mv = []

    def to_cuda_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(torch.float32).to('cuda')
        elif isinstance(x, torch.Tensor):
            return x.to(torch.float32).to('cuda')
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
    
    rotation_camera = to_cuda_tensor(rotation_camera)
    elevation_camera = to_cuda_tensor(elevation_camera)


    for img_idx, frame in enumerate(rotation_camera):
        # see GET3D-->dataset.py--L312
        theta = rotation_camera[img_idx] / 180 * np.pi
        phi = (90 - elevation_camera[img_idx]) / 180.0 * np.pi
        # theta = torch.from_numpy(np.array(theta)).to(device)
        # phi = torch.from_numpy(np.array(phi)).to(device)
        # see sample_camera_distribution.py--L107-L117
        compute_theta = -theta - 0.5 * math.pi
        output_points = torch.zeros((1, 3), device='cuda')
        sample_r = 1.2 # see render_shapenet.py--L209
        output_points[:, 0:1] = sample_r * torch.sin(phi) * torch.cos(compute_theta)
        output_points[:, 2:3] = sample_r * torch.sin(phi) * torch.sin(compute_theta)
        output_points[:, 1:2] = sample_r * torch.cos(phi)
        # see sample_camera_distribution.py--L78
        forward_vector = normalize_vecs(output_points)
        cam_pos = create_my_world2cam_matrix(forward_vector, output_points, device=device)


        # cam_pos = torch.from_numpy(cam_pos)
        cam_mv.append(cam_pos)
    cam_mv = torch.cat(cam_mv, dim=0)
    return cam_mv

def get_query_position(render_views, verts, faces, device, render_res):

    fovy = np.arctan(32 / 2 / 35) * 2
    fovyangle = fovy / np.pi * 180.0
    dmtet_camera = PerspectiveCamera(fovy=fovyangle)
    dmtet_renderer = NeuralRender(camera_model=dmtet_camera)
    
    verts = torch.from_numpy(np.array(verts)).to(device).float()
    faces = torch.from_numpy(np.array(faces)).to(device).float()
    rotation_camera, elevation_camera = render_views['rotation'], render_views['elevation']
    cam_mv = align_blender_imgs(rotation_camera, elevation_camera).float()
    # pdb.set_trace()
    position = []
    mask = []
    for i, cam_single_view in enumerate(cam_mv):
        return_value = render_mesh(dmtet_renderer, verts, faces, cam_single_view.unsqueeze(0),
                                   resolution=render_res)

        mk = return_value['mask'].squeeze(0)
        pc = return_value['tex_pos'].squeeze(0)
        position.append(pc)
        mask.append(mk)
    # pdb.set_trace()
    return torch.stack(position, dim=0), torch.stack(mask, dim=0)

def get_pos_textures(ctx, gb_pos, pc, device):
    '''
    :param ctx:
    :param gb_pos: position {'xyz': Nx3, 'mask': Nx1} Tensor
    :param pc: color point cloud {'xyz': Nx3, 'color': Nx3} Tensor
    :return: updated gb_pos_color: Nx3 Tensor
    '''
    xyz = pc['xyz']
    color = pc['color']

    tex_color = torch.zeros(gb_pos['xyz'].size()).to(device).float()
    tex_mask_idx = np.where(gb_pos['mask'].cpu().numpy()==0)
    valid_point = gb_pos['xyz'][tex_mask_idx]

    color_knn = knn_interpolate(color, xyz, valid_point, k=3)
    tex_color[tex_mask_idx] = color_knn
    '''
    row, col = knn(xyz, valid_point, 8)
    # import pdb; pdb.set_trace()
    edge_index = torch.stack([col, row], dim=0)
    color_max = knn_fusion(color, edge_index)
    tex_color[tex_mask_idx] = color_max
    '''
    return tex_color


def get_fps_points(gb_pos, tex_color, fps_num=fps_num):
    #param gb_pos: position {'xyz': Nx3, 'normal': Nx3, 'mask': Nx1} Tensor
    tex_points = gb_pos['xyz']
    tex_normals = gb_pos['normal']
    tex_mask_flat = gb_pos['mask'].view(-1)
    valid_points = tex_points[tex_mask_flat == 0] # valid points
    valid_normals = tex_normals[tex_mask_flat == 0] # valid normals
    valid_colors = tex_color[tex_mask_flat == 0] # valid colors

    ratio = (fps_num+1) / valid_points.shape[0]
    sample_idx = fps(valid_points, ratio= ratio, batch=None, random_start=False)
    sample_idx = sample_idx[:fps_num]

    assert sample_idx.shape[0] == fps_num, 'sampled points not enough'
    sampled_points = valid_points[sample_idx]
    sampled_normals = valid_normals[sample_idx]
    sampled_colors = valid_colors[sample_idx]

    # create texture map based on sampled points (training data for coarse stage)

    tex_color_fps = torch.zeros(tex_points.shape).to(device)
    color_knn_fps = knn_interpolate(sampled_colors, sampled_points, valid_points, k=3)
    tex_color_fps[tex_mask_flat == 0] = color_knn_fps
    color_map_fps = tex_color_fps.view(uv_resolution, uv_resolution, 3)

    color_map_fps_img = np.asarray(color_map_fps.data.cpu().numpy(), dtype=np.float32)
    color_map_fps_img = color_map_fps_img * 255
    color_map_fps_img = color_map_fps_img.clip(0,255).astype(np.uint8)

    return sampled_points, sampled_colors, sampled_normals, color_map_fps_img

'''
Helper function to create rendered images
'''
class TextureRender:
    
    def __init__(self, device='cuda'):
        self.device = device
        self.ctx = dr.RasterizeCudaContext(device='cuda')
        self.projection_mtx = None
        fovy = np.arctan(32 / 2 / 35) * 2
        fovyangle = fovy / np.pi * 180.0
        dmtet_camera = PerspectiveCamera(fovy=fovyangle, device=device)
        self.camera = dmtet_camera

    def texture_to_view(self, mesh, camera_mv_bx4x4, texture_map, resolution=256):
        '''
        :param mesh:
               mesh['v'] = vertices
                mesh['f'] = faces
                mesh['v_uv'] = uv_vertices
                mesh['f_uv'] = uv_faces

        :param camera_mv_bx4x4:
        :param texture_map:
        :return:
        '''
        # step1: transform to clip space
        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(
            camera_mv_bx4x4) else camera_mv_bx4x4
        v_pos = xfm_points(mesh['v'], mtx_in)  # Rotate it to camera coordinates
        v_pos_clip = self.camera.project(v_pos)  # Projection in the camera
        # step2: Rasterization
        rast = rasterize(self.ctx, v_pos_clip, mesh['f'], resolution)
        # step3: Interpolation
        rast_mask = rast[:, :, :, 3]
        #pdb.set_trace()
        #rast_patch = rast[:, 400:800, 400:800, :].contiguous()
        uv_view, _ = interpolate(mesh['v_uv'][None, ...], rast, mesh['f_uv'])

        # pdb.set_trace()
        # step4: sampling the texture image
        texture_image = texture_sample(texture_map, uv_view)
        alpha = (rast[..., 3:] > 0).float()
        texture_image_rgba = torch.cat([texture_image, alpha], dim=-1)

        return texture_image_rgba

# define some utils for sampling camera positions

def sample_fibonacci_sphere(n_views):
    # returns Nx3 unit vectors
    indices = np.arange(0, n_views, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n_views)
    theta = math.pi * (1 + 5**0.5) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    dirs = np.stack([x, y, z], axis=1)  # (N, 3)
    return dirs

def direction_to_angles(dirs):
    # dirs: (N, 3)
    rotation = np.degrees(np.arctan2(dirs[:, 0], dirs[:, 2])) % 360
    elevation = np.degrees(np.arcsin(dirs[:, 1]))  # y = sin(elevation)
    return rotation, elevation

def get_view_from_texture(renderer:TextureRender, data_dir, mesh_id, 
                            camera_save_dir, image_save_dir, view_num, device = device,
                            resolution = 512):
    '''
    Rendering the 3d mesh given multiple viewpoints
    Implementation is based on nvdiffrast
    '''

    # specify loading & saving paths
    # camera_save_dir = os.path.join(render_save_dir, 'camera', mesh_id)
    # img_save_dir = os.path.join(render_save_dir, 'img', mesh_id)
    # make_file(camera_save_dir)
    make_file(image_save_dir)
    make_file(camera_save_dir)
    mesh_path = os.path.join(data_dir, mesh_id, model_filename)
    texture_map_path = os.path.join(data_dir, mesh_id, 'texture.png')

    # generate camera position data
    # rotation_angle_list = np.random.rand(view_num)
    # elevation_angle_list = 2*np.random.rand(view_num)-1 #(-1,1)
    # rotation_camera = rotation_angle_list * 360
    # elevation_camera = elevation_angle_list * 30

    dirs = sample_fibonacci_sphere(view_num)
    rotation_camera, elevation_camera = direction_to_angles(dirs)
    ro_save_path = os.path.join(camera_save_dir, 'rotation')
    ele_save_path = os.path.join(camera_save_dir, 'elevation')
    # make_file(ro_save_path)
    # make_file(ele_save_path)
    np.save(ro_save_path, rotation_camera)
    np.save(ele_save_path, elevation_camera)

    # load mesh & texture data
    image = PIL.Image.open(texture_map_path)
    image = np.array(image) / 255
    color, mask = image[..., :3], image[..., 3:]
    color = np.ascontiguousarray(color[::-1, :, :])
    texture_map = torch.from_numpy(color).to(device).unsqueeze(0).float()

    tm_mesh = trimesh.load(mesh_path, process=False)
    tm_mesh = normalize_mesh(tm_mesh)

    mesh = {
    'v': torch.tensor(tm_mesh.vertices, dtype=torch.float32, device=device),
    'f': torch.tensor(tm_mesh.faces, dtype=torch.int32, device=device),
    'v_uv': torch.tensor(tm_mesh.visual.uv, dtype=torch.float32, device=device),
    'f_uv': torch.tensor(tm_mesh.faces, dtype=torch.int32, device=device)
    }

    cam_mv = align_blender_imgs(rotation_camera, elevation_camera)

    # rendering
    for i, cam_single_view in enumerate(cam_mv):
        texture_image = renderer.texture_to_view(mesh, cam_single_view.unsqueeze(0).float(), texture_map, resolution=resolution)
        texture_save = texture_image.permute(0, 3, 1, 2).detach().cpu()
        torchvision.utils.save_image(texture_save, os.path.join(image_save_dir,'%03d.png' % i))

    return None

'''
main uv mapping function
'''

def make_uv_map(file_name, category):
    '''
    :param file_name:
    :return : uv_texture
            uv_position
            uv_mask
            new_obj (has been normalized)
    '''

    logging.info(f"Start processing {file_name}...")

    mesh_path = os.path.join(mesh_dir, file_name, model_filename)
    camera_path = os.path.join(save_dir, 'rendered_views', category, 'camera', file_name)
    image_path = os.path.join(save_dir, 'rendered_views', category, 'img', file_name)
    coarse_model_path = os.path.join(save_dir, 'coarse_model', category)
    uv_model_path = os.path.join(save_dir, 'uv_model_%d' % uv_resolution, category, file_name)
    #save_file = os.path.join(save_dir, file_name)
    

    '''stored data structure
    ├── rendered_views
            ├── category
                ├── camera
                    ├── ${model_id}
                        ├── elevation.npy
                        ├── rotation.npy
                ├── img
                    ├── ${model_id}
                        ├── ${0view_num}.png
            ├── category
                ...
    ├── clip_image_data 
        ├── category
            ├── ${model_id}.png # reference/conditioning image
            ├── ${model_id}.pt # extracted embeddings from CLIP
        ├── category
            ...        
    ├── coarse_model
        ├── category
            ├── save_${fps_num}
                ├── ${model_id}.npz # fps point cloud data {points:... normal:... colors:...}
            ├── save_fpsuv_${fps_num}
        ├── category
            ...
    ├── final_split_files
        ├── category
    └── uv_model_512
        ├── category
            ├── model_id
                ├── uv_mask_${uv_res}.png
                ├── uv_position_${uv_res}.npz #point cloud data {xyz:... color:...}
                ├── uv_texture_${uv_res}.mtl
                ├── uv_texture_${uv_res}.obj
                ├── uv_texture_${uv_res}.png
                ├── uv_texture_hom_${uv_res}.png
    '''

    #make_file(save_file)

    # rendering -> store rendering cache in camera_dir / img_dir
    logging.info("Rendering from different views")
    T_Render = TextureRender()
    get_view_from_texture(renderer=T_Render, data_dir=mesh_dir, mesh_id=file_name,
                          camera_save_dir=camera_path, image_save_dir=image_path, view_num=view_num, resolution=render_res)

    logging.info("Loading mesh...")
    mesh = trimesh.load(mesh_path, force = 'mesh')
    mesh = normalize_mesh(mesh, mesh_scale=args.mesh_scale)
    mesh_v = torch.from_numpy(np.array(mesh.vertices)).to(device=device).float()
    mesh_f = torch.from_numpy(np.array(mesh.faces)).to(device=device).float()

    logging.info("Generating UV map with xatlas...")
    uvs, mesh_new_f, gb_info, mask, vmapping, indices, uvs_ori = xatlas_uvmap(ctx,
                                                                             mesh_v,
                                                                             mesh_f,
                                                                             resolution=uv_resolution)
    logging.info("Saving new OBJ with UVs...")
    # saving the new obj & mtl file
    make_file(uv_model_path)
    savemeshtes2(
        mesh_v.data.cpu().numpy(),
        uvs.data.cpu().numpy(),
        mesh_f.data.cpu().numpy(),
        mesh_new_f.data.cpu().numpy(),
        os.path.join(uv_model_path, 'uv_texture_%s.obj' % (uv_resolution))
    )

    logging.info("Loading render views")
    render_views = load_render_view(camera_path=camera_path,
                                    image_path = image_path, 
                                    device = device, 
                                    view_num=view_num)
    
    logging.info("Computing query positions")
    pos, pos_mask = get_query_position(render_views=render_views, 
                                        verts=mesh.vertices,
                                        faces=mesh.faces, 
                                        device=device, 
                                        render_res=render_res)

    logging.info("Filtering valid point cloud")
    xyz, color = del_invalid_pc(xyz=pos, color=render_views[None], mask=render_views['mask'])
    xyz = torch.from_numpy(xyz).to(device).float()
    color = torch.from_numpy(color).to(device).float()

    # non_zero_mask = ~(torch.all(color == 0.0, dim = 1)) #DEBUG
    # color = color[non_zero_mask]
    # xyz = xyz[non_zero_mask]

    pcd = {'color': color, 'xyz': xyz} # point cloud data

    tex_map = gb_info['pos'].squeeze(0)
    normal_map = gb_info['normal'].squeeze(0)

    img_pos = np.asarray(tex_map.data.cpu().numpy(), dtype=np.float32)
    mask = np.sum(img_pos.astype(float), axis=-1, keepdims=True)
    mask = (mask == 0.0).astype(float)

    gb_dir = {'xyz': tex_map.view(-1, 3).float(), 'mask': torch.from_numpy(mask).to(device).view(-1),
              'normal': normal_map.view(-1, 3).float()}
    #pdb.set_trace()

    logging.info("Projecting point cloud to texture map...")
    tex_color = get_pos_textures(None, gb_dir, pcd, device)
    color_map = tex_color.view(uv_resolution, uv_resolution, 3)
    # pdb.set_trace()

    #saving uv texture image
    logging.info("Saving UV texture image...")
    img = np.asarray(color_map.data.cpu().numpy(), dtype=np.float32)
    img = img * 255
    img = img.clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
        os.path.join(uv_model_path, 'uv_texture_%s.png' % (uv_resolution))
        )

    #saving uv texture hom
    kernel = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    hom_img = img * (1 - mask) + dilate_img * mask
    hom_img = hom_img.clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(np.ascontiguousarray(hom_img[::-1, :, :]), 'RGB').save(
        os.path.join(uv_model_path, 'uv_texture_hom_%s.png' % (uv_resolution))
        )

    logging.info("Saving UV position point cloud...")
    #pc_path = '%s/uv_position_%s' % (save_file, uv_resolution)
    pc_path = os.path.join(uv_model_path, 'uv_position_%s' % (uv_resolution))
    points = np.asarray(tex_map.view(-1, 3).data.cpu().numpy(), dtype=np.float16)
    normals = np.asarray(normal_map.view(-1, 3).data.cpu().numpy(), dtype=np.float16)
    np.savez(pc_path, points=points, normals=normals)

    logging.info("Saving UV mask...")
    mask_save = (255*(1 - mask[::-1, :, 0])).clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(np.ascontiguousarray(mask_save), 'L').save(
        os.path.join(uv_model_path, 'uv_mask_%s.png' % (uv_resolution))
        )


    #TODO: fps sampling on created color map -> coarse_model/save_4096/...npz
    # points: 4096*3
    # color: 4096*3
    # normal: 4096*3
    logging.info('Creating fps points, colors, normals...')
    fps_points, fps_colors, fps_normals, color_map_fps_img = get_fps_points(gb_dir, tex_color)
    
    # saving npz file : save_4096.npz
    logging.info('Saving fps data...')
    fps_npz_save_path = os.path.join(coarse_model_path, 'save_%d' % fps_num, '%s.npz' % file_name)
    make_file(fps_npz_save_path)
    np.savez(
        fps_npz_save_path,
        points = fps_points.cpu().numpy().astype(np.float32),
        color = fps_colors.cpu().numpy().astype(np.float32),
        normal = fps_normals.cpu().numpy().astype(np.float32)
    )

    # saving fps uv texture map
    color_map_fps_save_path = os.path.join(coarse_model_path, 'save_fpsuv_%d' % fps_num, '%s.png' % file_name)
    make_file(color_map_fps_save_path)
    PIL.Image.fromarray(np.ascontiguousarray(color_map_fps_img[::-1, :, :]), 'RGB').save(
        color_map_fps_save_path
    )

    logging.info("Finished processing")

def is_preprocessing_complete(category, model_id, save_dir = save_dir, uv_resolution = uv_resolution, fps_num = fps_num):
    """
    Check whether all critical output files for a mesh preprocessing run exist.
    """
    check_list = [
        # Camera
        os.path.join(save_dir, 'rendered_views', category, 'camera', model_id, 'elevation.npy'),
        os.path.join(save_dir, 'rendered_views', category, 'camera', model_id, 'rotation.npy'),
        # At least one rendered view image
        os.path.join(save_dir, 'rendered_views', category, 'img', model_id, '000.png'),
        # CLIP conditioning image + feature
        os.path.join(save_dir, 'clip_image_data', category, f'{model_id}.png'),
        os.path.join(save_dir, 'clip_image_data', category, f'{model_id}.pt'),
        # Coarse model npz
        os.path.join(save_dir, 'coarse_model', category, f'save_{fps_num}', f'{model_id}.npz'),
        os.path.join(save_dir, 'coarse_model', category, f'save_fpsuv_{fps_num}', f'{model_id}.png'),
        # UV model outputs (6 files minimum)
        os.path.join(save_dir, f'uv_model_{uv_resolution}', category, model_id, f'uv_texture_{uv_resolution}.png'),
        os.path.join(save_dir, f'uv_model_{uv_resolution}', category, model_id, f'uv_texture_hom_{uv_resolution}.png'),
        os.path.join(save_dir, f'uv_model_{uv_resolution}', category, model_id, f'uv_mask_{uv_resolution}.png'),
        os.path.join(save_dir, f'uv_model_{uv_resolution}', category, model_id, f'uv_position_{uv_resolution}.npz'),
        os.path.join(save_dir, f'uv_model_{uv_resolution}', category, model_id, f'uv_texture_{uv_resolution}.obj'),
        os.path.join(save_dir, f'uv_model_{uv_resolution}', category, model_id, f'uv_texture_{uv_resolution}.mtl'),
    ]
    return all(os.path.isfile(p) for p in check_list)


def generate_dataset(selected_models,
                     flag_csv_path = flag_csv_path,
                     save_dir = save_dir,
                     uv_resolution = uv_resolution,
                     fps_num = fps_num                     
                     ):
    #  Load or create the flag CSV
    if os.path.exists(flag_csv_path):
        df = pd.read_csv(flag_csv_path)
    else:
        df = pd.DataFrame({
                'category': pd.Series(dtype='str'),
                'model_id': pd.Series(dtype='str'),
                'is_complete': pd.Series(dtype='bool'),
                'timestamp': pd.Series(dtype='str'),
                            })


    # add pending mesh that need to be preprocessed
    for category, models in selected_models.items():
        for model in models:
            model_id = model['model_id']
            if not ((df['category'] == category) & (df['model_id'] == model_id)).any():
                df = pd.concat([df, pd.DataFrame([{
                    'category': category,
                    'model_id': model_id,
                    'is_complete': False,
                    'timestamp': ''
                }])], ignore_index=True)

    for idx, row in df.iterrows():
        category = row['category']
        model_id = row['model_id']
        if (is_preprocessing_complete(category, model_id)) or (row['is_complete'] == True):
            # Already done (maybe externally)
            df.at[idx, 'is_complete'] = True
            #df.at[idx, 'timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df.to_csv(flag_csv_path, index=False)
            logging.warning(f"[SKIP] {category}/{model_id} already complete")
            continue
        try:
            t0 = time.time()
            # main function for creating rendered views and uv maps
            make_uv_map(model_id, category)
            # main function for clip feature extraction
            extract_and_save_clip_embedding(data_dir = mesh_dir,
                                            save_dir = save_dir,
                                            mesh_id = model_id,
                                            category = category)
            t1 = time.time()
            if is_preprocessing_complete(category, model_id):
                # update csv
                df.at[idx, 'is_complete'] = True
                df.at[idx, 'timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                df.to_csv(flag_csv_path, index=False)
                logging.warning(f"[DONE] {category}/{model_id}, It takes {round(t1-t0,4)} seconds to process")
            else:
                logging.warning(f"[FAIL] Output format is wrong: {category}/{model_id}")
        except Exception as e:
            logging.warning(f"[FAIL] {category}/{model_id} - {e}")

if __name__ == '__main__':

    set_logging(log_path='./log/render_log.log')
    selected_models = filter_by_supercategory(category_list, length=None)
    generate_dataset(selected_models=selected_models)
    
    #print(category_list)





