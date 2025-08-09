from torch.utils import data
import os
import logging
import numpy as np
import torch
import PIL
from skimage import measure,color
import pandas as pd

logger = logging.getLogger(__name__)

class CameraField:

    def __init__(self, transform=None):
        self.transform = transform
    
    def load(self, camera_path):
        '''
        camera_path : save_dir/rendered_views/category/camera/model/... elevation.npy or rotation.npy

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
        
        data_out : {
            'rotation:' num_views np.array...
            'elevation:' num_views
        }
        '''
        rotation_camera = np.load(os.path.join(camera_path, 'rotation.npy'))
        elevation_camera = np.load(os.path.join(camera_path, 'elevation.npy'))
        assert len(rotation_camera) == len(elevation_camera)
        data_out = {}
        data_out['rotation'] = rotation_camera
        data_out['elevation'] = elevation_camera
        if self.transform is not None:
            data_out = self.transform(data_out)

        return data_out

def get_camera_field(args):
    field = CameraField()
    return field

class MeshField:

    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path):
        # here just return the path of mesh
        file_path = os.path.join(model_path, self.file_name)
        # mesh length is not the same, get problem with pytorch dataloader
        return file_path

def get_mesh_field(args):
    field = MeshField(
        args.mesh_file,
    )
    return field

class uvField:

    def __init__(self, mask_name, texture_name, position_name, transform=None):
        
        '''
        example input:
        mask_file: 'uv_mask_512.png'
        texture_file: 'uv_texture_512.png'
        position_file: 'uv_position_512.npz'
        
        '''
        self.mask_name = mask_name
        self.texture_name = texture_name
        self.position_name = position_name
        self.transform = transform
    
    def load(self, uv_path):
        '''
        uv_path: uv_folder/category/model/...
        '''
        uv_mask_file = os.path.join(uv_path, self.mask_name)
        uv_texture_file = os.path.join(uv_path, self.texture_name)

        # point cloud data
        uv_pos_file = os.path.join(uv_path, self.position_name)
        pointcloud_dict = np.load(uv_pos_file)
        points = pointcloud_dict['points'].astype(np.float32)
        normal = pointcloud_dict['normals'].astype(np.float32)
        normal[np.isnan(normal)] = 0

        # mask map
        uv_mask_image = PIL.Image.open(uv_mask_file)
        uv_mask_image = np.array(uv_mask_image) / 255

        # texture map
        uv_texture_image = PIL.Image.open(uv_texture_file)
        uv_texture_image = np.array(uv_texture_image) / 255
        #normalize to [-1, 1]
        uv_texture_image = uv_texture_image*2-1
                
        cluster_label = measure.label(uv_mask_image, connectivity=1)
        data_out = {}

        data_out['texture'] = uv_texture_image
        data_out['mask'] = uv_mask_image
        data_out['position'] = points
        data_out['normal'] = normal
        data_out['cluster_label'] = cluster_label

        if self.transform is not None:
            data_out = self.transform(data_out)

        return data_out

def get_uv_field(args):
    field = uvField(mask_name=args.mask_file,
                    texture_name=args.texture_file,
                    position_name=args.position_file,
                    )
    return field

# def get_coarse_map(file_path):
#     uv_texture_image = PIL.Image.open(file_path)
#     uv_texture_image = np.array(uv_texture_image) / 255

#     # normalize to [-1, 1]
#     uv_texture_image = uv_texture_image * 2 - 1

#     return uv_texture_image

def get_fps_point_info(file_path):
    pointcloud_dict = np.load(file_path)
    points = pointcloud_dict['points'].astype(np.float32) #4096*3
    normal = pointcloud_dict['normal'].astype(np.float32) # 4096*3
    color = pointcloud_dict['color'].astype(np.float32) # 4096*

    normal[np.isnan(normal)] = 0
    color = color*2-1

    return color, points, normal

class CoarseBaseDataset(data.Dataset):

    def __init__(self, split, args, mode):
        '''
        split:
            data_split:
                train_split: train_text
                val_split: test_text
                test_split: test_text
        args: see Point-UV-Diffusion\configs\datamodule\image_coarse_stage.yaml
        mode: exteranl parameter, 'train', 'val', or 'test'

        batch output structure:

        Key: input
            ├─ texture: Tensor torch.Size([2, 512, 512, 3]), dtype=torch.float64
            ├─ mask: Tensor torch.Size([2, 512, 512]), dtype=torch.float64
            ├─ position: Tensor torch.Size([2, 262144, 3]), dtype=torch.float32
            ├─ normal: Tensor torch.Size([2, 262144, 3]), dtype=torch.float32
            ├─ cluster_label: Tensor torch.Size([2, 512, 512]), dtype=torch.int64
            ├─ fps_color: Tensor torch.Size([2, 4096, 3]), dtype=torch.float32
            ├─ fps_points: Tensor torch.Size([2, 4096, 3]), dtype=torch.float32
            ├─ fps_normal: Tensor torch.Size([2, 4096, 3]), dtype=torch.float32
            ├─ clip_condition: Tensor torch.Size([2, 1, 768]), dtype=torch.float16
        Key: mesh_file
            └─ <class 'list'> -> ['/scratch/leuven/375/vsc37593/3D-FUTURE-Preprocessed-24Views/uv_model_512/Chair/064159ad-2984-3042-
        Key: name
            └─ <class 'list'> -> ['064159ad-2984-3042-a961-7ec585e55ee9', '6a20799e-8031-4184-9dd8-ae17ca36effc']
        Key: category
        └─ <class 'list'> -> ['Chair', 'Chair']
        '''
        self.mode = mode
        self.split = split
        self.model_folder = args.model_folder
        self.uv_folder = args.uv_folder
        self.uv_field = get_uv_field(args)
        self.coarse_point_folder = args.coarse_point_folder
        #self.text_folder = args.text_folder
        self.image_folder = args.image_folder
        #self.condition_type = args.condition_type
        self.clip_condition_folder = self.image_folder # we dont need condition_type here

        self.camera_field = get_camera_field(args)
        self.render_folder = getattr(args, 'render_folder', None)  # returns None if 'render_folder' attribute does not exist
        self.mesh_field = get_mesh_field(args)
        self.flag_csv = args.flag_csv
        self.args = args 
        self.get_all_models(args) 
    
    def get_all_models(self, args):
        '''
        TODO: Need modifications according to flag csv

        expected return:
            [
                {'category': xxx, 'model': model_id},
                {'category': xxx, 'model': model_id},
                ...
            ]
        '''
        split = self.split
        category = args.category
        self.models = []

        flag_df = pd.read_csv(self.flag_csv) # NEW configuration item!!!

        flag_df['category'] = flag_df['category'].astype(str)
        category_str = str(category)

        if category_str not in flag_df['category'].unique():
            raise ValueError(f'{category_str} not found in flag_csv')



        valid_models = flag_df[
            (flag_df['category'] == category_str) & (flag_df['is_complete'] == True)
        ]['model_id'].tolist()

        if split == 'all':
            subpath = os.path.join(self.model_folder, str(category))
            if not os.path.isdir(subpath):
                logger.warning(f'Category directory "{subpath}" does not exist.')
                return
            all_models = [
                d for d in os.listdir(subpath) if os.path.isdir(os.path.join(subpath, d) and d in valid_models)
            ]
        else:
            # need load split file in that case
            split_file = os.path.join(args.split_files, str(category), split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            models_c = [m for m in models_c if m != '']
            all_models = [m for m in models_c if m in valid_models]
        
        self.models +=[{'category': category, 'model': m} for m in all_models]

        if self.mode == 'test' and args.test_samples:
            self.models = self.models[:args.test_samples]

    def __len__(self):
        return len(self.models)
    
    def __getitem__(self, idx):

        category = self.models[idx]['category']
        model = self.models[idx]['model']
        uv_path = os.path.join(self.uv_folder, str(category), model)
        mesh_path = os.path.join(self.uv_folder, str(category), model)

        data = {}
        #uv_model
        uv_data = self.uv_field.load(uv_path)
        mesh_file = self.mesh_field.load(mesh_path)

        # fps point cloud
        # TODO: need further check
        # coarse_point_folder: 
        # .../test_preprocess_output/coarse_model/Chair/save_4096
        color, points, normal = get_fps_point_info(
            os.path.join(self.coarse_point_folder, model+'.npz')
        )
        uv_data['fps_color'] = color
        uv_data['fps_points'] = points
        uv_data['fps_normal'] = normal

        #clip embeddings
        clip_condition = torch.load(
            os.path.join(self.clip_condition_folder, model + '.pt'),
            map_location=torch.device('cpu')
        )
        uv_data['clip_condition'] = clip_condition

        data['input'] = uv_data
        data['mesh_file'] = mesh_file
        data['name'] = model
        data['category'] = category

        if self.render_folder is not None:
            # exmaple: .../test_preprocess_output/rendered_views/Chair/camera/0de60f95-775b-3a78-bd06-e2b04a309c4f/...
            camera_path = os.path.join(self.render_folder, str(category), 'camera', model)
            camera = self.camera_field.load(camera_path)
            data['camera'] = camera
        
        return data


if __name__ == "__main__":

    pass