import random
from typing import Any, List
import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)
import torch
import numpy as np
from geometry_tools.sample_camera_distribution import *
from geometry_tools.camera import *
from geometry_tools.rasterize_mesh import *
from nvdiffrast import torch as dr
from .base_diffusion import DiffusionModule as BaseModule

class DiffusionModule(BaseModule):

    def __init__(self, render_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_renderer(render_config)

    def build_renderer(self, render_config):
        '''
        initialize a differential render with the given config
        default render_config:
            view_num: 4
            patch_size: 224 #size of the cropped patch from the rendered uv view
            rast_resolution: 1024 # resolution used in rasterization
            render_loss:
                _target_: src.models.loss_utils.CombinedLoss.CombinedLoss
                loss_classes:
                - _target_: src.models.loss_utils.losses.L1Loss
                loss_weights: [ 1 ]
            render_weight: 1.0

        '''
        self.view_num = render_config.view_num
        self.patch_size = render_config.patch_size
        self.render_loss = render_config.render_loss
        self.render_weight = render_config.render_weight
        self.rast_resolution = render_config.rast_resolution
        self.ctx = dr.RasterizeCudaContext(device=self.device)
        self.projection_mtx = None
        fovy = np.arctan(32 / 2 / 35) * 2
        fovyangle = fovy / np.pi * 180.0
        dmtet_camera = PerspectiveCamera(fovy=fovyangle, device=self.device)
        self.camera = dmtet_camera
    
    def training_step(self, batch: Any, batch_idx: int):

        # Step 1: prepare input and condition
        x, cond, coarse_map = self.step(batch)  # x: GT texture map, cond: concat of coarse/pos/normal/mask
        loss, pred, x_noise = self.forward(x, cond)  # Run diffusion forward (compute denoise loss)

        # Step 2: mask: valid region (0 = invalid bg, 1 = valid region)
        mask = batch['input']['mask'].to(device=self.device).unsqueeze(1).float()  # shape: (B, 1, H, W)

        # Step 3: normalize prediction and GT to [0, 1]
        pred = (1 + pred) / 2
        x = (1 + x) / 2
        # Step 4: dilation to help boundary rendering
        dilated_pred = F.max_pool2d(pred, kernel_size=3, stride=1, padding=1)
        dilated_gt = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

        # Keep original values inside the mask, use dilated outside
        dilated_pred = dilated_pred * (1 - mask) + mask * pred
        dilated_gt = dilated_gt * (1 - mask) + mask * x

        # Prepare to accumulate rendering loss if needed
        render_loss_list = []
        pred_images = []
        gt_images = []

        # Step 5: if using rendering supervision
        mesh_file = batch['mesh_file']
        camera = batch.get('camera')

        if self.render_weight != 0:
            for b, one_file in enumerate(mesh_file):
                # Step 5.1: determine camera angles (from input or randomly sampled)
                if camera is not None:
                    rotation_camera = camera['rotation'][b]
                    elevation_camera = camera['elevation'][b]
                    indices = np.random.randint(rotation_camera.shape[0], size=self.view_num)
                    rotation_camera = rotation_camera[indices]
                    elevation_camera = elevation_camera[indices]
                else:
                    rotation_camera = np.random.rand(self.view_num) * 360
                    elevation_camera = np.random.rand(self.view_num) * 30
                # Step 5.2: convert angles to view matrices
                cam_mv = align_blender_imgs(rotation_camera, elevation_camera, self.device)

                #print(f'DEBUG: shape of cam_mv {cam_mv.shape}')

                # Step 5.3: load mesh from file
                mesh = self.load_one_mesh(one_file)

                # Step 5.4: render each view and compute image loss
                for cam_single_view in cam_mv:
                    render_loss_one, pred_image, gt_image = self.render_loss_fn(
                        mesh,
                        cam_single_view.unsqueeze(0).float(),
                        dilated_pred[b],
                        dilated_gt[b]
                    )
                    render_loss_list.append(render_loss_one)
                    pred_images.append(pred_image)
                    gt_images.append(gt_image)

                # here we modify the render_loss_fn to support batched camera_view input
                # render_loss_one, pred_images_bv, gt_images_bv = self.render_loss_fn(
                #         mesh,
                #         cam_mv.float(),  # (V, 4, 4)
                #         dilated_pred[b].unsqueeze(0).repeat(self.view_num, 1, 1, 1),  # (V, 3, H, W)
                #         dilated_gt[b].unsqueeze(0).repeat(self.view_num, 1, 1, 1)
                #     )
                # print(f'DEBUG: shape of render_loss_one {render_loss_one.shape}')

                # render_loss_list.append(render_loss_one)
                # pred_images.extend(pred_images_bv)
                # gt_images.extend(gt_images_bv)

            # Step 6: aggregate render loss over all views
            render_loss = torch.stack(render_loss_list, dim=0)
            render_loss = self.render_weight * torch.mean(render_loss)

            # Visualization image
            image = torch.cat([
                pred,        # predicted texture
                x,           # GT texture
                (x_noise + 1) / 2.0,  # noisy input
                (coarse_map + 1) / 2.0,  # coarse texture
                dilated_gt   # GT after dilation
            ], dim=-1)

            # Concatenate all rendered views
            render_image = torch.cat(pred_images + gt_images, dim=-1)

            # Combine reconstruction + render loss
            loss = loss + render_loss

            return {"loss": loss, "render_loss": render_loss}, {"image": image, "render_image": render_image}

        else:
            # No rendering supervision
            image = torch.cat([
                pred, x, (x_noise + 1) / 2.0, (coarse_map + 1) / 2.0, dilated_gt
            ], dim=-1)

            return {"loss": loss, "render_loss": 0}, {"image": image}
            
    def render_loss_fn(self, mesh, camera_mv_bx4x4, pred_tex_map, gt_tex_map):
        '''
        Differentiable rendering of predicted and GT texture maps, and compute image-level loss.

        Args:
            mesh (dict): Contains geometry and UV data of the mesh.
            camera_mv_bx4x4 (Tensor): Camera view matrix (B, 4, 4). (4*4 camera project matrix)
            pred_tex_map (Tensor): Predicted texture map in UV space (C, H, W), values in [0,1].
            gt_tex_map (Tensor): Ground-truth texture map in UV space (C, H, W), values in [0,1].

        Returns:
            render_loss (Tensor): Scalar loss between predicted and GT rendered images.
            pred_image (Tensor): Rendered RGB image from predicted texture (B, 3, H, W).
            gt_image (Tensor): Rendered RGB image from GT texture (B, 3, H, W).
        '''
        # Step 1: compute per-pixel UV coordinates from the given camera view
        uv_view = self.get_uv_view(
            mesh,
            camera_mv_bx4x4,
            resolution=self.rast_resolution,
            patch=self.patch_size
        )  # shape: (B, H, W, 2)
        
        #print(f'DEBUG: shape of uv_view {uv_view.shape}') # (B, H, W, 2)

        # Step 2: flip vertically to align with UV convention
        pred_tex_map = torch.flip(pred_tex_map, dims=[1])
        gt_tex_map = torch.flip(gt_tex_map, dims=[1])

        #print(f'DEBUG: shape of pred_tex_map {pred_tex_map.shape}')

        # Step 3: rearrange UV texture maps from (C, H, W) → (B=1, H, W, C)
        pred_tex_map = pred_tex_map.permute(1, 2, 0).unsqueeze(0).contiguous()  # (1, H, W, C)
        gt_tex_map = gt_tex_map.permute(1, 2, 0).unsqueeze(0).contiguous()

        # pred_tex_map = pred_tex_map.permute(0, 2, 3, 1).contiguous()  # (V, H, W, 3)
        # gt_tex_map = gt_tex_map.permute(0, 2, 3, 1).contiguous()      # (V, H, W, 3)

        # Step 4: render RGB images using nvdiffrast.texture from UV maps
        pred_image = dr.texture(pred_tex_map, uv_view)  # output: (1, H, W, 3)
        gt_image = dr.texture(gt_tex_map, uv_view).detach()  # detach to prevent gradient flow from GT

        # Step 5: rearrange to (B, C, H, W) format for loss computation
        pred_image = pred_image.permute(0, 3, 1, 2)
        gt_image = gt_image.permute(0, 3, 1, 2)

        # Step 6: compute differentiable image loss
        render_loss = self.render_loss(pred_image, gt_image)

        return render_loss, pred_image, gt_image

    def select_patch(self, rast, patch_size):
        '''
        cropping patches for computing patch-based rendering loss
        '''

        # rast shape: (B, H, W, 4), where the last channel indicates visibility mask (alpha)
        B, H, W, C = rast.size()
        rast_mask = rast[:, :, :, 3] # visible pixels
        rast_patchs = []
        for b in range(B):
            # Calculate mean mask values along height and width
            h_mask = torch.mean(rast_mask[b:b + 1], dim=2).cpu().numpy()
            w_mask = torch.mean(rast_mask[b:b + 1], dim=1).cpu().numpy()
            # Find the index range where mask > 0, meaning visible pixels exist
            h_idx = np.where(h_mask > 0)
            w_idx = np.where(w_mask > 0)
             # Compute bounding box of visible region
            h_min, h_max = h_idx[1][0], h_idx[1][-1]
            w_min, w_max = w_idx[1][0], w_idx[1][-1]
            # Ensure patch can fit within the visible bounding box
            h_min = min(h_min, H - patch_size)
            w_min = min(w_min, W - patch_size)
            h_max = max(h_max, h_min + patch_size)
            w_max = max(w_max, w_min + patch_size)
            # Randomly choose top-left corner of patch within the valid region
            h1 = random.randint(h_min, h_max - patch_size)
            w1 = random.randint(w_min, w_max - patch_size)

            # Crop patch from the raster
            rast_patch = rast[b:b + 1, h1:h1 + patch_size, w1:w1 + patch_size, :]
            rast_patchs.append(rast_patch)

        return torch.cat(rast_patchs, dim=0)

    def get_uv_view(self, mesh, camera_mv_bx4x4, resolution=1024, patch=False):
        '''
        compute uv coordinates from a given camera view

        :param mesh:
               mesh['v'] = vertices
                mesh['f'] = faces
                mesh['v_uv'] = uv_vertices
                mesh['f_uv'] = uv_faces

        :param camera_mv_bx4x4:
        :return:
        '''
        # step1: transform to clip space
        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(
            camera_mv_bx4x4) else camera_mv_bx4x4
        v_pos = xfm_points(mesh['v'], mtx_in)  # Rotate it to camera coordinates
        v_pos_clip = self.camera.project(v_pos)  # Projection in the camera
        # step2: Rasterization
        rast, _ = dr.rasterize(self.ctx, v_pos_clip, mesh['f'], (resolution, resolution))
        # step3: Interpolation
        if not patch:
            uv_view, _ = interpolate(mesh['v_uv'][None, ...], rast, mesh['f_uv'])
        else:
            rast_patch = self.select_patch(rast, self.patch_size).contiguous()
            uv_view, _ = interpolate(mesh['v_uv'][None, ...], rast_patch, mesh['f_uv']) # (B, H, W, 2)?

        return uv_view
    
    def get_uv_view_batched(self, mesh, camera_mv_bx4x4, resolution=1024, patch = False):
        """
        Batched version of get_uv_view, supporting multiple camera views for a single mesh.

        Args:
            mesh: dict containing v, f, v_uv, f_uv
            camera_mv_Vx4x4: Tensor of shape (V, 4, 4)
            resolution: int, rasterization resolution

        Returns:
            uv_view: (V, H, W, 2)
        """
        V = camera_mv_Vx4x4.shape[0]

        # 1. camera transform
        v_pos = xfm_points(mesh['v'].unsqueeze(0).expand(V, -1, -1), camera_mv_Vx4x4)  # (V, N, 3)
        v_pos_clip = self.camera.project(v_pos)  # (V, N, 4)

        # 2. rasterization
        rast, _ = dr.rasterize(self.ctx, v_pos_clip, mesh['f'], (resolution, resolution))  # (V, H, W, 4)

        # Step 3: patch crop if needed
        if patch:
            rast = self.select_patch(rast, self.patch_size)  # crop (V, patch, patch, 4)

        # 3. interpolate UV
        uv_view, _ = interpolate(mesh['v_uv'].unsqueeze(0).expand(V, -1, -1), rast, mesh['f_uv'])  # (V, H, W, 2)

        return uv_view

    def load_one_mesh(self, file_path):
        '''
        manually load obj file
        '''
        vertex_data = []
        face_data = []
        uv_vertex_data = []
        uv_face_data = []
        for line in open(file_path, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                vertex_data.append(v)

            elif values[0] == 'vt':
                vt = list(map(float, values[1:3]))
                uv_vertex_data.append(vt)

            elif values[0] == 'f':
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                uv_f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                uv_face_data.append(uv_f)

        vertices = torch.from_numpy(np.array(vertex_data)).to(self.device)
        faces = torch.from_numpy(np.array(face_data)).to(self.device)
        uv_vertices = torch.from_numpy(np.array(uv_vertex_data)).to(self.device)
        uv_faces = torch.from_numpy(np.array(uv_face_data)).to(self.device)

        mesh = {}
        mesh['v'] = vertices.float()
        mesh['f'] = faces.int() - 1
        mesh['v_uv'] = uv_vertices.float()
        mesh['f_uv'] = uv_faces.int() - 1

        return mesh