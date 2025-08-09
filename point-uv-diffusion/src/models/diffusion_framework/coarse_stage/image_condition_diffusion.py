import torch
import torch.nn.functional as F
import logging
import functools
from typing import Any, List
from torchmetrics import MaxMetric, MeanMetric
from timm.utils import ModelEmaV2
from geometry_tools.sample_camera_distribution import *
from geometry_tools.camera import *
from geometry_tools.rasterize_mesh import *
from .openai_diffusion.diffusion.resample import create_named_schedule_sampler, LossAwareSampler, UniformSampler
from .openai_diffusion.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from .openai_diffusion.models.configs import MODEL_CONFIGS, model_from_config
from models.module_utils.point_sample_gather import pc_to_uv
from .base_diffusion import DiffusionModule as BaseModule

class DiffusionModule(BaseModule):
    '''
    diffusion framework of coarse stage texture generation
    '''


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def forward(self, x, fps_pos_normal, shape_cond, text):
        '''
        basic forward pass in diffusion framework
        # 1. Sample timesteps t and importance weights
        # 2. Call diffusion.training_losses() to compute denoising loss
        # 3. If using LossAwareSampler, update its internal stats
        # 4. Return weighted loss
        '''
        t, weights = self.schedule_sampler.sample(x.shape[0], x.device)
        model_kwargs = {"fps_cond": fps_pos_normal, "shape_cond": shape_cond, "text": text}

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.net,
            x,
            t,
            model_kwargs=model_kwargs,

        )
        losses, pred, target, x_noise = compute_losses()
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()

        return loss, pred, x_noise
    
    def step(self, batch: Any):
        '''
        # Process batch into x (ground truth texture) and cond (condition maps):
        # Includes UV position maps, normals, masks, and coarse maps

        input batch format:

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
        # load ground-turth texture map
        x = batch['input']['texture'].to(device=self.device)
        x = x.permute(0, 3, 1, 2).float()  #[B, H, W, 3] -> [B, 3, H, W]

        # load and reshape position map
        cond_pos = batch['input']['position'].to(device=self.device) # [B, H*W, 3]
        b, c, h, w = x.size()
        cond_pos = cond_pos.reshape(b, h, w, c)
        cond_pos = torch.flip(cond_pos, dims=[1]) # important!
        cond_pos = cond_pos.permute(0, 3, 1, 2).float() # [B, 3, H, W]

        # load and reshape normal map
        cond_normal = batch['input']['normal'].to(device=self.device)
        b, c, h, w = x.size()
        cond_normal = cond_normal.reshape(b, h, w, c)
        cond_normal = torch.flip(cond_normal, dims=[1])  # important!
        cond_normal = cond_normal.permute(0, 3, 1, 2).float() # [B, 3, H, W]

        # load mask map
        mask = batch['input']['mask'].to(device=self.device)
        mask = mask.unsqueeze(1).float()

        # load clip embedding
        text = batch['input']['clip_condition'].to(device=self.device) # here text is conditioning image in our case
        text = text.squeeze(1) # squeeze!!! not unsqueeze

        shape_cond = torch.cat([cond_pos, cond_normal, mask], dim=1) #  Shape: [B, 7, 512, 512]

        # load fps point cloud data
        fps_color = batch['input']['fps_color'].to(device=self.device).permute(0, 2, 1) # N_Batch * Channel * N_points
        fps_pos = batch['input']['fps_points'].to(device=self.device).permute(0, 2, 1)
        fps_normal = batch['input']['fps_normal'].to(device=self.device).permute(0, 2, 1)
        fps_pos_normal = torch.cat([fps_pos, fps_normal], dim=1) # 	[B, 6, 4096]

        return fps_color, fps_pos_normal, shape_cond, text
    
    def training_step(self, batch: Any, batch_idx: int):

        x, fps_pos_normal, shape_cond, text = self.step(batch)
        #NOTE: model is trained to directly predict the point color tensor!!!
        # NOT noise!
        loss, pred, x_noise = self.forward(x, fps_pos_normal, shape_cond, text)
        sample = (1 + pred) / 2 # [-1,1] -> [0,1]

        mask = batch['input']['mask'].to(device=self.device)
        mask = mask.unsqueeze(1).float() # enforce mask map on the predicted texture map

        colors = pc_to_uv(
            sample.permute(0, 2, 1),                            # [B, N, 3]
            fps_pos_normal[:, :3, ...].permute(0, 2, 1),        # [B, N, 3]
            shape_cond[:, :3, ...].permute(0, 2, 3, 1)          # [B, H, W, 3]
        )
        colors = colors.permute(0, 3, 1, 2) * mask #  enforce mask map on the predicted texture map

        return {"loss": loss}, {"image": colors}
    
    def test_step(self, batch: Any, batch_idx: int):
        '''
        main inference step -> coarse texture map
        '''


        x, fps_pos_normal, shape_cond, text = self.step(batch)
        b, c, n = x.size()
        batch_shape = (b, c, n)


        # Use diffusion to sample x_0 from noise
        model_kwargs = {"fps_cond": fps_pos_normal, "shape_cond": shape_cond, "text": text}
        sample = self.diffusion.p_sample_loop(
            self.net,
            batch_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
        )
        sample = (1+sample)/2

        mask = batch['input']['mask'].to(device=self.device)
        mask = mask.unsqueeze(1).float()

        colors = pc_to_uv(sample.float().permute(0, 2, 1), 
                          fps_pos_normal[:, :3, ...].float().permute(0, 2, 1), 
                          shape_cond[:, :3, ...].permute(0, 2, 3, 1))
        colors = colors.permute(0, 3, 1, 2) * mask

        obj_c = batch['category']
        obj_name = batch['name']

        return {"loss": None}, {"image": colors, "texture_map": colors, "obj_c": obj_c, "obj_name": obj_name, "mask": mask}