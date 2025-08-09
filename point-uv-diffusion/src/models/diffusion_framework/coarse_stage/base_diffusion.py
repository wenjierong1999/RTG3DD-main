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
from .openai_diffusion.diffusion.resample import create_named_schedule_sampler, LossAwareSampler
from .openai_diffusion.diffusion.configs import diffusion_from_config

class DiffusionModule:

    def __init__(self,
                 diffusion_config: dict, #config dictionary to build the GaussianDiffusion, see diffusion_from_config
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 device = 'cuda',
                 model_ema_decay = None, # wraps the model in EMA to smooth parameter updates over time
                 local_rank = -1,
                 schedule_sampler = 'uniform'):

        super().__init__()
        self.device = device
        diffusion = diffusion_from_config(diffusion_config)
        self.diffusion = diffusion # KEY !!!!!!!!
        self.schedule_sampler = create_named_schedule_sampler(schedule_sampler, diffusion)
        self.net = net
        self.net.to(device)
        self.net_ema = None
        if model_ema_decay:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.net_ema = ModelEmaV2(
                self.net,
                decay=model_ema_decay,
            )
            print("Using EMA with decay = %.8f" % model_ema_decay)
        
        self.net_without_ddp = self.net

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        self.optimizer = optimizer(params=self.net.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)

    def forward(self, x: torch.Tensor, cond_Pos: torch.Tensor):
        '''
        basic forward pass in diffusion framework
        # 1. Sample timesteps t and importance weights
        # 2. Call diffusion.training_losses() to compute denoising loss
        # 3. If using LossAwareSampler, update its internal stats
        # 4. Return weighted loss
        '''
        # sample a timestep t for each item in the batch
        t, weights = self.schedule_sampler.sample(x.shape[0], x.device)

        model_kwargs = {"cond": cond_pos}

        # see 
        compute_losses = functools.partial(
            self.diffusion.training_losses,   # Main diffusion loss function
            self.net,                         # The neural network model
            x,                                # Ground-truth texture
            t,                                # Sampled timestep per sample
            model_kwargs=model_kwargs         # Conditioning input
        )
        losses, pred, target, x_noise = compute_losses()
        # losses: {'loss':..., 'mse':..., 'vb':...}
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
        cond_pos = batch['input']['position'].to(device=self.device)
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

        # load coarse texture map 
        coarse_map = batch['input']['coarse_map'].to(device=self.device)
        coarse_map = coarse_map.permute(0, 3, 1, 2).float()

        # concatenate all shape conditions
        # [B, 3+3+3+1 = 10, H, w]
        cond = torch.cat([coarse_map, cond_pos, cond_normal, mask], dim=1)

        # Return:
        # - x: the target texture
        # - cond: the full condition tensor for guided generation
        # - coarse_map: for possible logging/visualization

        # NOTE: this is base diffusion class, we need specific modifications for coarse/fine stage
        return x, cond, coarse_map
    
    def training_step(self, batch: Any, batch_idx: int):

        x, cond, coarse_map = self.step(batch)
        loss, pred, x_noise = self.forward(x, cond)

        # Construct an image for visualization (tensorboard or logs)
        # Concatenate prediction, GT texture, noise, and coarse input
        # Shape: [B, 3, H, 4W] — i.e. 4 images side-by-side
        # ------------------------------------------------------------------
        image = torch.cat([pred, x, x_noise, coarse_map], dim=-1)
        image = (image + 1) / 2.0  # Rescale from [-1,1] to [0,1] for image display
        return {"loss": loss}, {"image": image}

    
    def test_step(self, batch: Any, batch_idx: int):
        '''
        Basic inference process
        '''
        x, cond, coarse_map = self.step(batch)

        sample_fn = self.diffusion.p_sample_loop # specify denoising chain
        b, c, h, w = x.size()
        batch_shape = (b, c, h, w)

        model_kwargs = {"cond": cond}
        sample = sample_fn(
            self.net,
            batch_shape,
            clip_denoised=False,        # don't clip to [-1,1] inside the model
            model_kwargs=model_kwargs   # condition input
        )


        mask = cond[:, -1:, ...]
        sample = (1+sample)/2 * mask

        cond_pos = cond[:, 3:6, ...]
        image = torch.cat([sample, (1+coarse_map)/2, cond_pos+0.5], dim=-1)

        obj_c = batch['category']
        obj_name = batch['name']

        return {"loss": None}, {"image": image, "texture_map": sample, "obj_c": obj_c, "obj_name": obj_name, "mask": mask}

if __name__ == "__main__":
    pass