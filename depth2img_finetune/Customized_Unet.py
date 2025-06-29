from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch.nn as nn
#from diffusers import UNet2DConditionModel
from diffusers import UNet2DConditionModel
import torch

class UNetFuseTimeCamera(UNet2DConditionModel):

    def __init__(self, cam_dim=12, **kwargs):
        super().__init__(**kwargs)
        self.cam_dim = cam_dim
        self.camera_mlp = None  # lazy init

    def get_time_embed(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]):
        
        t_emb = super().get_time_embed(sample, timestep)  # 原始时间 embedding
        # lazy init camera mlp
        if self.camera_mlp is None:
            self.camera_mlp = nn.Sequential(
                nn.Linear(self.cam_dim, 128),
                nn.ReLU(),
                nn.Linear(128, t_emb.shape[-1])  # same dim as t_emb
            ).to(t_emb.device)
            self.camera_mlp = self.camera_mlp.to(dtype=torch.float32)

        if hasattr(self, "_cached_camera") and self._cached_camera is not None:
            cam_emb = self.camera_mlp(self._cached_camera.to(t_emb.device, dtype=t_emb.dtype))
            # print(cam_emb.shape)
            t_emb = t_emb + cam_emb  # 合并

        return t_emb

    def forward(self, *args, view_dir_emb=None, **kwargs):

        self._cached_camera = view_dir_emb  
        # print(">> sample:", kwargs["sample"].shape)  # [B, 4, 64, 64]
        # print(">> timestep:", kwargs["timestep"].shape)  # [B]
        # print(">> encoder_hidden_states:", kwargs["encoder_hidden_states"].shape)  # [B, 77, 768]
        # print(">> view_dir_emb:", view_dir_emb.shape if view_dir_emb is not None else "None")
        out = super().forward(*args, **kwargs)
        self._cached_camera = None  # 清理，避免跨 batch
        return out

    @classmethod
    def from_config(cls, config):
        # IMPORTANT! otherwise model will not correctly initialized
        return cls(**config)



    
if __name__ == "__main__":

#     unet = UNetFuseTimeCamera(
#     cam_dim=12,
#     in_channels=4,
#     out_channels=4,
#     block_out_channels=(320, 640, 1280, 1280),
#     down_block_types=(
#         "CrossAttnDownBlock2D",
#         "CrossAttnDownBlock2D",
#         "CrossAttnDownBlock2D",
#         "DownBlock2D",
#     ),
#     up_block_types=(
#         "UpBlock2D",
#         "CrossAttnUpBlock2D",
#         "CrossAttnUpBlock2D",
#         "CrossAttnUpBlock2D",
#     ),
#     cross_attention_dim=768,
# )
#     unet.eval()

    config = UNet2DConditionModel.load_config("runwayml/stable-diffusion-v1-5", subfolder="unet")
    print(config)
    #unet = UNet2DConditionModel.from_config(config)
    #config = UNet2DConditionModel.load_config("runwayml/stable-diffusion-v1-5", subfolder="unet")
    unet = UNetFuseTimeCamera.from_config(config)

    # pretrained_unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    # missing, unexpected = unet.load_state_dict(pretrained_unet.state_dict(), strict=False)
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)
    unet.eval()

    for name, _ in unet.named_parameters():
        print(name)

    # # ==== fake inputs ====
    batch_size = 2
    height = width = 64  # SD 1.5 latent resolution for 512x512 input is 64x64

    # latent input (B, 4, 64, 64)
    sample = torch.randn(batch_size, 4, height, width)

    # timestep (int64 tensor)
    timestep = torch.randint(0, 1000, (batch_size,), dtype=torch.long)


    # text embedding from CLIP (B, 77, 768)
    encoder_hidden_states = torch.randn(batch_size, 77, 768)

    # your camera view embedding (B, 12)
    view_dir_emb = torch.randn(batch_size, 12)

    with torch.no_grad():
        output = unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            view_dir_emb=view_dir_emb,
        )
    
    print("Output:", output.sample.shape)

    # unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    # unet.eval()

    # # ==== fake inputs ====
    # batch_size = 2
    # height = width = 64  # Latent resolution for 512x512 image

    # # latent (B, 4, 64, 64)
    # sample = torch.randn(batch_size, 4, height, width)

    # # timestep (B,)
    # timestep = torch.randint(0, 1000, (batch_size,), dtype=torch.long)

    # # encoder hidden states (B, 77, 768)
    # encoder_hidden_states = torch.randn(batch_size, 77, 768)

    # timestep_cond = torch.randn(batch_size, 1280)

    # t_emb = unet.get_time_embed(sample=sample, timestep=timestep)
    # print(t_emb.shape)
    #time_emb = unet.time_embedding(t_emb)  # shape: [B, 1280]

    

    # ==== forward ====
    # with torch.no_grad():
    #     output = unet(
    #         sample=sample,
    #         timestep=timestep,
    #         encoder_hidden_states=encoder_hidden_states,
    #         timestep_cond=timestep_cond,
    #     )

    # print("✅ Forward success. Output shape:", output.sample.shape)
