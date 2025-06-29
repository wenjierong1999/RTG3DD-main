import time
import torch
from PIL import Image, PngImagePlugin
from peft import PeftModel, PeftConfig 
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.schedulers import EulerAncestralDiscreteScheduler
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from depth2img_finetune.Customized_Unet import UNetFuseTimeCamera  

class DirectionAwarenDepth2ImgControlNet:

    def __init__(self, config, torch_dtype=torch.float16, device=torch.device('cuda')):
        
        # === 1. Load ControlNet(s) ===
        controlnet_list = []
        for cnet_unit in config.controlnet_units:
            controlnet = ControlNetModel.from_pretrained(cnet_unit.controlnet_key, torch_dtype=torch_dtype)
            controlnet_list.append(controlnet)

        # === 2. Load original pipeline ===
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            config["sd_model_key"],
            controlnet=controlnet_list if len(controlnet_list) > 1 else controlnet_list[0],
            torch_dtype=torch_dtype
        )
        
        # === 4. Handle customized UNet ===
        self.use_custom_unet = bool(config.get("use_customized_unet", False))
        if self.use_custom_unet:
            print("[INFO] Using customized UNet...")
            # === update config ===
            unet_config = pipe.unet.config
            unet_config["class_embed_type"] = "timestep"  # ← VERY IMPORTANT
                # === initialize ===
            customized_unet = UNetFuseTimeCamera.from_config(unet_config)

            unet_weights_path = config.get("customized_unet_weights_path")

            if unet_weights_path:
                state_dict = torch.load(unet_weights_path, map_location="cpu")
                customized_unet.load_state_dict(state_dict, strict=False)
                print(f"[INFO] Loaded customized UNet weights from: {unet_weights_path}")
            else:
                customized_unet.load_state_dict(pipe.unet.state_dict(), strict=False)
                print("[INFO] No UNet weights provided, using base model's weights.")
            
            pipe.unet = customized_unet  # ← ✅ Critical replacement
        
        # === 5. Load LoRA (PEFT) if available ===
        lora_path = config.get("lora_weights_path")
        if lora_path:
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
            pipe.unet = pipe.unet.to(dtype=torch_dtype, device=device)
            print(f"[INFO] Loaded PEFT LoRA weights from: {lora_path}")
        
        # === 6. Load IP Adapter if applicable ===
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.safetensors")

        # === 7. Scheduler + disable safety checker ===
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(config["sd_model_key"], subfolder="scheduler")
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        pipe.enable_model_cpu_offload()

        # === 8. Save attributes ===
        self.pipe = pipe
        self.cfg = config
        self.guidance_scale = config["guidance_scale"]
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        self.device = device

    @torch.no_grad()
    def inference(self,
                  cond_img: torch.Tensor, # depth map
                  direction_prompt: list[str],
                  camera_label: torch.Tensor = None):       # [B, D], optional, used by customized UNet

        self.use_camera_label = camera_label is not None
        cfg = self.cfg
        B = cond_img.shape[0]

        height = cfg["height"] #512*512
        width = cfg["width"]

        if cond_img.shape[2] != height or cond_img.shape[3] != width:
            cond_img = F.interpolate(cond_img, size=(height, width), mode='bilinear', align_corners=False)
        
        cond_img = cond_img.to(self.device, dtype=self.pipe.controlnet.dtype)

        # 1. Encode direction prompt
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=direction_prompt,
            device=self.device,
            num_images_per_prompt=cfg["num_images_per_prompt"],
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt= [cfg.get("negative_prompt", "")] * B
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        
        print(f'prompt_embeds.dtype: {prompt_embeds.dtype}')
        
        # 2. Prepare IP-Adapter embeddings (optional)

        added_cond_kwargs = {}
        if cfg.get('ip_adapter_image_path') is not None:
            ip_image = Image.open(cfg['ip_adapter_image_path']).convert("RGB")
            image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=ip_image,
                ip_adapter_image_embeds=None,
                device=self.device,
                num_images_per_prompt=B * cfg['num_images_per_prompt'],
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
            added_cond_kwargs["image_embeds"] = image_embeds
        
        # 3. Latents
        self.pipe.scheduler.set_timesteps(cfg["num_inference_steps"], device=self.device)
        timesteps = self.pipe.scheduler.timesteps


        latents = self.pipe.prepare_latents(
            batch_size=B * cfg['num_images_per_prompt'],
            num_channels_latents=self.pipe.unet.config.in_channels,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=self.device,
            generator=torch.manual_seed(cfg.get("seed", 42))
        )

        # === 5. Denoising loop ===
        for t in timesteps:
            latent_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)

            # ControlNet forward

            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                latent_input,
                t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=torch.cat([cond_img]*2) if self.do_classifier_free_guidance else cond_img,
                conditioning_scale=cfg["controlnet_units"][0]["weight"],
                return_dict=False,
            )

            # UNet forward

            unet_kwargs = {
            "sample": latent_input,
            "timestep": t,
            "encoder_hidden_states": prompt_embeds,
            "down_block_additional_residuals": down_block_res_samples,
            "mid_block_additional_residual": mid_block_res_sample,
            "added_cond_kwargs": added_cond_kwargs,
            "return_dict": False,
        }

            if self.use_custom_unet:
                # === camera condition: either class_labels or view_dir_emb ===
                if getattr(self, "use_camera_label", False):
                    class_labels = camera_label.to(self.device)
                    unet_kwargs["class_labels"] = (
                        torch.cat([class_labels] * 2) if self.do_classifier_free_guidance else class_labels
                    )
                else:
                    raise NotImplementedError

            # === UNet call ===
            noise_pred = self.pipe.unet(**unet_kwargs)[0]

            # === Classifier-Free Guidance ===
            if self.do_classifier_free_guidance:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + cfg["guidance_scale"] * (noise_text - noise_uncond)
            
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # === 6. Decode latents into images ===
        images = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample

        # === 7. Postprocess to PIL images ===
        images = self.pipe.image_processor.postprocess(images, output_type="pil")

        return images # # List[PIL.Image]



if __name__ == "__main__":

    from omegaconf import OmegaConf

    cfg_path = "config/finetuned_depth_based_cnet.yaml"
    cfg_all = OmegaConf.load(cfg_path)
    cfg = cfg_all["depth2img"]

    print(cfg)

    # === 3. Create manual inputs ===
    model_id = "0034935b-20a0-45e6-b2a5-9bff89a94198"
    test_data_path = os.path.join("/scratch/leuven/375/vsc37593/finetune_depth2img_3dfuture/")
    raw_data_path = "/scratch/leuven/375/vsc37593/3D-FUTURE-model"
    cfg.ip_adapter_image_path = os.path.join(raw_data_path,model_id,'image.jpg')

    # === 2. Initialize model ===
    model = DirectionAwarenDepth2ImgControlNet(cfg)

    depth_map_front_path = os.path.join(test_data_path, model_id, 'view_000','depth.png')
    depth_map_back_path = os.path.join(test_data_path, model_id, 'view_002','depth.png')

    depth_front_img = Image.open(depth_map_front_path).convert("RGB").resize((cfg.width, cfg.height))
    depth_back_img = Image.open(depth_map_back_path).convert("RGB").resize((cfg.width, cfg.height))

    depth_front = torch.from_numpy(np.array(depth_front_img)).permute(2, 0, 1).float() / 255.0
    depth_back = torch.from_numpy(np.array(depth_back_img)).permute(2, 0, 1).float() / 255.0

    direction_prompt = [
        "a photo of a modern furniture from front view",
        "a photo of a modern furniture from back view"
    ]

    cond_img = torch.stack([depth_front, depth_back], dim=0).to(model.device)

    # print(cond_img.shape)

    camera_label = torch.tensor([0, 1], dtype=torch.long).to(model.device)

    # === 6. Run inference ===

    output_images = model.inference(
        cond_img=cond_img,
        direction_prompt=direction_prompt,
        camera_label=camera_label
    )

    os.makedirs("test_outputs", exist_ok=True)
    for i, img in enumerate(output_images):
        img.save(f"test_outputs/output_{i}.png")







