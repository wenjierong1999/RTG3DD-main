import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
from diffusers import UNet2DConditionModel
from types import SimpleNamespace
from PIL import Image, PngImagePlugin
from tqdm import tqdm
from Customized_Unet import UNetFuseTimeCamera

# config template
'''
txt2img:
  "sd_model_key": "runwayml/stable-diffusion-v1-5"
  "ip_adapter_image_path": 
  "prompt": "turn around, monkey head, (Sci-Fi digital painting:1.5), colorful, painting, high quality"
  "negative_prompt": "strong light, Bright light, intense light, dazzling light, brilliant light, radiant light, Shade, darkness, silhouette, dimness, obscurity, shadow, glasses"
  "seed": 1713428430
  "width": 1024
  "height": 512
  "num_images_per_prompt": 1
  "guidance_scale": 7.0
  "num_inference_steps": 20
  "controlnet_units":
    - "preprocessor": "none"
      "controlnet_key": "lllyasviel/control_v11f1p_sd15_depth"
      "condition_image_path":
      "weight": 1.0
'''

class Depth2imgManualPipe:

    def __init__(
        self,
        cfg,
        ip_adapter_name = 'ip-adapter-plus_sd15.safetensors',
        customized_unet = None,
        device = torch.device('cuda')
    ):
       # load controlnet(s)
        controlnet_list = []
        for unit in cfg["controlnet_units"]:
            model = ControlNetModel.from_pretrained(unit["controlnet_key"], torch_dtype=torch.float16)
            controlnet_list.append(model)
        
        # load original pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            cfg["sd_model_key"],
            controlnet=controlnet_list if len(controlnet_list) > 1 else controlnet_list[0],
            torch_dtype=torch.float16
        )

        # load customized unet
        self.use_customized_unet = customized_unet is not None
        if customized_unet is not None:
            customized_unet.load_state_dict(pipe.unet.state_dict(), strict=False)
            # pipe.unet = customized_unet.to(device)
            pipe.unet = customized_unet.to(dtype=torch.float16, device=device)
        

        # load ip adapter
        if cfg.get("ip_adapter_image_path"):
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=ip_adapter_name)

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(cfg["sd_model_key"], subfolder="scheduler")
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        pipe.enable_model_cpu_offload()

        self.pipe = pipe
        self.cfg = cfg
        self.guidance_scale = cfg["guidance_scale"]
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        self.device = device

    @torch.no_grad() 
    def inference(self):

        cfg = self.cfg

        #  Define call parameters
        if cfg['prompt'] is not None and isinstance(cfg['prompt'], str):
            batch_size = 1
        elif cfg['prompt'] is not None and isinstance(cfg['prompt'], list):
            batch_size = len(cfg['prompt'])
        else:
            raise NotImplementedError

        # 1. encode prompts

        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt = cfg['prompt'],
            device = self.device,
            num_images_per_prompt = cfg['num_images_per_prompt'],
            do_classifier_free_guidance = self.do_classifier_free_guidance,
            negative_prompt = cfg['negative_prompt']
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        print(prompt_embeds.shape)

        # 2: IP-Adapter image embedding 

        if cfg['ip_adapter_image_path'] is not None:

            image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image = Image.open(cfg['ip_adapter_image_path']),
                ip_adapter_image_embeds = None,
                device = self.device,
                num_images_per_prompt = batch_size * cfg['num_images_per_prompt'],
                do_classifier_free_guidance = self.do_classifier_free_guidance,
            )

            print(image_embeds[0].shape)

            # Add image embeds for IP-Adapter
            added_cond_kwargs = {"image_embeds": image_embeds}

        # 3. Prepare image for controlnet

        image = self.pipe.prepare_image(
                image=Image.open(cfg["controlnet_units"][0]["condition_image_path"]),
                width=cfg['width'],
                height=cfg['height'],
                batch_size=batch_size * cfg['num_images_per_prompt'],
                num_images_per_prompt=cfg['num_images_per_prompt'],
                device=self.device,
                dtype=self.pipe.controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        height, width = image.shape[-2:]
        print(image.shape)

        # 4. prepare latents

        num_channels_latents = self.pipe.unet.config.in_channels

        latents = self.pipe.prepare_latents(
            batch_size = batch_size * cfg['num_images_per_prompt'],
            num_channels_latents = num_channels_latents,
            height = height,
            width = width,
            dtype = prompt_embeds.dtype,
            device = self.device,
            generator = None
        )

        # print(latents.shape)

        # SKIP 6.5
        # SKIP 7 7.1 7.2 

        # 5. Scheduler
        self.pipe.scheduler.set_timesteps(cfg['num_inference_steps'], device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        # denoising loop 
        for t in timesteps:

            latent_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)

            # run controlnet
            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                latent_input,
                t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=image,
                conditioning_scale=cfg["controlnet_units"][0]["weight"],
                return_dict=False,
            )
            
            # run unet
            #TODO: replace with customized unet
            if self.use_customized_unet:
            
                noise_pred = self.pipe.unet(
                    sample=latent_input,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    view_dir_emb=torch.randn(batch_size, 12), # additional camera position embedding
                    down_block_additional_residuals=down_block_res_samples, # controlnet input
                    mid_block_additional_residual=mid_block_res_sample, # controlnet input
                    added_cond_kwargs=added_cond_kwargs, # ip-adapter input
                    return_dict=False,
                )[0]
            
            else:
                noise_pred = self.pipe.unet(
                    sample=latent_input,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples, # controlnet input
                    mid_block_additional_residual=mid_block_res_sample, # controlnet input
                    added_cond_kwargs=added_cond_kwargs, # ip-adapter input
                    return_dict=False,
                )[0]

            if self.do_classifier_free_guidance:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)
            
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 7: Decode
        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
        image = self.pipe.image_processor.postprocess(image, output_type="pil")

        image[0].save('/data/leuven/375/vsc37593/my_py_projects/test_img.png')

        return image


        # raise NotImplementedError


if __name__ == "__main__":


    config = {
    "sd_model_key": "runwayml/stable-diffusion-v1-5",
    "ip_adapter_image_path": "/scratch/leuven/375/vsc37593/3D-FUTURE-model/0b9f1125-92d6-4be2-9503-22200d4c7e16/image.jpg",  # 可设为 None
    "prompt": "A modern furniture from front view",
    "negative_prompt": "strong light, Bright light, intense light, dazzling light, brilliant light, radiant light, Shade, darkness, silhouette, dimness, obscurity, shadow, glasses",
    "seed": 1713428430,
    "width": 512,
    "height": 512,
    "num_images_per_prompt": 1,
    "guidance_scale": 7.0,
    "num_inference_steps": 20,
    "controlnet_units": [
        {
            "preprocessor": "none",
            "controlnet_key": "lllyasviel/control_v11f1p_sd15_depth",
            "condition_image_path": "/scratch/leuven/375/vsc37593/finetune_depth2img_3dfuture/0b9f1125-92d6-4be2-9503-22200d4c7e16/view_000/depth.png", # depth map
            "weight": 1.0,
        }
    ]
}
    # config = SimpleNamespace(**config_dict)
    # print(config.controlnet_units)

    unet_config = UNet2DConditionModel.load_config("runwayml/stable-diffusion-v1-5", subfolder="unet")
    custom_unet = UNetFuseTimeCamera.from_config(unet_config)

    pipe = Depth2imgManualPipe(config, customized_unet = custom_unet)

    _ = pipe.inference()
    

