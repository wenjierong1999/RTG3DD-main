import os
import torch
from tqdm import tqdm
import pynvml
from torchvision.utils import make_grid, save_image
from peft import get_peft_model, LoraConfig, TaskType
from torchvision import transforms
from accelerate import Accelerator
from PIL import Image
from transformers import get_cosine_schedule_with_warmup
from cleanfid import fid
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler, UNet2DConditionModel
import argparse
import shutil

from Customized_Unet import UNetFuseTimeCamera
# from lora_utils import add_lora_to_unet
from Depth2ImgFineTuneDataset import get_dataloader, Depth2ImgFineTuneDataset, collect_split_ids


class Depth2ImgTrainer:

    def __init__(self, 
                sd_cfg,
                train_loader, 
                val_loader = None, 
                customized_unet=None,
                use_camera_label=False,
                use_directional_prompts=False,
                output_dir="results/depth2img", 
                fid_gt_dir=None,
                use_lora=False, 
                lora_rank=8, 
                lr=1e-5,
                num_epochs=10,
                eval_interval=500, # step
                log_interval=100,
                resume=False,
                max_train_steps=None,
                max_grad_norm=1.0
                ):      
        # accelerator setting
        self.accelerator = Accelerator(mixed_precision="no")
        self.device = self.accelerator.device

        # GPU memory monitoring        
        if torch.cuda.is_available():
            pynvml.nvmlInit()
            num_devices = pynvml.nvmlDeviceGetCount()

            # 防止 local_process_index 超出 GPU 数量（尤其是手动设置 CUDA_VISIBLE_DEVICES 时）
            self.gpu_index = min(self.accelerator.local_process_index, num_devices - 1)

            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except pynvml.NVMLError as e:
                print(f"[Warning] Failed to get NVML handle: {e}")
                self.gpu_handle = None
        else:
            print("[Warning] No CUDA device available")
            self.gpu_handle = None     

        # init sd pipeline setting
        self.sd_cfg = sd_cfg
        self.customized_unet = customized_unet
        self.use_camera_label = use_camera_label
        self.use_directional_prompts = use_directional_prompts

        # dataloader setting
        self.train_loader = train_loader
        self.val_loader = val_loader

        # training setting
        self.output_dir = output_dir
        self.fid_gt_dir = fid_gt_dir
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lr = lr
        self.num_epochs = num_epochs
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.resume = resume
        self.max_train_steps = max_train_steps
        self.max_grad_norm = max_grad_norm

        os.makedirs(self.output_dir, exist_ok=True)

        self._init_model()
        self._init_optimizer()

        self.global_step = 0

        if self.resume:
            raise NotImplementedError


    def _init_model(self):

        cfg = self.sd_cfg

        # 1. ControlNet
        controlnet_units = cfg.get("controlnet_units", [])
        controlnet_list = []
        for unit in controlnet_units:
            model = ControlNetModel.from_pretrained(unit["controlnet_key"], torch_dtype=torch.float32)
            controlnet_list.append(model)
        controlnet = controlnet_list[0] if len(controlnet_list) == 1 else controlnet_list

        # 2. Load base pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            cfg["sd_model_key"],
            controlnet=controlnet,
            torch_dtype=torch.float32
        )

        # 3. Replace with customized UNet
        self.use_customized_unet = self.customized_unet is not None
        if self.customized_unet is not None:
            # Load base weights
            self.customized_unet.load_state_dict(pipe.unet.state_dict(), strict=False)
            pipe.unet = self.customized_unet.to(dtype=torch.float32, device=self.device)
        else:
            pipe.unet.to(dtype=torch.float32, device=self.device)

        # 4. Load IP-Adapter if specified
        # if cfg.get("ip_adapter_image_path"):
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.safetensors")
        
        # 5. Scheduler and safety
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(cfg["sd_model_key"], subfolder="scheduler")
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

         # 7. Freeze unnecessary modules
        pipe.vae.requires_grad_(False)

        if isinstance(pipe.controlnet, list):
            for cnet in pipe.controlnet:
                cnet.requires_grad_(False)
        else:
            pipe.controlnet.requires_grad_(False)
        
        if hasattr(pipe, "image_proj_model") and pipe.image_proj_model is not None:
            pipe.image_proj_model.requires_grad_(False)
            print('image_proj_model is frozen')

            # 8. Set UNet trainable
        if not self.use_lora:
            pipe.unet.requires_grad_(True)
        else: # lora config
            lora_config = LoraConfig(
                r = self.lora_rank,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                lora_dropout=0.01,
            )
            pipe.unet = get_peft_model(pipe.unet, lora_config)
            pipe.unet.print_trainable_parameters()
            # raise NotImplementedError("LoRA training not implemented.")
        # === Manually move submodules to correct device ===

        pipe.text_encoder.to(self.device, dtype=torch.float32)
        pipe.text_encoder.requires_grad_(False)
        print('text_encoder is frozen')

        pipe.vae = pipe.vae.to(dtype=torch.float32, device=self.device)

        if hasattr(pipe, "image_proj_model") and pipe.image_proj_model is not None:
            pipe.image_proj_model.image_encoder.to(self.device, dtype=torch.float32)
            print(f'move image_proj_model.image_encoder to {self.device}')
            if hasattr(pipe.image_proj_model, "image_encoder"):
                pipe.image_proj_model.image_encoder.to(self.device, dtype=torch.float32)
                pipe.image_proj_model.image_encoder.requires_grad_(False)

        if pipe.image_encoder is not None:
            pipe.image_encoder.requires_grad_(False)
            pipe.image_encoder.to(self.device, dtype=torch.float32)
            print(f'move image_encoder to {self.device}')
        
        if isinstance(pipe.controlnet, list):
            for cnet in pipe.controlnet:
                cnet.to(self.device, dtype=torch.float32)  
        else:
            pipe.controlnet.to(self.device, dtype=torch.float32)
        
        self.pipe = pipe
        
        self.pipe.unet.enable_gradient_checkpointing()

        self.eval_scheduler = self._init_eval_scheduler() #TEST

        # post checking
        print(f"[RANK {self.accelerator.process_index}] UNet device: {next(pipe.unet.parameters()).device}")

    def _init_optimizer(self):

        params = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            eps=1e-08
        )
        self.total_steps = self.num_epochs * len(self.train_loader)
        warmup_steps = int(0.05 * self.total_steps)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps
        )

        self.pipe, self.optimizer, self.lr_scheduler, self.train_loader = self.accelerator.prepare(
            self.pipe, self.optimizer, self.lr_scheduler, self.train_loader
        )
    
    def _init_eval_scheduler(self):
        from copy import deepcopy
        return deepcopy(self.pipe.scheduler)
    
    def train(self):

        # self._set_train_scheduler()

        self.accelerator.print(f"Process {self.accelerator.process_index} using device {self.device}, total {self.accelerator.num_processes} processes")

        self.pipe.unet.train()
        self.pipe.vae.eval()
        self.pipe.controlnet.eval()

        if hasattr(self.pipe, "image_proj_model"):
            self.pipe.image_proj_model.eval()
        
        self.guidance_scale = self.sd_cfg.get("guidance_scale", 7.5)
        self.do_classifier_free_guidance = self.guidance_scale > 1.0

        # training loop
        for epoch in range(self.num_epochs):
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):

                if self.max_train_steps is not None and self.global_step >= self.max_train_steps:
                    self.accelerator.print(f"[INFO] Reached max_train_steps = {self.max_train_steps}, stopping early.")
                    return

                self.pipe.unet.train()
                self.optimizer.zero_grad()

                # === 1. Flatten multi-view batch ===
                rgb = batch["rgb"]            # (B, V, 3, H, W) GT rendered view
                depth = batch["depth"]        # (B, V, 3, H, W) depth map for controlnet
                view_dir = batch["view_dir"]  # (B, V, D) camera parameter (optional)
                prompts = batch["prompt"]     # (B, V) list of str  (optional)
                cond_img_paths = batch["cond_img_path"]  # (B,) # conditioning image for ip adapter
                class_labels = batch["camera_pos_label"].view(-1).to(self.device)

                B, V, C, H, W = rgb.shape

                # breakpoint()
                rgb = rgb.view(B * V, C, H, W).to(self.device)
                depth = depth.view(B * V, C, H, W).to(self.device)
                view_dir = view_dir.view(B * V, -1).to(self.device)

                if self.use_directional_prompts:
                    flat_prompts = [p for plist in prompts for p in plist]
                else:
                    flat_prompts = [" "] * (B * V)

                # === 2. Prompt encoding (with negative prompt support) ===
                # flat_prompts = [p for plist in prompts for p in plist]
                negative_prompt = self.sd_cfg.get("negative_prompt", None)
                negative_prompt = [negative_prompt] * len(flat_prompts)

                with torch.no_grad():
                    #DEBUG
                    # print(f"UNet dtype: {next(self.pipe.unet.parameters()).dtype}")
                    # print(f"Text encoder dtype: {next(self.pipe.text_encoder.parameters()).dtype}")
                    self.pipe.text_encoder.to(dtype=torch.float32)
                    prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                        prompt=flat_prompts,
                        negative_prompt=negative_prompt,
                        device= next(self.pipe.text_encoder.parameters()).device,
                        num_images_per_prompt=1, # *batch size
                        do_classifier_free_guidance=self.do_classifier_free_guidance
                    )
                    if self.do_classifier_free_guidance:
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
  
                # === 3. IP-Adapter images ===

                #TODO: current ip-adapter implementation does not support batch operation!!!
                #Check https://github.com/huggingface/diffusers/discussions/7933

                ip_image = Image.open(cond_img_paths[0]).convert("RGB")
                # print(f'using ip adapter image {cond_img_paths[0]}')

                with torch.no_grad():
                    image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                        ip_adapter_image=ip_image,
                        ip_adapter_image_embeds=None,
                        # device= next(self.pipe.text_encoder.parameters()).device,
                        device = next(self.pipe.unet.parameters()).device,
                        num_images_per_prompt=len(flat_prompts),
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                    )  # [torch.Size([8, 1, 257, 1280])]
                
                    # breakpoint()
                    added_cond_kwargs = {"image_embeds": image_embeds}

                #breakpoint()
                # === 4. VAE encode target image ===
                with torch.no_grad():
                    rgb = rgb.to(dtype=torch.float32, device=self.device)

                    latents = self.pipe.vae.encode(rgb).latent_dist.sample()
                    latents = latents * self.pipe.vae.config.scaling_factor

                    # debug_check("latents (after VAE encode)", latents)

                # === 5. Add noise ===
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, self.pipe.scheduler.config.num_train_timesteps,(latents.size(0),), device=self.device).long()
                noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)

                if self.do_classifier_free_guidance:
                    timesteps = torch.cat([timesteps] * 2)

                # === 6. Expand latents for classifier-free guidance ===
                latent_input = torch.cat([noisy_latents] * 2) if self.do_classifier_free_guidance else noisy_latents

                # === 7. ControlNet forward ===
                depth = depth.to(dtype=torch.float32, device=self.device)
                down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                    latent_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=torch.cat([depth] * 2) if self.do_classifier_free_guidance else depth,
                    conditioning_scale=1.0,
                    return_dict=False
                )

                # breakpoint()

                # === 8. UNet forward ===

                if self.use_customized_unet:
                    if self.use_camera_label:
                        unet_class_labels = torch.cat([class_labels] * 2) if self.do_classifier_free_guidance else class_labels
                    else:
                        unet_class_labels = None

                    noise_pred = self.pipe.unet(
                        sample=latent_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        class_labels = unet_class_labels,
                        # view_dir_emb=torch.cat([view_dir] * 2) if self.do_classifier_free_guidance else view_dir,
                        view_dir_emb = None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False
                    )[0]
                else:
                    if self.use_camera_label:
                        unet_class_labels = torch.cat([class_labels] * 2) if self.do_classifier_free_guidance else class_labels
                    else:
                        unet_class_labels = None

                    noise_pred = self.pipe.unet(
                        sample=latent_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        # class_labels = unet_class_labels,
                        # view_dir_emb=torch.cat([view_dir] * 2) if self.do_classifier_free_guidance else view_dir,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False
                    )[0]
                
                # breakpoint()
                #DEBUG
                if torch.isnan(noise_pred).any():
                    print("[FATAL] UNet output contains NaN")
                
                # === 9. Classifier-free guidance loss ===
                if self.do_classifier_free_guidance:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)
                
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[FATAL] Loss is NaN or Inf: {loss}")
                # else:
                #     print(f"[DEBUG] Loss = {loss.item():.6f}")
                
                self.accelerator.backward(loss)

                # === Gradient Clipping
                if self.accelerator.sync_gradients:
                    params_to_clip = self.pipe.unet.parameters()
                    self.accelerator.clip_grad_norm_(params_to_clip, max_norm=self.max_grad_norm)

                self.optimizer.step()
                self.lr_scheduler.step()

                # breakpoint()
                # === 10. Logging ===
                if self.global_step % self.log_interval == 0:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    used_mem = mem_info.used / 1024**2
                    total_mem = mem_info.total / 1024**2

                    self.accelerator.print(
                        f"[Epoch {epoch} | Step {self.global_step}] "
                        f"Loss: {loss.item():.4f} | "
                        f"GPU Real Mem: {used_mem:.1f}/{total_mem:.1f} MB"
                    )

                # breakpoint()
                
                # === 11. Save UNet checkpoint ===
                if self.global_step % self.eval_interval == 0:
                    if self.accelerator.is_main_process:
                        ckpt_path = os.path.join(self.output_dir, f"checkpoint_{self.global_step}.pt")
                        if self.use_lora:
                            self.pipe.unet.save_pretrained(ckpt_path)
                            self.accelerator.print(f"Saved LoRA adapter at {ckpt_path}")
                        else:
                            torch.save(self.accelerator.unwrap_model(self.pipe.unet).state_dict(), ckpt_path)
                            self.accelerator.print(f"Saved UNet checkpoint: {ckpt_path}")
                        #TODO: periodical evaluation
                        self.evaluate(grid_samples=20)

                        # === Save full resume state (overwrite)
                        resume_path = os.path.join(self.output_dir, "state_latest")
                        self.accelerator.save_state(resume_path)
                        self.accelerator.print(f"Saved training state to: {resume_path}")
                    self.accelerator.wait_for_everyone()
                
                self.global_step += 1
                # self._set_train_scheduler()  # restore for next training step

    @torch.no_grad()
    def evaluate(self, empty_prompt=False,
                 cache_individual_views=True,
                 grid_samples=2):

        self.guidance_scale = self.sd_cfg.get("guidance_scale", 7.5)
        self.do_classifier_free_guidance = self.guidance_scale > 1.0

        # set pipe to evaluation mode
        self.pipe.unet.eval()
        self.pipe.vae.eval()
        self.pipe.controlnet.eval()
        if hasattr(self.pipe, "image_proj_model"):
            self.pipe.image_proj_model.eval()

        # breakpoint()
        
        # === fallback to train_loader if val_loader is None ===
        eval_loader = self.val_loader if self.val_loader is not None else self.train_loader
        self.accelerator.print(f"Using {'val_loader' if self.val_loader is not None else 'train_loader'} for evaluation.")
        
        # === create output folders
        eval_cache_dir = os.path.join(self.output_dir, "eval_cache", f"eval_cache_step{self.global_step}")
        eval_grid_dir = os.path.join(self.output_dir, "eval_grid")

        if self.accelerator.is_main_process:
            if cache_individual_views:
                if os.path.exists(eval_cache_dir):
                    shutil.rmtree(eval_cache_dir)
                os.makedirs(eval_cache_dir, exist_ok=True)
            os.makedirs(eval_grid_dir, exist_ok=True)

        for idx, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"[Eval @ step {self.global_step}]"):

            rgb = batch["rgb"]            # (B, V, 3, H, W)
            depth = batch["depth"]        # (B, V, 3, H, W)
            view_dir = batch["view_dir"]  # (B, V, D)
            prompts = batch["prompt"]     # (B, V)
            cond_img_paths = batch["cond_img_path"]  # (B,)
            model_id = batch["model_id"]  # str
            class_labels = batch["camera_pos_label"].view(-1).to(self.device)

            B, V, C, H, W = rgb.shape
            rgb = rgb.view(B * V, C, H, W).to(self.device)
            depth = depth.view(B * V, C, H, W).to(self.device)
            view_dir = view_dir.view(B * V, -1).to(self.device)

            
            flat_prompts = [p for plist in prompts for p in plist]
            if not self.use_directional_prompts:
                flat_prompts = [" "] * len(flat_prompts)
            negative_prompt = self.sd_cfg.get("negative_prompt", None)
            negative_prompt = [negative_prompt] * len(flat_prompts)

            # === prompt encoding
            self.pipe.text_encoder.to(dtype=torch.float32)
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt=flat_prompts,
                negative_prompt=negative_prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=self.do_classifier_free_guidance
            )
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            
            # === IP-Adapter image
            ip_image = Image.open(cond_img_paths[0]).convert("RGB")
            image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=ip_image,
                ip_adapter_image_embeds=None,
                device=self.device,
                num_images_per_prompt=len(flat_prompts),
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
            added_cond_kwargs = {"image_embeds": image_embeds}

            # === Initial noise ===
            latents = torch.randn((B * V, self.pipe.unet.in_channels, H // 8, W // 8),
                                device=self.device, dtype=torch.float32)
            scheduler = self.eval_scheduler
            scheduler.set_timesteps(self.sd_cfg['num_inference_steps'], device=self.device)
            timesteps = scheduler.timesteps

            for t in timesteps:

                latent_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
                latent_input = scheduler.scale_model_input(latent_input, t)
                
                #IMPORTANT:
                depth = depth.to(dtype=torch.float32, device=self.device)
                prompt_embeds = prompt_embeds.to(dtype=torch.float32)

                down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                    latent_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=torch.cat([depth] * 2) if self.do_classifier_free_guidance else depth,
                    conditioning_scale=1.0,
                    return_dict=False
                )
                
                # breakpoint()
                if self.use_customized_unet:

                    if self.use_camera_label:
                        unet_class_labels = torch.cat([class_labels] * 2) if self.do_classifier_free_guidance else class_labels
                        view_dir_emb = None
                    else:
                        unet_class_labels = None
                        # view_dir_emb = torch.cat([view_dir] * 2) if self.do_classifier_free_guidance else view_dir
                        view_dir_emb = None
                        
                    noise_pred = self.pipe.unet(
                        sample=latent_input,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        class_labels = unet_class_labels,
                        view_dir_emb=view_dir_emb,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False
                    )[0]
                else:
                    noise_pred = self.pipe.unet(
                        sample=latent_input,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False
                    )[0]
                
                # breakpoint()
                
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            images = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
            # print(images.shape)
            images = (images.clamp(-1, 1) + 1) / 2.0  # Normalize to [0, 1]
            images = images.cpu()

            if idx < grid_samples:
                gt_images = batch["rgb"].view(B * V, 3, H, W).cpu()
                gen_grid = make_grid(images, nrow=V)
                gt_grid = make_grid(gt_images, nrow=V)
                full_grid = torch.cat([gen_grid, gt_grid], dim=1)
                grid_save_path = os.path.join(eval_grid_dir, f"eval_grid_{model_id[0]}_{self.global_step}.png")
                save_image(full_grid, grid_save_path)
                self.accelerator.print(f"[Eval] Saved view grid image to {grid_save_path}")
            
            # === Save individual view images for fid computation
            if cache_individual_views:
                for i, img in enumerate(images):
                    model_name = model_id[i // V]
                    view_idx = i % V
                    save_path = os.path.join(eval_cache_dir, f"{model_name}_view_{view_idx}.png")
                    save_image(img, save_path)
            
        # === After all batches, compute FID if GT folder provided
        if self.fid_gt_dir is not None and self.accelerator.is_main_process:

            fid_score = fid.compute_fid(
                self.fid_gt_dir,
                eval_cache_dir,
                mode="clean",
                num_workers=4
            )
            self.accelerator.print(f"[FID] Step {self.global_step}: {fid_score:.4f}")

                        # Save to file
            fid_log_file = os.path.join(self.output_dir, "fid_scores.csv")
            header = "step,fid\n"
            line = f"{self.global_step},{fid_score:.4f}\n"
            if not os.path.exists(fid_log_file):
                with open(fid_log_file, "w") as f:
                    f.write(header)
                    f.write(line)
            else:
                with open(fid_log_file, "a") as f:
                    f.write(line)



def parse_args():

    parser = argparse.ArgumentParser(description="Depth2Img Training / Evaluation")

    # I/O
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--fid_gt_dir", type=str, default=None, help="GT folder for FID")

    # Mode
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume training")

    # Model training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=25000)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=20)

    # Architecture options
    parser.add_argument("--use_customized_unet", action="store_true", help="Use customized UNetFuseTimeCamera")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--use_camera_label", action="store_true")
    parser.add_argument("--use_directional_prompts", action="store_true")

    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()

    # basic config for loading sd pipelines
    config = {
    "sd_model_key": "runwayml/stable-diffusion-v1-5",
    "ip_adapter_image_path": "/scratch/leuven/375/vsc37593/3D-FUTURE-model/0b9f1125-92d6-4be2-9503-22200d4c7e16/image.jpg",  # 可设为 None
    "prompt": "A modern furniture from front view",
    # "negative_prompt": "strong light, Bright light, intense light, dazzling light, brilliant light, radiant light, Shade, darkness, silhouette, dimness, obscurity, shadow, glasses",
    "negative_prompt": "strong light, harsh lighting, blurry, noisy, low quality, cluttered, messy, shadow, dirty",
    "seed": 912240,
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
    
    # val_loader, train_loader = get_dataloader(root='/scratch/leuven/375/vsc37593/finetune_depth2img_3dfuture', batch_size=4)

    dataset_root = '/scratch/leuven/375/vsc37593/finetune_depth2img_3dfuture'
    train_ids, test_ids = collect_split_ids(max_test_per_category=40)

    train_loader = get_dataloader(
        root=dataset_root,
        batch_size=4,
        views_per_model=4,
        model_ids=train_ids, #MODIFIED
        shuffle=False
    )

    # test dataloader
    test_loader = get_dataloader(
        root=dataset_root,
        batch_size=4,
        views_per_model=4,
        model_ids=test_ids,
        shuffle=False
    )
    if args.use_customized_unet:
        unet_config = UNet2DConditionModel.load_config(config["sd_model_key"], subfolder="unet")
        if args.use_camera_label:
            unet_config["class_embed_type"] = "timestep"
        customized_unet = UNetFuseTimeCamera.from_config(unet_config)
    else:
        customized_unet = None

    trainer = Depth2ImgTrainer(
        sd_cfg=config,
        train_loader=train_loader,
        val_loader=test_loader,
        customized_unet=customized_unet,
        use_camera_label=args.use_camera_label,
        use_directional_prompts=args.use_directional_prompts,
        use_lora=args.use_lora, # if finetune the whole unet
        output_dir=args.output_dir,
        fid_gt_dir=args.fid_gt_dir,
        lr=args.lr,
        num_epochs=args.num_epochs,
        eval_interval=args.eval_interval,  # Skip eval to only test one epoch
        log_interval=args.log_interval,
        resume=False,
        max_train_steps=args.max_steps
    )


    trainer.train()

    # trainer.evaluate(grid_samples=20)


            
        




    
    