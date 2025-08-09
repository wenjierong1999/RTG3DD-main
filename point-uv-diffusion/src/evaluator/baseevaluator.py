import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import PIL
import cv2
import open3d as o3d
from pathlib import Path
import shutil
import time

class MyBaseEvaluator:
    '''
    main evaluator class
    since we use one GPU for evaluation,
    we dont need to set up the torch distributed mode
    '''

    def __init__(
        self,
        device = 'cuda',
        coarse_ckpt = None,
        fine_ckpt = None,
        coarse_ckpt_resume = None,
        fine_ckpt_resume = None,
        raw_data_folder = None,
        preprocessed_data_folder = None
    ):
        super().__init__()
        self.device = device
        self.coarse_ckpt = coarse_ckpt
        self.fine_ckpt = fine_ckpt
        self.coarse_ckpt_resume = coarse_ckpt_resume
        self.fine_ckpt_resume = fine_ckpt_resume
        self.raw_data_folder = raw_data_folder
        self.preprocessed_data_folder = preprocessed_data_folder
    
    def test_coarse(
        self,
        datamodule,
        modelmodule
    ):
        print("Running coarse-stage test...", flush=True)
        print(f"Saving coarse output to {str(Path(self.coarse_ckpt))}", flush=True)
        coarse_time, coarse_samples = self._run_test(
            datamodule=datamodule,
            modelmodule=modelmodule,
            ckpt_path=Path(self.coarse_ckpt_resume),
            save_dir_path=Path(self.coarse_ckpt),
            stage='coarse',
            extra_coarse_save_path=Path(self.coarse_ckpt) / 'coarse_output' / 'save_fpsuv_4096'
        )

        return coarse_time, coarse_samples

    def test_fine(
        self,
        datamodule,
        modelmodule
    ):
        print("Running fine-stage test...", flush=True)
        fine_time, fine_samples = self._run_test(
            datamodule=datamodule,
            modelmodule=modelmodule,
            ckpt_path=Path(self.fine_ckpt_resume),
            save_dir_path=Path(self.fine_ckpt),
            stage='fine'
        )

        return fine_time, fine_samples
        

    def joint_test(
        self,
        coarse_dm,
        coarse_model,
        fine_dm,
        fine_model
    ):
    
        # run coarse stage
        print("[joint] Start runing coarse stage", flush=True)
        coarse_time, coarse_samples = self._run_test(
            datamodule=coarse_dm,
            modelmodule=coarse_model,
            ckpt_path=Path(self.coarse_ckpt_resume),
            save_dir_path=Path(self.coarse_ckpt),
            stage='coarse',
            extra_coarse_save_path=Path(self.coarse_ckpt) / 'coarse_output' / 'save_fpsuv_4096'
        )

        # run fine stage
        print("[joint] Start runing fine stage", flush=True)
        fine_time, fine_samples = self._run_test(
            datamodule=fine_dm,
            modelmodule=fine_model,
            ckpt_path=Path(self.fine_ckpt_resume),
            save_dir_path=Path(self.fine_ckpt),
            stage='fine'
        )

        total_time = coarse_time + fine_time
        total_samples = max(coarse_samples, fine_samples)  #  coarse==fine
        joint_avg = total_time / total_samples if total_samples > 0 else 0.0
        print(f"[joint] Average end-to-end inference time per sample: {joint_avg:.4f} seconds", flush=True)

        
    def _run_test(
        self,
        datamodule,
        modelmodule,
        ckpt_path: Path,
        save_dir_path: Path,
        stage,
        extra_coarse_save_path: Path = None
    ):
        '''
        Shared logic for evaluating a model (coarse or fine).

        Args:
            datamodule: The test datamodule
            modelmodule: The model to be tested
            ckpt_path : Path to model checkpoint
        '''
        modelmodule.net.to(self.device)
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # load trained model (EMA or default)
        if 'model_ema' in checkpoint:
            print('Loading EMA weights', flush=True)
            modelmodule.net.load_state_dict(checkpoint['model_ema'])
        else:
            modelmodule.net.load_state_dict(checkpoint['model'])
        
        modelmodule.net.eval()
        dataloader = datamodule.test_dataloader()

        # determine output dir based on stage
        if stage == 'coarse':
            output_root = Path(save_dir_path) / 'coarse_output'
        elif stage in ['fine', 'joint']:
            output_root = Path(save_dir_path) / 'fine_output'
        else:
            raise ValueError(f'Unknown stage: {stage}')
        
        output_root.mkdir(parents=True, exist_ok=True)
        obj_dir = Path(datamodule.data_detail.uv_folder)

        total_time = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                start_time = time.time()
                # main inference step
                _, batch_media = modelmodule.test_step(batch, batch_idx)
                images = batch_media['image']
                texture_maps = batch_media['texture_map']
                masks = batch_media['mask']

                _, _, res, _ = texture_maps.size()

                for i in range(images.size(0)):
                    obj_c = batch_media['obj_c'][i] # model category
                    obj_name = batch_media['obj_name'][i] # model_id
                    obj_file = obj_dir / obj_c / obj_name / f'uv_texture_{res}.obj'
                    output_save_path = output_root / obj_name
                    output_save_path.mkdir(parents=True, exist_ok=True)

                    texture_map = texture_maps[i:i + 1]
                    mask = masks[i:i + 1]

                    ori_path = self.save_texture_model(
                        obj_file=obj_file,
                        obj_name=obj_name,
                        save_dir=output_save_path,
                        texture_map=texture_map,
                        mask=mask
                    )

                    # save test step visualization
                    vis_path = output_save_path / f'{obj_name}_cat.jpg'
                    torchvision.utils.save_image(images[i:i + 1].detach().cpu(), str(vis_path))

                    # Copy *_ori.png to coarse->fine folder if in coarse stage
                    # 
                    if stage == 'coarse' and extra_coarse_save_path:
                        extra_coarse_save_path.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(
                            str(ori_path),
                            str(extra_coarse_save_path / f'{obj_name}_ori.png')
                        )
                    
                    #TODO: save GT texture and GT rendered image for reference
                    gt_texture_map_path = obj_dir / obj_c / obj_name / f'uv_texture_{res}.png'
                    gt_rendered_img_path = Path(self.raw_data_folder) / obj_name / 'image.jpg'
                    shutil.copyfile(
                        str(gt_texture_map_path),
                        str(output_save_path / gt_texture_map_path.name)
                    )
                    shutil.copyfile(
                        str(gt_rendered_img_path),
                        str(output_save_path / gt_rendered_img_path.name)
                    )
                end_time = time.time()
                batch_time = end_time - start_time
                batch_avg_time = batch_time / images.size(0)
                print(f"[{stage} stage] Batch {batch_idx}: avg inference time per sample = {batch_avg_time:.4f} seconds", flush=True)

                total_time += batch_time
                total_samples += images.size(0)
            
        return total_time, total_samples


    
    def save_texture_model(
        self,
        obj_file: Path, # path to obj file,
        obj_name: str, # model id
        save_dir: Path,
        texture_map, # output of coarse/fine model
        mask, # mask map
    ):
        '''
        NOTE: save_dir = results/experiment-coarse/pos_tag/ # coarse_ckpt or fine_ckpt specified in the cfg file
        example store structure of coarse stage outpyt

        save_dir_coarse/coarse_output/
        └── model_id1                     
            ├── model_id_ori.png         ← original output of coarse stage
            ├── model_id.png             ← output of coarse stage after hole filling operation 
            ├── model_id.obj     ← obj file
            ├── model_id.mtl     ← mtl file
            ├── model_id_cat.jpg ← additional visualization of coarse stage test_step (concatenated image)
        └── model_id2
            ├── ...
        └── ...
        └── finetune_folder              # intermediate store of coarse model output, need to set datamodule.data_detail.finetune_folder to this folder
            ├── model_id1_ori.png
            ├── model_id2_ori.png
            ├── ...
        
        example store sturcture of fine stage output

        save_dir_fine/fine_output/
        └── model_id1                     
            ├── model_id_ori.png         ← original output of fine stage
            ├── model_id.png             ← output of fine stage after hole filling operation 
            ├── model_id.obj     ← obj file
            ├── model_id.mtl     ← mtl file
            ├── model_id_cat.jpg ← additional visualization of fine stage test_step (concatenated image)
        └── model_id2
            ├── ...
        └── ...   

        example store sturcture of fine stage output (joint mode)
        # we store the results of joint mode under folder of fine stage experiment

        save_dir_fine/joint_output/

        same as above...     
        '''
        # Save modified .obj file
        with open(obj_file, 'r') as f:
            lines = f.readlines()
        lines[0] = f'mtllib {obj_name}.mtl\n' # mtllib coarse_xxxx.mtl
        with open(os.path.join(save_dir, f'{obj_name}.obj'), 'w') as f:
            f.writelines(lines)

         # Save .mtl file
        with open(os.path.join(save_dir, f'{obj_name}.mtl'), 'w') as fid:
            fid.write(
                f'newmtl material_0\n'
                f'Kd 1 1 1\nKa 0 0 0\nKs 0.4 0.4 0.4\n'
                f'Ns 10\nillum 2\nmap_Kd {obj_name}.png'
            )

        # save original output of coarse model
        color_map = texture_map.permute(0, 2, 3, 1)[0]
        img = np.asarray(color_map.data.cpu().numpy(), dtype=np.float32)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        ori_path = save_dir / f'{obj_name}_ori.png'
        PIL.Image.fromarray(np.ascontiguousarray(img), 'RGB').save(ori_path)

        # save filled texture
        mask = mask.permute(0, 2, 3, 1)[0]
        mask = np.asarray(mask.data.cpu().numpy(), dtype=np.float32)
        kernel = np.ones((3, 3), 'uint8')
        dilate_img = cv2.dilate(img, kernel, iterations=1)
        hom_img = img * mask + dilate_img * (1 - mask)
        hom_img = hom_img.clip(0, 255).astype(np.uint8)
        PIL.Image.fromarray(np.ascontiguousarray(hom_img), 'RGB').save(save_dir / f'{obj_name}.png')

        return ori_path

