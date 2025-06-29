import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

class Depth2ImgFineTuneDataset(Dataset):

    def __init__(self, root_dir, views_per_model=4, image_size=(512, 512), transform=None,
                 raw_data_dir = '/scratch/leuven/375/vsc37593/3D-FUTURE-model'):
        super().__init__()
        self.root_dir = root_dir
        self.raw_data_dir = raw_data_dir
        self.views_per_model = views_per_model
        self.image_size = image_size
        self.transform = transform

        self.model_folders = []
        for model_id in sorted(os.listdir(root_dir)):
            model_path = os.path.join(root_dir, model_id)
            if os.path.isdir(model_path):
                views = sorted([v for v in os.listdir(model_path) if v.startswith("view_")])
                if len(views) >= views_per_model:
                    self.model_folders.append((model_id, views[:views_per_model]))
        
        print(f'Collected {len(self.model_folders)} samples')
    

    def extract_camera_label(self, prompt):

        prompt = prompt.lower()
        if "front" in prompt:
            return 0
        elif "back" in prompt:
            return 1
        elif "left" in prompt:
            return 2
        elif "right" in prompt:
            return 3
        else:
            return 0  # default to front

    def __len__(self):
        return len(self.model_folders)

    def __getitem__(self, idx):

        model_id, view_folders = self.model_folders[idx]
        model_path = os.path.join(self.root_dir, model_id)
        
        # get global condition image
        cond_img_path = os.path.join(self.raw_data_dir, model_id, 'image.jpg')

        rgb_list, depth_list, direction_list, prompt_list, cam_label_list = [], [], [], [], []

        for view in view_folders:
            view_path = os.path.join(model_path, view)
            rgb = Image.open(os.path.join(view_path, "rgb.png")).convert("RGB").resize(self.image_size)
            depth = Image.open(os.path.join(view_path, "depth.png")).convert("RGB").resize(self.image_size)

            if self.transform:
                rgb = self.transform(rgb)
                depth = self.transform(depth)
            else:
                rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0
                depth = torch.from_numpy(np.array(depth)).permute(2, 0, 1).float() / 255.0

            # camera parameters  view_dir shape: torch.Size([4, 12])
            cam_path_pt = os.path.join(view_path, "camera_transform.pt")
            if os.path.exists(cam_path_pt):
                cam_matrix = torch.load(cam_path_pt).float()
                view_dir = cam_matrix.flatten() / cam_matrix.flatten().norm(p=2)
            else:
                raise FileNotFoundError("camera_transform.pt not found")

            # prompt from camera.json
            cam_json_path = os.path.join(view_path, "camera.json")
            if os.path.exists(cam_json_path):
                with open(cam_json_path, "r") as f:
                    camera_data = json.load(f)
                    prompt = camera_data.get("prompt")
            else:
                prompt = "a modern furniture"

            cam_label = self.extract_camera_label(prompt)

            rgb_list.append(rgb)
            depth_list.append(depth)
            direction_list.append(view_dir)
            prompt_list.append(prompt)
            cam_label_list.append(cam_label)

        return {
            "rgb": torch.stack(rgb_list),
            "depth": torch.stack(depth_list),
            "view_dir": torch.stack(direction_list),
            "prompt": prompt_list,
            "camera_pos_label": torch.tensor(cam_label_list, dtype=torch.long),
            "model_id": model_id,
            "cond_img_path": cond_img_path # str
        }

def get_dataloader(root, batch_size, views_per_model=4, transform=None):

    assert batch_size % views_per_model == 0
    dataset = Depth2ImgFineTuneDataset(
        root_dir=root,
        views_per_model=views_per_model,
        transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size // views_per_model, shuffle=False, num_workers=4)

if __name__ == "__main__":

    dataset_root = '/scratch/leuven/375/vsc37593/finetune_depth2img_3dfuture'
    test_dataset = Depth2ImgFineTuneDataset(
        root_dir=dataset_root
    )

    sample = test_dataset[9]

    print("model_id:", sample['model_id'])
    print("prompt:", sample['prompt'])  # list[str], 每个 view 一个 prompt
    print("rgb shape:", sample['rgb'].shape)         # [V, 3, H, W]
    print("depth shape:", sample['depth'].shape)     # [V, 1, H, W]
    print("camera_pos_label:", sample['camera_pos_label'])  # list[int],
    print("view_dir shape:", sample['view_dir'].shape)  # [V, 12]
    print(sample['cond_img_path'])


    # print(sample['view_dir'])

    