import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader

# from paint3d import utils

def phi_to_view(phi_deg):
    """
    Strictly map phi angle (in degrees) to a known view.
    Only exact matches (e.g., 0, 90, 180, 270) are accepted.
    """
    phi_rounded = int(round(phi_deg)) % 360
    view_names = {0: "front", 90: "right", 180: "back", 270: "left"}
    label_map = {"front": 0, "back": 1, "left": 2, "right": 3}

    view_name = view_names.get(phi_rounded, None)
    if view_name is None:
        return None, None

    return view_name, label_map[view_name]



def init_dataloaders(cfg, device=torch.device("cpu")):
    init_train_dataloader = MultiviewDataset(cfg.render, device=device).dataloader()
    val_large_loader = ViewsDataset(cfg.render, device=device, size=cfg.log.full_eval_size).dataloader()
    dataloaders = {'train': init_train_dataloader, 'val_large': val_large_loader}
    return dataloaders


class MultiviewDataset:
    def __init__(self, cfg, device):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.type = type  # train, val, tests
        size = self.cfg.n_views
        self.phis = [(index / size) * 360 for index in range(size)]
        self.thetas = [self.cfg.base_theta for _ in range(size)]

        # Alternate lists
        alternate_lists = lambda l: [l[0]] + [i for j in zip(l[1:size // 2], l[-1:size // 2:-1]) for i in j] + [
            l[size // 2]]
        if self.cfg.alternate_views: 
            self.phis = alternate_lists(self.phis)
            self.thetas = alternate_lists(self.thetas)

        for phi, theta in self.cfg.views_before:
            self.phis = [phi] + self.phis
            self.thetas = [theta] + self.thetas
        for phi, theta in self.cfg.views_after:
            self.phis = self.phis + [phi]
            self.thetas = self.thetas + [theta]

        self.size = len(self.phis)

        self.phi_to_label = { 0: (0, "front"), 
                             90: (3, "right"), 
                             180: (1, "back"), 
                             270: (2, "left")}

    def collate(self, index):
        phi = self.phis[index[0]]
        theta = self.thetas[index[0]]
        radius = self.cfg.radius
        thetas = torch.FloatTensor([np.deg2rad(theta)]).to(self.device).item()
        phis = torch.FloatTensor([np.deg2rad(phi)]).to(self.device).item()

        # Check if phi is exact match
        label_prompt = self.phi_to_label.get(int(phi))
        if label_prompt:
            label, view_name = label_prompt
            direction_prompt = f"a modern furniture from {view_name} view"
            camera_label = label
        else:
            direction_prompt = None
            camera_label = -1

        return {'theta': thetas, 
                'phi': phis, 
                'radius': radius,
                'direction_prompt': direction_prompt,
                'camera_label': camera_label}

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=False,
                            num_workers=0)
        loader._data = self  
        return loader

class ViewsDataset:
    def __init__(self, cfg, device, size=100):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.type = type  # train, val, test
        self.size = size

    def collate(self, index):
        phi = (index[0] / self.size) * 360
        thetas = torch.FloatTensor([np.deg2rad(self.cfg.base_theta)]).to(self.device).item()
        phis = torch.FloatTensor([np.deg2rad(phi)]).to(self.device).item()
        return {'theta': thetas,  'phi': phis, 'radius': self.cfg.radius}

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=False,
                            num_workers=0)
        loader._data = self  
        return loader

if __name__ == "__main__":

    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Optional, Tuple, List
    from config.train_config_paint3d import TrainConfig

    cfg = TrainConfig()
    print(cfg.render)

    dataset = MultiviewDataset(cfg.render, device=torch.device("cpu"))
    dataloader = dataset.dataloader()

    print(f"[INFO] Total views in dataset: {len(dataset.phis)}\n")

    for i, batch in enumerate(dataloader):
        print(f"--- Sample {i} ---")
        print(f"Theta (rad): {batch['theta']:.3f}")
        print(f"Phi (rad): {batch['phi']:.3f} | Phi (deg): {np.rad2deg(batch['phi']):.1f}")
        print(f"Radius: {batch['radius']}")
        print(f"Direction Prompt: {batch['direction_prompt']}")
        print(f"Camera Label: {batch['camera_label']}")
        print()
        # if i >= 10:  # Print only first 10
        #     break


