import os
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".gitignore"],
    pythonpath=True,
    dotenv=True,
)
#print(root)
import torch
import math
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from src.datamodule.coarse_clip_condition_dataset import CoarseBaseDataset
from src.datamodule.fine_clip_condition_dataset import FineBaseDataset
from src.dist_utils.device_utils import *

class Future3DDataModule:

    def __init__(self,
                 data_split,
                 data_detail,
                 dataset_type: str,
                 batch_size: int,
                 num_workers,
                 pin_memory: bool,
                 is_distributed = True): # turn on when using gpu
        
        super().__init__()
        if dataset_type == 'coarse':
            DatasetClass = CoarseBaseDataset
        elif dataset_type == 'fine':
            DatasetClass = FineBaseDataset
        else:
            raise ValueError(f'Unknown dataset_type: {dataset_type}')
        
        self.data_detail = data_detail # args for BaseDataset
        self.dataset_train = DatasetClass(data_split.train_split, data_detail, mode='train')
        self.dataset_val = DatasetClass(data_split.val_split, data_detail, mode='val')
        self.dataset_test = DatasetClass(data_split.test_split, data_detail, mode='test')

        if is_distributed:
        # prepare data sampler for multiple GPU 
            num_tasks = get_world_size()
            global_rank = get_rank()
            self.sampler_train = torch.utils.data.DistributedSampler(
                self.dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            self.sampler_val = torch.utils.data.SequentialSampler(self.dataset_val)
            self.sampler_test = SequentialDistributedSampler(self.dataset_test, batch_size)
        else:
            self.sampler_train = torch.utils.data.SequentialSampler(self.dataset_train)
            self.sampler_val = torch.utils.data.SequentialSampler(self.dataset_val)
            self.sampler_test = torch.utils.data.SequentialSampler(self.dataset_test)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            sampler=self.sampler_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            sampler=self.sampler_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            sampler=self.sampler_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    
    #TODO: debugging
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate


    @hydra.main(config_path="../../configs/datamodule", config_name="future3d_image_coarse", version_base=None)
    def run(cfg: DictConfig):
        print("Loading config：")
        print(OmegaConf.to_yaml(cfg))

        datamodule = instantiate(cfg)

        # run dataloader
        # train_loader = datamodule.train_dataloader()
        # for batch in train_loader:
        #     print("train one batch")
        #     break

    run()


