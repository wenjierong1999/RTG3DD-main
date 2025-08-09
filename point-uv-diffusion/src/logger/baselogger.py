import psutil
import pynvml
from pytorch_lightning.utilities import rank_zero_only
import torchvision
import torch.distributed as dist
import os
from typing import Dict

class MyBaseLogger:
    '''
    Customized logger that tracks training progress,
    this section is modified to support monitoring memory usage and GPU resource usage.
    '''


    def __init__(self, log_dir, log_freq, log_img_freq, save_ckpt_freq):

        self.log_dir = log_dir
        self.log_freq = log_freq
        self.log_img_freq = log_img_freq
        self.save_ckpt_freq = save_ckpt_freq
        self.rank = int(dist.get_rank())
        self.set_up()

        # initialize NVIDIA Management Library (for GPU monitoring)
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            print(f"[Rank {self.rank}] NVML init failed: {e}")
    
    @rank_zero_only       
    def set_up(self):
        """
        Creates necessary directories for logs, images, and checkpoints.
        Only executed once (rank 0).
        """
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'image'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'ckpt'), exist_ok=True)
        filename = os.path.join(self.log_dir, 'mylog.log')
        self.file = open(filename, "wt")
        self.image_dir = os.path.join(self.log_dir, 'image')
        self.save_ckpt_dir = os.path.join(self.log_dir, 'ckpt')


    def get_gpu_stats(self) -> Dict[str, float]:
        """
        Collects usage stats for all available GPUs using NVIDIA's NVML.
        Returns a dictionary mapping each GPU's utilization and memory usage.
        """
        gpu_stats = {}
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_stats[f'GPU{i}_util_%'] = util.gpu
                gpu_stats[f'GPU{i}_mem_used_MB'] = round(meminfo.used / 1024**2, 2)
                gpu_stats[f'GPU{i}_mem_total_MB'] = round(meminfo.total / 1024**2, 2)
                gpu_stats[f'GPU{i}_mem_%'] = round(100 * meminfo.used / meminfo.total, 1)
        except pynvml.NVMLError as e:
            gpu_stats['GPU_error'] = str(e)

        return gpu_stats

    @rank_zero_only
    def save_command(self, command_line):
        filename = os.path.join(self.log_dir, 'command.txt')
        with open(filename, "wt") as file:
            file.write(command_line)
    
    def writekvs(self, kvs: Dict[str, float]):
        """
        Writes key-value stats as a single line per step (CSV-style), with header.
        Only rank 0 logs.
        """
        # Sort keys for consistent column order
        keys = sorted(kvs.keys(), key=lambda k: k.lower())

        # Define log file
        filename = os.path.join(self.log_dir, 'mylog_tabular.log')

        # Check if we need to write header (file doesn't exist yet)
        write_header = not os.path.exists(filename)

        with open(filename, "a") as f:
            if write_header:
                header_line = "\t".join(keys)
                f.write(header_line + "\n")
                #print(header_line)  # optional: print to terminal

            # Format values
            val_line = "\t".join([f"{kvs[k]:.4g}" if hasattr(kvs[k], "__float__") else str(kvs[k]) for k in keys])
            f.write(val_line + "\n")
            #print(val_line)  # optional: print to terminal
    
    def print_stat(self, stat: Dict[str, float], step: int):
        """
        Main logging function.
        At the specified frequency, logs system RAM, all GPU usage stats, and user-provided training metrics.
        Only active for rank 0.
        """
        if step % self.log_freq != 0 or self.rank != 0:
            return

        stat = dict(stat)  # Copy to avoid side effects
        stat['step'] = step
        stat['rank'] = self.rank

        # RAM usage stats
        mem = psutil.virtual_memory()
        stat['RAM_used_MB'] = round((mem.total - mem.available) / 1024**2, 2)
        stat['RAM_percent'] = mem.percent

        # Append all GPU stats
        stat.update(self.get_gpu_stats())

        # Write to disk + print
        self.writekvs(stat)
    
    @rank_zero_only
    def save_image(self, image_batch, step, name=''):
        """
        Save a batch of images as a single image grid, only on rank 0.

        Args:
            image_batch (Tensor): Tensor of shape (B, C, H, W) in [0, 1] or normalized format
            step (int): Current training step, used in filename
            name (str): Optional name prefix
        """
        if self.rank == 0 and step % self.log_img_freq == 0 and image_batch is not None:
            filename = os.path.join(self.log_dir, 'image', f'{name}{step:08d}.jpg')
            torchvision.utils.save_image(image_batch.detach().cpu(), filename)

    def _truncate(self, s: str) -> str:
        """
        Truncates a string if it's too long for display.
        """
        return s[:27] + "..." if len(s) > 30 else s