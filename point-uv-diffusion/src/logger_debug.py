import os
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".gitignore"],
    pythonpath=True,
    dotenv=True,
)
import torch
import torch.distributed as dist
from time import sleep
from src.logger.baselogger import MyBaseLogger

def main():

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    logger = MyBaseLogger(
        log_dir='./test_log',
        log_freq=2,
        log_img_freq=5,
        save_ckpt_freq=10
    )
    # ----------------------------
    # Simulate training loop
    # ----------------------------
    for step in range(1, 11):
        # Simulated training statistics
        dummy_stat = {
            'loss': 1.0 / step,
            'accuracy': step * 0.1
        }

        logger.print_stat(dummy_stat, step)
        #sleep(0.5)  # Just to simulate a real training delay

    # Cleanup
    dist.destroy_process_group()



if __name__ == "__main__":
    main()

    # import pynvml

    # pynvml.nvmlInit()
    # device_count = pynvml.nvmlDeviceGetCount()

    # for i in range(device_count):
    #     handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    #     name = pynvml.nvmlDeviceGetName(handle)  # Already str in Python 3

    #     try:
    #         util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    #         meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #         print(f"GPU {i} ({name}): util={util.gpu}%, mem_used={meminfo.used/1024**2:.2f} MB")
    #     except pynvml.NVMLError as e:
    #         print(f"GPU {i} ({name}): Error - {e}")