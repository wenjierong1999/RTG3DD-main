import os
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".gitignore"],
    pythonpath=True,
    dotenv=True,
)
import torch
from src.dist_utils import device_utils
from src.dist_utils.config_utils import *
from src.dist_utils import utils
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pathlib import Path
import torch.distributed as dist

slurm_job_id = os.environ.get("SLURM_JOB_ID", "manual")
ext_dir = f"/scratch/leuven/375/vsc37593/{os.environ['USER']}/torch_ext_{slurm_job_id}"
os.environ["TORCH_EXTENSIONS_DIR"] = ext_dir
print(f"[INFO] TORCH_EXTENSIONS_DIR set to: {ext_dir}")


def compile_nvdiffrast_once():
    import nvdiffrast.torch as dr
    print(f"[rank {dist.get_rank()}] Compiling nvdiffrast...")
    glctx = dr.RasterizeCudaContext()
    # dummy_pos = torch.rand(1, 3, 4, device='cuda')
    # dummy_tri = torch.randint(0, 4, (1, 1, 3), dtype=torch.int32, device='cuda')
    #dr.rasterize(dummy_pos, dummy_tri, resolution=[64, 64])
    print(f"[rank {dist.get_rank()}] Compilation finished")

def maybe_compile_nvdiffrast():
    if dist.is_initialized() and dist.get_rank() == 0:
        compile_nvdiffrast_once()
    if dist.is_initialized():
        dist.barrier()  

#TODO: CUDA error when ckpt_resume is not False

def train(args):

    device = utils.distributed_init(args)

    maybe_compile_nvdiffrast()

    # instantiate datamodule
    datamodule = hydra.utils.instantiate(args.datamodule)
    print(f'Train dataset size: {datamodule.dataset_train.__len__()}')
    print('Datamodule is instantiated...')

    #args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    #instantiate modelmodule

    print(f'local rank: {args.local_rank}')
    modelmodule = hydra.utils.instantiate(args.model, local_rank=args.local_rank, device=device)
    print('modelmodule is instantiated...')
    #print(f"[rank: {torch.distributed.get_rank()}] Before DDP wrapping")
    modelmodule.net = torch.nn.parallel.DistributedDataParallel(modelmodule.net, device_ids=[args.local_rank],
                                  find_unused_parameters=False)
    print('modelmodule.net is instantiated...')
    modelmodule.net_without_ddp = modelmodule.net.module
    print('modelmodule.net_without_ddp is instantiated...')

    # instantiate trainer
    print(f'ckpt_resume:{args.ckpt_resume}')
    trainer = hydra.utils.instantiate(
        args.trainer,
        ckpt_resume=args.ckpt_resume,
        device=device,
        modelmodule=modelmodule,
    )
    print('trainer is instantiated...')

    # save command line and configure file (for resuming training)
    command_line = get_command()
    trainer.logger.save_command(command_line)
    save_yaml_path = os.path.join(trainer.logger.log_dir, 'config.yaml')
    save_config(args, save_yaml_path)
    print('config file is saved...')

    if args.get("train"):
        trainer.train(modelmodule=modelmodule, datamodule=datamodule)

def main(config_path='../configs', config_name='train'):

    # register custom resolvers for dynamic config values in YAML
    OmegaConf.register_new_resolver("dir_resolver", dir_resolver) # see config_utils.dir_resolver()
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y) # define a sum function

    # Parse command-line arguments passed via Hydra (e.g. model.lr=0.001)
    cli_args = OmegaConf.from_cli()

    # Optional: Add 'task_name' if 'experiment' is provided via CLI
    task_name = cli_args.get("experiment")
    if task_name is not None:
        cli_args["task_name"] = task_name

    # Convert CLI args to a flat dict and handle '--local_rank' from torchrun
    cli_args_dict = OmegaConf.to_container(cli_args, resolve=True)
    # Extract --local_rank (passed by torchrun) and convert to 'local_rank' key
    # Not working for torchrun
    #cli_args_dict['local_rank'] = cli_args_dict.pop("--local_rank", None)

    cli_args_dict['local_rank'] = int(os.environ.get("LOCAL_RANK", -1)) # we get local_rank from enviroment variable "LOCAL_RANK"


    # Flatten nested dictionaries (e.g. {"model": {"lr": 0.01}} -> {"model.lr": 0.01})
    flat_cli_args = flatten_dict(cli_args_dict) # see config_utils.flatten_dict
    # Create a list of override strings for Hydra compose (e.g. ["model.lr=0.01"])
    overrides_list = [f'{k}={v}' for k, v in flat_cli_args.items()]

     # Initialize Hydra config system manually
    with hydra.initialize(version_base=None, config_path=config_path, job_name="test_app"):
        # Compose the base args
        base_config = hydra.compose(config_name=config_name, overrides=overrides_list)
    
    #print(base_config)

    # chekc ckpt_resume
    ckpt_resume_path = cli_args_dict.get("ckpt_resume", False)
    if ckpt_resume_path and Path(ckpt_resume_path).is_dir():
        config_load = Path(ckpt_resume_path) / "config.yaml"
        if not os.path.exists(config_load):
            raise FileNotFoundError(f"config.yaml not found at {config_load}")
        print(f"[INFO] Resuming config from: {config_load}")
        resumed_config = OmegaConf.load(config_load)

        final_args = OmegaConf.merge(resumed_config, cli_args)
        final_args.ckpt_resume = Path(ckpt_resume_path) / "ckpt"
        final_args.config_load = config_load
    else:
        final_args = base_config
    
    # note that local_rank is overrided by the old config file!!!
    final_args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # print(f"[DEBUG] Final device: {final_args.device}")
    # print(f"[DEBUG] local_rank: {final_args.local_rank}")
    # print(f"[DEBUG] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Using Batch size: {final_args.datamodule.batch_size}")
    # Start the training loop with the fully composed and ready config
    train(final_args) # main training function

if __name__ == "__main__":
    main()
