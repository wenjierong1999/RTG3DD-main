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

def train(args):

    device = utils.distributed_init(args)

    #print(device)

    # instantiate datamodule
    datamodule = hydra.utils.instantiate(args.datamodule)
    # DEBUG:
    train_loader = datamodule.train_dataloader()
    print(datamodule.dataset_train.__len__())

    # for batch in train_loader:
    #     print("First batch:")
    #     for key, value in batch.items():
    #         print(f"Key: {key}")
    #         if isinstance(value, dict):
    #             for sub_key, sub_value in value.items():
    #                 if isinstance(sub_value, torch.Tensor):
    #                     print(f"  ├─ {sub_key}: Tensor {sub_value.shape}, dtype={sub_value.dtype}")
    #                 elif isinstance(sub_value, np.ndarray):
    #                     print(f"  ├─ {sub_key}: np.ndarray {sub_value.shape}, dtype={sub_value.dtype}")
    #                 else:
    #                     print(f"  ├─ {sub_key}: {type(sub_value)} -> {str(sub_value)[:100]}")
    #         elif isinstance(value, torch.Tensor):
    #             print(f"  └─ Tensor {value.shape}, dtype={value.dtype}")
    #         else:
    #             print(f"  └─ {type(value)} -> {str(value)[:100]}")
    #     break

    #instantiate modelmodule
    modelmodule = hydra.utils.instantiate(args.model, local_rank=args.local_rank, device=device)

    modelmodule.net = torch.nn.parallel.DistributedDataParallel(modelmodule.net, device_ids=[args.local_rank],
                                                         find_unused_parameters=False)
    modelmodule.net_without_ddp = modelmodule.net.module

    trainer = hydra.utils.instantiate(
        args.trainer,
        ckpt_resume=args.ckpt_resume,
        device=device,
        modelmodule=modelmodule,
    )

    command_line = get_command()
    trainer.logger.save_command(command_line)
    save_yaml_path = os.path.join(trainer.logger.log_dir, 'config.yaml')
    save_config(args, save_yaml_path)

    if args.get("train"):
        trainer.train(modelmodule=modelmodule, datamodule=datamodule, ckpt_path=None)


def main(config_path='../configs', config_name='demo'):

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
        # Compose the final config by merging base config and CLI overrides
        args = hydra.compose(config_name=config_name, overrides=overrides_list)

    # Check if we are resuming from a previous run (args.config_load is a folder path)
    config_load = args.config_load
    if config_load:
        config_load = args.paths.output_dir
        # Load the saved config from disk
        args = OmegaConf.load(config_load + '/config.yaml')
        # Set checkpoint path to resume training
        args.ckpt_resume = os.path.join(config_load, 'ckpt')
        # Make sure config_load stays in the config for future reference
        args.config_load = config_load
    
    #print(args)

    # Start the training loop with the fully composed and ready config
    train(args) # main training function


if __name__ == "__main__":
    main()
