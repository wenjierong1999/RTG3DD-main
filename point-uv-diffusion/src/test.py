import os
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".gitignore"],
    pythonpath=True,
    dotenv=True,
)
import torch
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from src.dist_utils import device_utils
from src.dist_utils.config_utils import *
from src.dist_utils import utils
from pathlib import Path
import glob

def compile_nvdiffrast_once():
    import nvdiffrast.torch as dr
    print("Compiling nvdiffrast...")
    glctx = dr.RasterizeCudaContext()
    print("Compilation finished")

def test(args):

    #compile_nvdiffrast_once()

    #print(args['test'])

    mode = args['test'].get("mode", None)
    print(f"Using inference mode: {mode}")
    device = 'cuda'

    #  instantiate evaluator
    evaluator = hydra.utils.instantiate(
        args['test'].evaluator,
        coarse_ckpt = args['test'].get('coarse_ckpt'),
        fine_ckpt = args['test'].get('fine_ckpt'),
        coarse_ckpt_resume = args['test'].get('coarse_ckpt_resume'),
        fine_ckpt_resume = args['test'].get('fine_ckpt_resume')
    )
    print("evaluator is instantiated...")

    #  === Mode: Coarse ===
    if mode == "coarse":
        if args['coarse_train'] is None:
            raise ValueError("Missing coarse_train config for coarse mode.")

        datamodule = hydra.utils.instantiate(
            args['coarse_train'].datamodule,
            batch_size=args['test'].get("test_batch_size", 1),
            is_distributed=False,
            num_workers = 2,
            data_detail = {
                'test_samples':args['test'].get('test_samples', False)
            }
            )
        print("coarse datamodule is instantiated...")
        print(f"test sample size: {datamodule.dataset_test.__len__()}")

        modelmodule = hydra.utils.instantiate(args['coarse_train'].model, device=device)
        print("coarse modelmodule is instantiated...")

        # main inference step
        evaluator.test_coarse(
            datamodule=datamodule,
            modelmodule=modelmodule
        )
    #  === Mode: Fine ===
    elif mode == "fine":
        if args['fine_train'] is None:
            raise ValueError("Missing fine_train config for fine mode.")

        datamodule = hydra.utils.instantiate(
            args['fine_train'].datamodule,
            batch_size=args['test'].get("test_batch_size", 1),
            is_distributed=False,
            num_workers=2,
            data_detail = {
                'test_samples':args['test'].get('test_samples')
            }
            )
        print("fine datamodule is instantiated...")
        print(f"test sample size: {datamodule.dataset_test.__len__()}")
        
        modelmodule = hydra.utils.instantiate(args['fine_train'].model, device=device)
        print("fine modelmodule is instantiated")

        #  main inference step
        evaluator.test_fine(
            datamodule=datamodule,
            modelmodule=modelmodule
        )

    elif mode == "joint":
        if args['coarse_train'] is None or args['fine_train'] is None:
            raise ValueError("Joint mode requires both coarse and fine configs.")

        # === 1. Instantiate coarse datamodule & model ===
        coarse_dm = hydra.utils.instantiate(
            args['coarse_train'].datamodule,
            batch_size=args['test'].get("test_batch_size", 1),
            is_distributed=False,
            num_workers=2,
            data_detail={
                'test_samples': args['test'].get('test_samples')
            }
        )
        print(f"[joint] coarse datamodule loaded. #samples: {len(coarse_dm.dataset_test)}")

        coarse_model = hydra.utils.instantiate(args['coarse_train'].model, device=device)
        print("[joint] coarse modelmodule instantiated.")

        coarse_time, coarse_samples = evaluator.test_coarse(
            datamodule=coarse_dm,
            modelmodule=coarse_model
        )

        # === 2. Instantiate fine datamodule & model ===
        # Make sure the fine datamodule points to coarse output as finetune_folder
        fine_finetune_folder = Path(evaluator.coarse_ckpt) / 'coarse_output' / 'save_fpsuv_4096'

        fine_dm = hydra.utils.instantiate(
            args['fine_train'].datamodule,
            batch_size=args['test'].get("test_batch_size", 1),
            is_distributed=False,
            num_workers=2,
            data_detail={
                'test_samples': args['test'].get('test_samples'),
                'use_stage1_test': True,
                'finetune_folder': str(fine_finetune_folder)
            }
        )
        print(f"[joint] fine datamodule loaded. #samples: {len(fine_dm.dataset_test)}")

        fine_model = hydra.utils.instantiate(args['fine_train'].model, device=device)
        print("[joint] fine modelmodule instantiated.")

        fine_time, fine_samples = evaluator.test_fine(
            datamodule=fine_dm,
            modelmodule=fine_model
        )
        
        total_time = coarse_time + fine_time
        total_samples = max(coarse_samples, fine_samples)  #  coarse==fine
        joint_avg = total_time / total_samples if total_samples > 0 else 0.0
        print(f"[joint] Average end-to-end inference time per sample: {joint_avg:.4f} seconds")

        # === 3. run joint inference procedure ===
        # evaluator.joint_test(
        #     coarse_dm = coarse_dm,
        #     coarse_model = coarse_model,
        #     fine_dm = fine_dm,
        #     fine_model = fine_model
        # )
    else:
        raise ValueError(f'Unsupported mode {mode}')
        

def auto_find_ckpt_resume_path(ckpt_store_path):

    ckpt_store_path = Path(ckpt_store_path)

    if not ckpt_store_path.is_dir():
        raise ValueError("Wrong/Missing ckpt file")
    latest_path = ckpt_store_path / "checkpoint-latest.pth"
    if latest_path.is_file():
        return latest_path

    all_checkpoints = glob.glob(str(ckpt_store_path / "checkpoint-*.pth"))
    latest_ckpt = -1
    for ckpt in all_checkpoints:
        ckpt_name = Path(ckpt).stem  # "checkpoint-42"
        suffix = ckpt_name.split('-')[-1]
        if suffix.isdigit():
            latest_ckpt = max(int(suffix), latest_ckpt)

    if latest_ckpt >= 0:
        return ckpt_store_path / f'checkpoint-{latest_ckpt}.pth'
    else:
        raise FileNotFoundError("No valid ckpts found")
    
# res = auto_find_ckpt_resume_path('/data/leuven/375/vsc37593/my_py_projects/point-uv-diffusion-dev/results/cond_fine_train_chair/demo-Chair/static_timestamped/ckpt')
# print(res)

def main(config_path="../configs", config_name="test"):

    # Register the new resolver
    OmegaConf.register_new_resolver("dir_resolver", dir_resolver)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y)
    # set configuration
    cli_args = OmegaConf.to_container(OmegaConf.from_cli(), resolve=True)
    
    cli_args['local_rank'] = int(os.environ.get("LOCAL_RANK", -1)) # we get local_rank from enviroment variable "LOCAL_RANK"

    # # Flatten nested dictionaries (e.g. {"model": {"lr": 0.01}} -> {"model.lr": 0.01})
    flat_cli_args = flatten_dict(cli_args)
    # Create a list of override strings for Hydra compose (e.g. ["model.lr=0.01"])
    overrides_list = [f'{k}={v}' for k, v in flat_cli_args.items()]

    with hydra.initialize(version_base=None, config_path=config_path):
        arg_test = hydra.compose(config_name=config_name, overrides=overrides_list) # overrides is a list
    
    coarse_results_path = arg_test.get('coarse_ckpt', False)
    fine_results_path = arg_test.get('fine_ckpt', False)

    # ========== Load coarse model config & ckpt ==========
    if isinstance(coarse_results_path, str):
        if Path(coarse_results_path).is_dir():
            coarse_cfg_path = Path(coarse_results_path) / "config.yaml"
            arg_test['coarse_cfg'] = str(coarse_cfg_path)
            arg_test['coarse_ckpt_resume'] = str(auto_find_ckpt_resume_path(Path(coarse_results_path) / "ckpt"))
            args_train_coarse = OmegaConf.load(coarse_cfg_path)
        else:
            raise ValueError(f"Coarse ckpt path was provided, but not a valid directory: {coarse_results_path}")
    else:
        print("[INFO] coarse_ckpt not provided, skipping coarse model.")
        args_train_coarse = None
        # arg_test['coarse_cfg'] = False
        # arg_test['coarse_ckpt_resume'] = False

    # ========== Load fine model config & ckpt ==========
    if isinstance(fine_results_path, str):
        if Path(fine_results_path).is_dir():
            # If it's a valid directory, proceed
            fine_cfg_path = Path(fine_results_path) / "config.yaml"
            arg_test['fine_cfg'] = str(fine_cfg_path)
            arg_test['fine_ckpt_resume'] = str(auto_find_ckpt_resume_path(Path(fine_results_path) / "ckpt"))
            args_train_fine = OmegaConf.load(fine_cfg_path)
        else:
            # User provided path, but directory doesn't exist
            raise ValueError(f"Fine ckpt path was provided, but not a valid directory: {fine_results_path}")
    else:
        # fine_ckpt is None or not set — skip silently or log if needed
        print("[INFO] fine_ckpt not provided, skipping fine model.")
        args_train_fine = None
    
    args = {
        'test': arg_test,
        'coarse_train': args_train_coarse,
        'fine_train': args_train_fine
    }

    #print(args)

    test(args=args)

if __name__ == "__main__":
    main()