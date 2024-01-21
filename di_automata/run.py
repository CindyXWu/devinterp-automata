import torch
import os
import hydra
import logging
import math
import omegaconf
import wandb
from hydra.core.config_store import ConfigStore
from typing import Dict

from create_sweep import construct_sweep_config, load_config
from train_utils import train, create_dataloaders, create_or_load_dataset, model_constructor, optimizer_constructor
from config_setup import MainConfig


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Register the defaults from the structured dataclass config schema:
cs = ConfigStore.instance()
cs.store(name="config_base", node=MainConfig)


@hydra.main(config_path="configs/", config_name="defaults", version_base=None)
def main(config: MainConfig) -> None:
    """config is typed as MainConfig for duck-typing, but during runtime it's actually an OmegaConf object."""
    logging.info(f"Hydra current working directory: {os.getcwd()}")

    logger_params = {
    "name": f"{config.dataset_type}_{config.dataset.input_length}_{config.model_type}",
    "project": config.wandb_project_name,
    "settings": wandb.Settings(start_method="thread"),
    "config": omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    "mode": "disabled" if not config.log_to_wandb else "online",
    }
    wandb.init(**logger_params)
    # Probably won't do sweeps over these - okay to put here relative to call to update_with_wandb_config() below
    wandb.config.dataset_type = config.dataset_type
    wandb.config.model_type = config.model_type
    
    dataset = create_or_load_dataset(config.dataset_type, config.dataset)
    train_loader, test_loader = create_dataloaders(dataset, config.dataloader)
    
    model = model_constructor(config)
    model.to(DEVICE)
    
    optimizer = optimizer_constructor(config=config, model=model)
    
    try:
        eval_frequency = config.eval_frequency if config.eval_frequency is not None else len(train_loader)
    except TypeError as e:
        msg = f"eval_frequency must be specified if using an iterable train_loader."
        raise TypeError(msg) from e
    
    epochs = math.ceil(config.num_training_iter / eval_frequency)
    train_params = {
        'model': model,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'default_lr': config.optimization.default_lr,
        'epochs': epochs,
        'loss_threshold': config.loss_threshold,
        'num_eval_batches': config.num_eval_batches,
        'optimizer': optimizer,
        'project': config.wandb_project_name,
        'model_save_path': config.model_save_path,
        'device': DEVICE
    }
    train_params = update_with_wandb_config(train_params) # For wandb sweeps: update with wandb values
    train(**train_params)

    print(wandb.config)
    
    # Save teacher model and config as wandb artifacts:
    if config.save_model_as_artifact:
        model_artifact = wandb.Artifact("model", type="model", description="The trained model state_dict")
        model_artifact.add_file(".hydra/config.yaml", name="config.yaml")
        wandb.log_artifact(model_artifact)


def update_with_wandb_config(params_dict: Dict) -> Dict:
    """Check if each parameter exists in wandb.config and update it if it does."""
    for param in params_dict:
        if param in wandb.config:
            print("Updating param: ", param)
            params_dict[param] = wandb.config[param]
    return params_dict


if __name__ == "__main__":
    """May have to edit this hard coding opening one single config file in the future."""
    config: Dict = load_config('configs/defaults.yaml')
    if config.get("sweep"):
        sweep_config = construct_sweep_config('defaults', 'sweep_configs')
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=config.get("wandb_project_name"),
        )
        wandb.agent(sweep_id, function=main, count=10)
    else:
        main()