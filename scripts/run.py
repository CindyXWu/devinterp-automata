import os
import hydra
import wandb
from hydra.core.config_store import ConfigStore
from hydra_zen import builds, instantiate

from di_automata.create_sweep import load_config
from di_automata.config_setup import MainConfig
from di_automata.train_utils import Run, update_with_wandb_config
from di_automata.config_setup import MainConfig


os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CHANGE THESE  
config_filename = "main_config"
sweep_filename = ""

# Use hydra-zen library to make Pydantic play nice with Hydra
HydraConf = builds(MainConfig, populate_full_signature=True)
cs = ConfigStore.instance()
cs.store(name="config_base", node=HydraConf)
    
@hydra.main(config_path="configs/", config_name=config_filename, version_base=None)
def main(config_temp: HydraConf) -> None:
    """config is typed as MainConfig for duck-typing, but during runtime it's actually an OmegaConf object.
    
    MainConfig class (or any class used as type hint for config parameter) doesn't restrict what keys can be in config. It provides additional information to editor and Hydra's instantiate function. Contents of this object are determined entirely by config files and command line arguments.
    """
    config: MainConfig = instantiate(config_temp)
    # if config.wandb_config.sweep:
    #     config = update_with_wandb_config(config, sweep_params)
    run = Run(config)
    run.train()


# def run_sweep():
#     """Call main function using WandB syntax."""
#     sweep_id = wandb.sweep(
#         sweep=sweep_config,
#         project=config['wandb_config']['wandb_project_name'],
#     )
#     wandb.agent(sweep_id, function=main, count=config['wandb_config']['sweep_num'])
    
    
if __name__ == "__main__":
    # config: dict = load_config(f"configs/{config_filename}.yaml")
    # if config['wandb_config']['sweep']:
    #     sweep_config = load_config(f"configs/sweep/{sweep_filename}.yaml")
    #     sweep_params = list(sweep_config['parameters'].keys()) # Needs to be a global
    #     run_sweep()
    # else:
    #     main()
    main()