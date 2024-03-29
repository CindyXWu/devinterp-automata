import hydra
from omegaconf import OmegaConf

from scripts.create_sweep import load_config
from di_automata.config_setup import MainConfig
from di_automata.train_utils import Run

# CHANGE THESE  
config_filename = "main_config"
sweep_filename = ""

# # Legacy code: drop use of ConfigStore to make Pydantic play nice with Hydra
# cs = ConfigStore.instance()
# cs.store(name="config_base", node=HydraConf)


@hydra.main(config_path="configs/", config_name=config_filename)
def main(config: MainConfig) -> None:
    """config is typed as MainConfig for duck-typing, but during runtime it's actually an OmegaConf object.
    
    MainConfig class (or any class used as type hint for config parameter) doesn't restrict what keys can be in config. It provides additional information to editor and Hydra's instantiate function. Contents of this object are determined entirely by config files and command line arguments.
    """
    # if config.wandb_config.sweep:
    #     config = update_with_wandb_config(config, sweep_params)
    
    # Convert OmegaConf object to dictionary before passing into Pydantic
    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, False)
    # Convert OmegaConf object to MainConfig Pydantic model for dynamic type validation - NECESSARY DO NOT SKIP
    pydantic_config = MainConfig(**config)
    # Convert back to OmegaConf object for compatibility with existing code
    omegaconf_config = OmegaConf.create(pydantic_config.model_dump())

    run = Run(omegaconf_config)
    run.train()
    run.finish_run()


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