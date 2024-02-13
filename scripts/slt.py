import hydra
from omegaconf import OmegaConf

from di_automata.config_setup import MainConfig
from di_automata.slt_analysis import PostRunSLT

# CHANGE THESE  
config_filename = "main_config"
sweep_filename = ""


@hydra.main(config_path="configs/", config_name=config_filename)
def main(config: MainConfig) -> None:
    """config is typed as MainConfig for duck-typing, but during runtime it's actually an OmegaConf object.
    """

    # Convert OmegaConf object to dictionary before passing into Pydantic
    OmegaConf.resolve(config)
    # Convert OmegaConf object to MainConfig Pydantic model for dynamic type validation - NECESSARY DO NOT SKIP
    pydantic_config = MainConfig(**config)
    # Convert back to OmegaConf object for compatibility with existing code
    omegaconf_config = OmegaConf.create(pydantic_config.dict())

    post_run = PostRunSLT(omegaconf_config)
    if config.ed_cp:
        post_run._ed()
    if config.llc_cp:
        post_run._rlct()
    
    
if __name__ == "__main__":
    main()