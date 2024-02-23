import hydra
from omegaconf import OmegaConf

from di_automata.config_setup import PostRunSLTConfig
from di_automata.slt_analysis import PostRunSLT

# CHANGE THESE  
config_filename = "slt_config"
sweep_filename = ""


@hydra.main(config_path="configs/", config_name=config_filename)
def main(config: PostRunSLTConfig) -> None:
    """config is typed as MainConfig for duck-typing, but during runtime it's actually an OmegaConf object.
    """

    # Convert OmegaConf object to dictionary before passing into Pydantic
    OmegaConf.resolve(config)
    # Convert OmegaConf object to MainConfig Pydantic model for dynamic type validation - NECESSARY DO NOT SKIP
    pydantic_config = PostRunSLTConfig(**config)
    # Convert back to OmegaConf object for compatibility with existing code
    omegaconf_config = OmegaConf.create(pydantic_config.model_dump())
    
    post_run = PostRunSLT(omegaconf_config)
    if omegaconf_config.ed:
        post_run.do_ed()
        
    if omegaconf_config.llc:
        post_run.calculate_rlct()
    
    if omegaconf_config.form:
        post_run.plot_form_potential()
    
    post_run.finish_run()
    
    
if __name__ == "__main__":
    main()