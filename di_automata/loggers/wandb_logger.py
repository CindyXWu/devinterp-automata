from typing import Optional, Sequence, Union, Dict
import numpy.typing as npt

import wandb
from di_automata.loggers.logger import LoggerBase


class WandbLogger(LoggerBase):
    def __init__(self, **wandb_params) -> None:
        super().__init__()
        import wandb
        self._run = wandb.init(**wandb_params)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        wandb.log({name: value}, step=self._step if step is None else step)

    def log_scalars(self, name: str, values: Union[Sequence[float], npt.NDArray], step: Optional[int] = None):
        wandb.log({name: wandb.Histogram(values)}, step=self._step if step is None else step)
        
    def log_dictionary(self, dictionary: Dict[str, float], step: Optional[int] = None):
        wandb.log(dictionary, step=self._step if step is None else step)
        
    def add_config(self, name, val):
        """Adds a config value that allows run filtering - wandb only."""
        setattr(wandb.config, name, val)

    def __del__(self):
        wandb.finish()