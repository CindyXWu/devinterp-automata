from typing import Union, Optional
import numpy as np
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from di_automata.config_setup import (
    MainConfig,
    ModelType,
    ParameterisationType,
    OptimizerType,
    RLCTLossType,
)
from di_automata.losses import predictive_kl_loss, ce_rlct_loss
from di_automata.architectures.nano_gpt import Transformer
from di_automata.tasks.data_utils import TorchDatasetFromIterable
from di_automata.mup.inf_types import InfParam, get_inf_types, get_params_without_init
from di_automata.mup.init import (
    mup_initialise,
    scale_init_inplace,
    standard_param_initialise,
    torch_param_initialise,
)
from di_automata.mup.optim_params import (
    get_adam_param_groups,
    get_mup_sgd_param_groups,
)
from di_automata.mup.utils import get_param_name


def create_dataloader_hf(config: MainConfig, deterministic: Optional[bool] = False) -> DataLoader:
    """Load dataset from automata.py.
    
    Note the Automata dataset class automatically handles which instance of which dataclass it is based on the config parameters.
    """
    # Wrap generator with custom IterableDataset which instantiates a new automaton dataset instance per epoch
    iterable_dataset = TorchDatasetFromIterable(config, deterministic)
    train_loader = DataLoader(iterable_dataset, batch_size=config.dataloader_config.train_bs)
    return train_loader


def construct_model(config: MainConfig) -> tuple[nn.Module, dict[str, InfParam]]:
    """Return model and tensor program initialisation values."""
    if config.model_type == ModelType.NANO_GPT:
        model = Transformer(config=config.nano_gpt_config)
        param_inf_types = get_inf_types(
            model=model,
            input_weights_names=[
                get_param_name(
                    model,
                    model.token_embedding.weight,
                ),
                get_param_name(
                    model,
                    model.pos_embedding.weight,
                    
                ),
            ],
            output_weights_names=[get_param_name(model, model.unembedding[-1].weight)],  # type: ignore
        )
    else:
        raise ValueError(f"Unknown architecture type: {config.architecture_type}")
    return model, param_inf_types


class CustomLRScheduler(object):
    def __init__(self, optim: torch.optim.Optimizer, num_training_iter: int, base_lr: float, final_lr: float):
        self.base_lr = base_lr
        decay_iter = num_training_iter + 1
        self.lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))        
        self.optimizer = optim
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr

    def get_lr(self):
        return self.current_lr
    
    
SchedulerType = Union[torch.optim.lr_scheduler.CosineAnnealingLR, torch.optim.lr_scheduler.StepLR, CustomLRScheduler]


def optimizer_constructor(
    config: MainConfig,
    model: nn.Module,
    param_inf_properties: dict[str, InfParam],
) -> tuple[torch.optim.Optimizer, Optional[SchedulerType]]:
    named_params = list(model.named_parameters())
    param_names = {name for name, _ in named_params}
    
    # Validate that all the per_param_lr parameter names are valid:
    for name in config.optimizer_config.per_param_lr.keys():
        if name not in param_names:
            raise ValueError(
                f"Parameter name '{name}' in 'per_param_lr' is not a valid parameter name."
                f"\nValid parameter names are: {param_names}"
            )
    # Learning rates per param:
    lr_scale_per_param = {
        name: (
            config.optimizer_config.per_param_lr[name]
            if name in config.optimizer_config.per_param_lr.keys()
            else config.optimizer_config.default_lr
        )
        * config.optimizer_config.global_lr
        for name, param in named_params
    }
    if config.optimizer_config.weight_decay > 0:
        raise NotImplementedError(
            "Need to separate out parameter groups so as to not apply weight decay to LayerNorm multipliers and Embedding layers"
        )

    logging.info(f"Base learning rates per parameter: {lr_scale_per_param}")

    match config.optimizer_config.optimizer_type:
        case OptimizerType.SGD:
            optim_constructor = torch.optim.SGD
        case OptimizerType.ADAM:
            optim_constructor = torch.optim.Adam
        case OptimizerType.ADAMW:
            optim_constructor = torch.optim.AdamW
        case _:
            raise ValueError(f"Unknown optimizer type: {config.optimizer_config.optimizer_type}")

    if config.parameterisation == ParameterisationType.MUP:
        match config.optimizer_config.optimizer_type:
            case OptimizerType.SGD:
                param_groups = get_mup_sgd_param_groups(
                    named_params=named_params,
                    init_lr_scale=lr_scale_per_param,
                    param_inf_types=param_inf_properties,
                )
            case OptimizerType.ADAM | OptimizerType.ADAMW:
                param_groups = get_adam_param_groups(
                    named_params=named_params,
                    init_lr_scale=lr_scale_per_param,
                    param_inf_types=param_inf_properties,
                )
            case _:
                raise ValueError(f"Unknown optimizer type: {config.optimizer_config.optimizer_type}")
    else:
        param_groups = [
            {"params": [param], "lr": lr_scale_per_param[name]}
            for name, param in model.named_parameters()
        ]
    optim = optim_constructor(
        params=param_groups,  # type: ignore
        lr=config.optimizer_config.default_lr,
        **config.optimizer_config.optimizer_kwargs,
    )

    if config.optimizer_config.cosine_lr_schedule:
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optim,
        #     T_max=config.num_training_iter,
        # )
        scheduler = CustomLRScheduler(
            optim=optim,
            num_training_iter=config.num_training_iter,
            base_lr=config.optimizer_config.default_lr,
            final_lr=config.optimizer_config.final_lr,
        )
    else:
        scheduler = None
    return optim, scheduler


def initialise_model(
    config: MainConfig, 
    model: nn.Module, 
    param_inf_properties: dict[str, InfParam],
) -> None:
    named_params = list(model.named_parameters())
    param_names = {name for name, _ in named_params}

    params_without_init = get_params_without_init(model=model)
    logging.info(f"Params without mup initialization: {params_without_init}")

    for param_name in config.initialisation.init_scales_per_param.keys():
        if param_name in params_without_init:
            raise ValueError(
                f"This parameter should not have the init. scale overriden: {param_name}"
            )

    init_scales = {
        name: (
            config.initialisation.init_scales_per_param[name]
            if name in config.initialisation.init_scales_per_param.keys()
            else config.initialisation.default_init_scale
        )
        * config.initialisation.global_init_scale
        for name, _ in named_params
    }
    # Validate that all the init_scales_per_param parameter names are valid:
    for name in config.initialisation.init_scales_per_param.keys():
        if name not in param_names:
            raise ValueError(
                f"Parameter name '{name}' in 'init_scales_per_param' is not a valid parameter name."
                f"\nValid parameter names are: {param_names}"
            )
    logging.info(f"Initialisation scales: {init_scales}")
    if config.parameterisation == ParameterisationType.MUP:
        mup_initialise(
            named_params=named_params,
            param_inf_types=param_inf_properties,
            init_scale=init_scales,
            distribution=config.initialisation.init_distribution,
            params_without_init=params_without_init,
        )
    elif config.parameterisation == ParameterisationType.SP:
        # If not using muP, initialise using SP.
        standard_param_initialise(
            named_params=named_params,
            init_scale=init_scales,
            distribution=config.initialisation.init_distribution,
            params_without_init=params_without_init,
        )
    elif config.parameterisation == ParameterisationType.PYTORCH:
        torch_param_initialise(
            named_params=named_params,
            init_scale=init_scales,
            distribution=config.initialisation.init_distribution,
            params_without_init=params_without_init,
        )
    elif config.parameterisation == ParameterisationType.NONE:
        scale_init_inplace(named_params, init_scales)
    else:
        raise ValueError(f"Unknown parameterisation: {config.parameterisation}")


def construct_rlct_criterion(config: MainConfig):
    if config.rlct_config.rlct_loss_type == RLCTLossType.CE:
        return ce_rlct_loss
    elif config.rlct_config.rlct_loss_type == RLCTLossType.DISTILL:
        return predictive_kl_loss