"""Construct models, get class numbers, get alpha, construct dataloaders."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from typing import Tuple, Optional
from dataclasses import asdict
import datasets
import numpy as np
import logging
import omegaconf
import datasets
from omegaconf import OmegaConf
from functools import partial

import os

from di_automata.datasets.automata import AutomatonDataset
from di_automata.config_setup import MainConfig, ModelType, DatasetType, DatasetConfig, OptimizerType
from di_automata.architectures.nano_gpt import Transformer
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


def create_dataloader_hf(config: MainConfig) -> DataLoader:
    """Load dataset from automata.py using HuggingFace architecture.
    
    Provided code only has train split.
    """
    automaton_dataset = AutomatonDataset(config.dataset_config)
    dataset = datasets.load_dataset(automaton_dataset)
    train_loader = DataLoader(dataset['train'], batch_size=config.dataloader_config.train_bs, shuffle=True, drop_last=True)
    return train_loader


def create_or_load_dataset(dataset_type: str, dataset_config: DatasetConfig) -> Dataset:
    """Create or load an existing dataset based on a specified filepath and dataset type."""
    filepath = f'{dataset_config.data_folder}/{dataset_config.filename}.pt'
    if os.path.exists(filepath):
        dataset = torch.load(filepath)
    else:
        dataset_type = globals()[dataset_type]
        dataset = dataset_type(dataset_config)
        torch.save(dataset, filepath)
    return dataset
        

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


class LRScheduler(object):
    def __init__(self, optimizer, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
        decay_iter = iter_per_epoch * num_epochs
        self.lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))        
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr

    def get_lr(self):
        return self.current_lr


def optimizer_constructor(
    config: MainConfig,
    model: nn.Module,
    train_loader: DataLoader) -> optim.Optimizer:
    match config.optimization.optimizer_type:
        case OptimizerType.SGD:
            optim_constructor = torch.optim.SGD
        case OptimizerType.ADAM:
            optim_constructor = torch.optim.Adam
        case _:
            raise ValueError(f"Unknown optimizer type: {config.optimization.optimizer_type}")
    optim = optim_constructor(
        params=model.parameters(),
        lr=config.optimization.base_lr,
        **config.optimization.optimizer_kwargs,
    )
    
    if config.optimization.cosine_lr_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optim,
            T_max=config.num_iters,
        )
        scheduler = LRScheduler(optim, config.epochs, base_lr=config.optimization.base_lr, final_lr=config.optimization.final_lr, iter_per_epoch=len(train_loader))
        
    return optim, scheduler