from typing import Callable, Dict, List, Literal, Optional, Type, Union
import itertools
import inspect
import warnings
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.multiprocessing import cpu_count, get_context
from torch.utils.data import DataLoader, IterableDataset

from di_automata.tasks.data_utils import take_n
from di_automata.config_setup import MainConfig, RLCTConfig
from di_automata.devinterp.optim.sgld import SGLD
from di_automata.devinterp.optim.sgld_ma import SGLD_MA
from di_automata.devinterp.optim.sgnht import SGNHT
from di_automata.constructors import construct_model
from di_automata.losses import predictive_kl_loss
from di_automata.devinterp.slt.callback import validate_callbacks, SamplerCallback


def call_with(func: Callable, **kwargs):
    """Check the func annotation and call with only the necessary kwargs."""
    sig = inspect.signature(func)
    
    # Filter out the kwargs that are not in the function's signature
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    # Call the function with the filtered kwargs
    return func(**filtered_kwargs)


def sample_single_chain(
    loader: DataLoader,
    criterion: Callable,
    sampling_method: Type[torch.optim.Optimizer],
    main_config: MainConfig,
    checkpoint: dict,
    num_iter: int,
    chain: int = 0,
    optimizer_kwargs: Optional[Dict] = None,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    callbacks: List[SamplerCallback] = [],
):
    """Original code does a deep copy. 
    
    In my code as deep copy has issues with MUP, I am just instantiating a new model and state dict.

    Args:
        ref_model: Currently unused. Instead instantiate a new model class based on config and load state dict.
        use_distil_loss: Whether to use distillation loss. I introduced this as a correction for negative lambda hat estimates.
        This modifies the loss landscape to be a local minimum, allowing RLCT estimations early on in training.
    """
    config: RLCTConfig = main_config.rlct_config
    
    if config.num_burnin_steps:
        warnings.warn('Burn-in is currently not implemented correctly, please set num_burnin_steps to 0.')
    if config.num_draws > len(loader):
        warnings.warn('You are taking more sample batches than there are dataloader batches available, this removes some randomness from sampling but is probably fine. (All sample batches beyond the number dataloader batches are cycled from the start, f.e. 9 samples from [A, B, C] would be [B, A, C, B, A, C, B, A, C].)')
    
    # Note original Timaeus code used deepcopy and passing in of current model
    model, _ = construct_model(main_config)
    model.load_state_dict(checkpoint["state dict"])
    model = model.to(device)
    model.train()

    optimizer = sampling_method(model.parameters(), **(optimizer_kwargs or {}))

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = config.num_draws * config.num_steps_bw_draws + config.num_burnin_steps

    local_draws = pd.DataFrame(
        index=range(config.num_draws),
        columns=["chain", "step", "loss"] + (["model_weights"] if config.return_weights else []),
    )

    iterator = loader if isinstance(loader, IterableDataset) else zip(range(num_steps), itertools.cycle(loader))
    
    if config.use_distill_loss:
        baseline_model, _ = construct_model(main_config)
        baseline_model.load_state_dict(checkpoint["state dict"])
        baseline_model = baseline_model.to(device)
        baseline_model.eval()

    for i, data in enumerate(take_n(iterator, num_iter)): # TODO: ADD TQDM
        
        inputs, labels = data["input_ids"].to(device), data["label_ids"].to(device)
        logits = model(inputs)
        
        def closure(backward=True):
            """
            Compute loss for the current state of the model and update the gradients.
            
            Args:
                backward: Whether to perform backward pass. Only used for updating weight grad at proposed location. See SGLD_MA.step() for more details.
            """
            outputs = model(xs)
            loss = criterion(outputs, logits)
            if backward:
                optimizer.zero_grad() 
                loss.backward()
            return loss
        
        if config.use_distill_loss:
            loss, student_logits = predictive_kl_loss(x=inputs, y=labels, teacher_model=baseline_model, student_model=model)
        else:
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        
        if sampling_method in [SGLD, SGNHT]:
            optimizer.step(closure=None)
        elif sampling_method in [SGLD_MA]:
            optimizer.step(closure=closure)

        if i >= config.num_burnin_steps and (i - config.num_burnin_steps) % config.num_steps_bw_draws == 0:
            draw = (i - config.num_burnin_steps) // config.num_steps_bw_draws  # required for locals()
            loss = loss.item()

            with torch.no_grad():
                for callback in callbacks:
                    call_with(callback, **locals())  # Cursed. This is the way. 

    return local_draws


def _sample_single_chain(kwargs):
    return sample_single_chain(**kwargs)


def sample(
    loader: DataLoader,
    criterion: Callable,
    main_config: MainConfig,
    checkpoint: dict,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    device: torch.device = torch.device("cpu"),
    callbacks: List[SamplerCallback] = [],
):
    """
    Sample model weights using a given optimizer, supporting multiple chains.

    Parameters:
        model (torch.nn.Module): The neural network model.
        sampling_method (torch.optim.Optimizer): Sampling method to use (really a PyTorch optimizer).
        loader (DataLoader): DataLoader for input data.
        criterion (torch.nn.Module): Loss function.
        checkpoint: Model state dict if not using deepcopy to instantiate new model.
        seed (Optional[Union[int, List[int]]]): Random seed(s) for sampling.
        optimizer_kwargs (Optional[Dict[str, Union[float, Literal['adaptive']]]]): Keyword arguments for the optimizer.
        callbacks (List[SamplerCallback]): list of callbacks, each of type SamplerCallback.
        device (Union[torch.device, str]): Device to perform computations on, e.g., 'cpu' or 'cuda'.
    """
    config: RLCTConfig = main_config.rlct_config
    
    if config.cores is None:
        cores = min(4, cpu_count())

    seed = config.seed
    num_chains = config.num_chains
    if seed is not None:
        warnings.warn("You are using seeded runs, for full reproducibility check https://pytorch.org/docs/stable/notes/randomness.html")
        if isinstance(seed, int):
            seeds = [seed + i for i in range(num_chains)]
        elif len(seed) != num_chains:
            raise ValueError("Length of seed list must match number of chains")
        else:
            seeds = seed
    else:
        seeds = [None] * num_chains

    def get_args(i):
        return dict(
            chain=i,
            seed=seeds[i],
            loader=loader,
            criterion=criterion,
            main_config=main_config,
            checkpoint=checkpoint,
            sampling_method=sampling_method,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
        )
    
    if main_config.rlct_config.cores > 1:
        ctx = get_context("spawn")
        with ctx.Pool(cores) as pool:
            results = pool.map(_sample_single_chain, [get_args(i) for i in range(num_chains)])
    else:
        for i in range(num_chains):
            _sample_single_chain(get_args(i))

    for callback in callbacks:
        if hasattr(callback, "finalize"):
            callback.finalize()
