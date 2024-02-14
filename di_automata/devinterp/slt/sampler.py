from typing import Callable, Dict, List, Literal, Optional, Type, Union, OrderedDict
import itertools
import inspect
import warnings
from tqdm import tqdm
import math

import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.multiprocessing import cpu_count, get_context

from di_automata.config_setup import MainConfig
from di_automata.devinterp.optim.sgld import SGLD
from di_automata.devinterp.slt.callback import SamplerCallback
from di_automata.devinterp.rlct_utils import create_callbacks

from di_automata.tasks.data_utils import take_n
from di_automata.config_setup import MainConfig, RLCTConfig
from di_automata.devinterp.optim.sgld import SGLD
from di_automata.devinterp.optim.sgld_ma import SGLD_MA
from di_automata.constructors import construct_model
from di_automata.losses import predictive_kl_loss
from di_automata.devinterp.slt.callback import SamplerCallback


def estimate_learning_coeff_with_summary(
    loader: DataLoader,
    criterion: Callable,
    main_config: MainConfig,
    checkpoint: OrderedDict,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    device: torch.device = torch.device("cpu"),
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
) -> tuple[dict, list[str]]:
    config = main_config.rlct_config

    if optimizer_kwargs is None: # If not specified at run-time (e.g. where you want multiple types run at once) then use config
        match config.sampling_method:
            case "SGLD" | "SGLD_MA": optimizer_kwargs = config.sgld_kwargs
            case "SGNHT": optimizer_kwargs = config.sgnht_kwargs
    
    callbacks, callback_names = create_callbacks(main_config, device)
    
    sample(
        loader=loader,
        criterion=criterion,
        main_config=main_config,
        checkpoint=checkpoint,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
        callbacks=callbacks,
    )

    results = {}
    for callback in callbacks:
        if hasattr(callback, "sample"):
            results.update(callback.sample())

    return results, callback_names


def call_with(func: Callable, **kwargs):
    """Check the func annotation and call with only the necetask_config.lengthask_config.lengthsary kwargs."""
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
    checkpoint: OrderedDict,
    chain: int = 0,
    seed = None,
    optimizer_kwargs: Optional[Dict] = None,
    device: torch.device = torch.device("cpu"),
    callbacks: List[SamplerCallback] = [],
):
    """Original code does a deep copy. 
    
    In my code as deep copy has issues with MUP, I am just instantiating a new model and state dict.

    Args:
        loader: Iterable infinite dataloader. Original devinterp code wraps it as an iterable.
        criterion: KL or distillation loss.
        sampling_method: SGLD, SGNHT, or SGLD_MA class. Passed in separately to allow for multiple sampling types to be called in the same run.
        checkpoint: Model state dict. Note original Timaeus code used deepcopy and passing in of current model, but deepcopy is not great for entire models.
        Optimizer_kwargs: Also passed in separately to config to allow for multiple sampling types to be called in the same run.
        use_distill_loss: Whether to use distillation loss. I introduced this as a correction for negative lambda hat estimates.
        This modifies the loss landscape to be a local minimum, allowing RLCT estimations early on in training.
        callbacks: Important for storing and returning sample results.
    """
    config: RLCTConfig = main_config.rlct_config
    
    if seed is not None:
        torch.manual_seed(seed)
    
    if config.num_draws > len(loader):
        warnings.warn('You are taking more sample batches than there are dataloader batches available, this removes some randomness from sampling but is probably fine. (All sample batches beyond the number dataloader batches are cycled from the start, f.e. 9 samples from [A, B, C] would be [B, A, C, B, A, C, B, A, C].)')
    
    model, _ = construct_model(main_config)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.train()

    if config.use_distill_loss:
        print("Using distill loss")
        baseline_model, _ = construct_model(main_config)
        baseline_model.load_state_dict(checkpoint)
        baseline_model = baseline_model.to(device)
        baseline_model.eval()
        
    optimizer = sampling_method(model.parameters(), **(optimizer_kwargs or {}))

    num_steps = config.num_draws * config.num_steps_bw_draws + config.num_burnin_steps
    iterator = loader if isinstance(loader.dataset, IterableDataset) else itertools.cycle(loader)
    progress_bar = tqdm(total=num_steps, disable=not config.verbose)

    for (i, data) in enumerate(take_n(iterator, num_steps)):
        inputs, labels = data["input_ids"].to(device), data["label_ids"].to(device)
        logits = model(inputs)
        
        def closure(backward=False):
            """
            Compute loss for current state of model and update gradients.
            
            Args:
                backward: Whether to perform backward pass. Only used for updating weight grad at proposed location. 
                See SGLD_MA.step() for more details.
            """
            logits = model(inputs)
            loss, student_logits = predictive_kl_loss(x=inputs, y=labels, teacher_model=baseline_model, student_model=model) if config.use_distill_loss else criterion(logits, labels)
            if backward:
                optimizer.zero_grad() 
                loss.backward()
            return loss
        
        loss, student_logits = predictive_kl_loss(x=inputs, teacher_model=baseline_model, student_model=model) if config.use_distill_loss else criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        
        if sampling_method in [SGLD]:
            optimizer.step(closure=None)
        elif sampling_method in [SGLD_MA]:
            optimizer.step(closure=closure)
            acceptance_ratio = optimizer.acceptance_ratio # required for locals()

        if i >= config.num_burnin_steps and (i - config.num_burnin_steps) % config.num_steps_bw_draws == 0:
            draw = (i - config.num_burnin_steps) // config.num_steps_bw_draws  # required for locals()
            loss = loss.item()
            
            assert not math.isnan(loss), "LLC mean is NaN. This is likely due to a bug in the optimizer - put breakpoints there."

            with torch.no_grad():
                for callback in callbacks:
                    call_with(callback, **locals())  # Cursed. This is the way. 
        
        progress_bar.update()
        progress_bar.set_description(f"Chain {chain}, sampler {config.sampling_method}, accept ratio {acceptance_ratio}")


def _sample_single_chain(kwargs):
    return sample_single_chain(**kwargs)


def sample(
    loader: DataLoader,
    criterion: Callable,
    main_config: MainConfig,
    checkpoint: OrderedDict,
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
            callbacks=callbacks,
        )
    
    if main_config.rlct_config.cores > 1:
        ctx = get_context("spawn")
        with ctx.Pool(cores) as pool:
            pool.map(_sample_single_chain, [get_args(i) for i in range(num_chains)])
    else:
        for i in range(num_chains):
            _sample_single_chain(get_args(i))

    for callback in callbacks:
        if hasattr(callback, "finalize"):
            callback.finalize()
