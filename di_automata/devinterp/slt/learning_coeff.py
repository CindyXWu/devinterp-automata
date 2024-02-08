from typing import Callable, Dict, Literal, Optional, Type, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from di_automata.config_setup import MainConfig
from di_automata.devinterp.optim.sgld import SGLD
from di_automata.devinterp.slt.callback import SamplerCallback
from di_automata.devinterp.slt.sampler import sample


class LLCEstimator(SamplerCallback):
    """
    Callback for estimating the Local Learning Coefficient (LLC) in a rolling fashion during a sampling process. 
    It calculates the LLC based on the average loss across draws for each chain:
    $$
    TODO
    $$
    
    Attributes:
        num_chains (int): Number of chains to run. (should be identical to param passed to sample())
        num_draws (int): Number of samples to draw. (should be identical to param passed to sample())
        n (int): Number of samples used to calculate the LLC.
        device (Union[torch.device, str]): Device to perform computations on, e.g., 'cpu' or 'cuda'.
    """
    def __init__(self, num_chains: int, num_draws: int, n: int, device: Union[torch.device, str]="cpu"):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)

        self.n = torch.tensor(n, dtype=torch.float32).to(device)
        self.llc_per_chain = torch.zeros(num_chains, dtype=torch.float32).to(device)
        self.llc_mean = torch.tensor(0., dtype=torch.float32).to(device)
        self.llc_std = torch.tensor(0., dtype=torch.float32).to(device)

        self.device = device

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss 

    @property
    def init_loss(self):
        return self.losses[0, 0]

    def finalize(self):
        avg_losses = self.losses.mean(axis=1)
        self.llc_per_chain = (self.n / self.n.log()) * (avg_losses - self.init_loss)
        self.llc_mean = self.llc_per_chain.mean()
        self.llc_std = self.llc_per_chain.std()
        
    def sample(self):
        return {
            "llc/mean": self.llc_mean.cpu().numpy().item(),
            "llc/std": self.llc_std.cpu().numpy().item(),
            **{f"llc-chain/{i}": self.llc_per_chain[i].cpu().numpy().item() for i in range(self.num_chains)},
            "loss/trace": self.losses.cpu().numpy(),
        }
    
    def __call__(self, chain: int, draw: int, loss: float):
        self.update(chain, draw, loss)
        

class OnlineLLCEstimator(SamplerCallback):
    """
    Callback for estimating the Local Learning Coefficient (LLC) in an online fashion during a sampling process. 
    It calculates LLCs using the same formula as LLCEstimator, but continuously and including means and std across draws (as opposed to just across chains).

    Attributes:
        num_chains (int): Number of chains to run. (should be identical to param passed to sample())
        num_draws (int): Number of samples to draw. (should be identical to param passed to sample())
        n (int): Number of samples used to calculate the LLC.
        device (Union[torch.device, str]): Device to perform computations on, e.g., 'cpu' or 'cuda'.
    """
    def __init__(self, num_chains: int, num_draws: int, n: int, device="cpu"):
        self.num_chains = num_chains
        self.num_draws = num_draws

        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.llcs = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)

        self.n = torch.tensor(n, dtype=torch.float32).to(device)

        self.llc_means = torch.tensor(num_draws, dtype=torch.float32).to(device)
        self.llc_stds = torch.tensor(num_draws, dtype=torch.float32).to(device)

        self.device = device

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss 
        init_loss = self.losses[chain, 0]

        if draw == 0:  # TODO: We can probably drop this and it still works (but harder to read)
            self.llcs[chain, draw] = 0.
        else:
            t = draw + 1
            prev_llc = self.llcs[chain, draw - 1]
            # print(chain, draw, prev_llc, self.n, loss, init_loss, loss - init_loss)

            with torch.no_grad():
                self.llcs[chain, draw] = (1 / t) * (
                    (t - 1) * prev_llc + (self.n / self.n.log()) * (loss - init_loss)
                )



def estimate_learning_coeff(
    loader: DataLoader,
    criterion: Callable,
    main_config: MainConfig,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    checkpoint: Optional[dict] = None,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    device: torch.device = torch.device("cpu"),
) -> float:
    trace = sample(
        loader=loader,
        criterion=criterion,
        main_config=main_config,
        checkpoint=checkpoint,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
    )
    baseline_loss = trace.loc[trace["chain"] == 0, "loss"].iloc[0]
    avg_loss = trace.groupby("chain")["loss"].mean().mean()
    num_samples = optimizer_kwargs["num_samples"]

    return (avg_loss - baseline_loss) * num_samples / np.log(num_samples)


def estimate_learning_coeff_with_summary(
    loader: DataLoader,
    criterion: Callable,
    main_config: MainConfig,
    checkpoint: Optional[dict] = None,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    trace = sample(
        loader=loader,
        criterion=criterion,
        main_config=main_config,
        checkpoint=checkpoint,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
    )

    num_chains = main_config.rlct_config.num_chains
    
    baseline_loss = trace.loc[trace["chain"] == 0, "loss"].iloc[0]
    num_samples = optimizer_kwargs["num_samples"]
    avg_losses = trace.groupby("chain")["loss"].mean()
    results = torch.zeros(num_chains, device=device)

    for i in range(num_chains):
        chain_avg_loss = avg_losses.iloc[i]
        results[i] = (chain_avg_loss - baseline_loss) * num_samples / np.log(num_samples)

    avg_loss = results.mean()
    std_loss = results.std()

    return {
        "mean": avg_loss.item(),
        "std": std_loss.item(),
        **{f"chain_{i}": results[i].item() for i in range(num_chains)},
        "trace": trace,
    }


def plot_learning_coeff_trace(trace: pd.DataFrame, **kwargs):
    import matplotlib.pyplot as plt

    for chain, df in trace.groupby("chain"):
        plt.plot(df["step"], df["loss"], label=f"Chain {chain}", **kwargs)

    plt.xlabel("Step")
    plt.ylabel(r"$L_n(w)$")
    plt.title("Learning Coefficient Trace")
    plt.legend()
    plt.show()
