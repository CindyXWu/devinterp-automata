from typing import Callable
import pandas as pd
import numpy as np
from plotnine import *

from di_automata.config_setup import MainConfig
from di_automata.devinterp.slt.wbic import OnlineWBICEstimator
from di_automata.devinterp.slt.llc import LLCEstimator, OnlineLLCEstimator
from di_automata.devinterp.slt.norms import WeightNorm, GradientNorm, NoiseNorm
from di_automata.devinterp.slt.gradient import GradientDistribution
from di_automata.devinterp.slt.trace import OnlineTraceStatistics
from di_automata.devinterp.slt.loss import OnlineLossStatistics
from di_automata.devinterp.slt.callback import validate_callbacks

def plot_components(
    component_0: list[float], 
    component_1: list[float], 
    component_2: list[float], 
) -> ggplot:
    seq_id = list(range(len(component_0))) 
    df_0_vs_1 = pd.DataFrame({'x': component_0, 'y': component_1, 'Comparison': 'Component 0 vs. Component 1', 'SeqID': seq_id})
    df_0_vs_2 = pd.DataFrame({'x': component_0, 'y': component_2, 'Comparison': 'Component 0 vs. Component 2', 'SeqID': seq_id})
    df_1_vs_2 = pd.DataFrame({'x': component_1, 'y': component_2, 'Comparison': 'Component 1 vs. Component 2', 'SeqID': seq_id})
    df = pd.concat([df_0_vs_1, df_0_vs_2, df_1_vs_2])

    p = (
        ggplot(df, aes('x', 'y', color='SeqID')) +
        geom_point() +
        scale_color_gradient(low='blue', high='red') +
        facet_wrap('~Comparison', scales='free') + 
        labs(title='Component Comparisons', x='Component Value', y='Component Value') +
        coord_fixed(ratio=1)
    )
    
    return p


def plot_trace(trace: np.ndarray, name: str):
    """Plot a diagnostic trace used to examine chain health."""
    df = pd.DataFrame(trace).reset_index().melt(id_vars="index", var_name="timestep", value_name=name)
    df['timestep'] = df['timestep'].astype(int)
    
    p = (
        ggplot(df, aes(x='timestep', y=name, color='factor(index)')) +
        geom_line()
    )
    return p


def create_callbacks(config: MainConfig, device: str) -> tuple[list[Callable], list[str]]:
    rlct_config = config.rlct_config
    num_chains, num_draws, num_samples = rlct_config.num_chains, rlct_config.num_draws, rlct_config.num_samples
    
    llc_estimator = OnlineLLCEstimator(num_chains, num_draws, num_samples, device) if rlct_config.online else LLCEstimator(num_chains, num_draws, num_samples, device)

    callbacks = [
            OnlineWBICEstimator(num_chains, num_draws, num_samples, device),
            WeightNorm(num_chains, num_draws, device, p_norm=2),
            NoiseNorm(num_chains, num_draws, device, p_norm=2),
            GradientNorm(num_chains, num_draws, device, p_norm=2),
            # GradientDistribution(num_chains, num_draws, device=device),
    ] if rlct_config.use_diagnostics else []
    callbacks = [llc_estimator, *callbacks]
    validate_callbacks(callbacks)
    
    callback_names = []
    for callback in callbacks:
        callback_names.append(type(callback).__name__)
    
    return callbacks, callback_names