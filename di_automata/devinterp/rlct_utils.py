from typing import Callable
import pandas as pd
import numpy as np
from plotnine import *
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import wandb
import os
from plotly.subplots import make_subplots


from di_automata.config_setup import MainConfig
from di_automata.devinterp.slt.wbic import OnlineWBICEstimator
from di_automata.devinterp.slt.llc import LLCEstimator, OnlineLLCEstimator
from di_automata.devinterp.slt.norms import WeightNorm, GradientNorm, NoiseNorm
from di_automata.devinterp.slt.gradient import GradientDistribution
from di_automata.devinterp.slt.trace import OnlineTraceStatistics
from di_automata.devinterp.slt.loss import OnlineLossStatistics
from di_automata.devinterp.slt.callback import validate_callbacks


def plot_pca_ggplot(
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


def plot_pca_plotly(
    component_0: list[float], 
    component_1: list[float], 
    component_2: list[float], 
    config: MainConfig,
):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Component 0 vs. 1", "Component 0 vs. 2", "Component 1 vs. 2"))

    color_scale = 'sunset'
    checkpoint_idx = list(range(len(component_0)))
    
    fig.add_trace(go.Scatter(x=component_0, y=component_1, mode='markers', 
                             marker=dict(color=checkpoint_idx, colorscale=color_scale, colorbar=dict(title="Checkpoint Index"), showscale=True),
                             showlegend=False), 
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=component_0, y=component_2, mode='markers', 
                             marker=dict(color=checkpoint_idx, colorscale=color_scale, showscale=False),
                             showlegend=False), 
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=component_1, y=component_2, mode='markers', 
                             marker=dict(color=checkpoint_idx, colorscale=color_scale, showscale=False),
                             showlegend=False), 
                  row=1, col=3)

    fig.update_xaxes(title_text="Component 1", row=1, col=1)
    fig.update_yaxes(title_text="Component 2", row=1, col=1)
    
    fig.update_xaxes(title_text="Component 1", row=1, col=2)
    fig.update_yaxes(title_text="Component 3", row=1, col=2)

    fig.update_xaxes(title_text="Component 2", row=1, col=3)
    fig.update_yaxes(title_text="Component 3", row=1, col=3)

    fig.update_layout(
        title_text=f"Essential Dynamics PCA {config.task_config.dataset_type} seqlen {config.task_config.length} its {config.num_training_iter} cpfreq {config.rlct_config.ed_config.eval_frequency}", 
        height=500, width=1500,
    )

    fig.write_image("PCA.png")
    
    
def plot_explained_var(explained_var: np.ndarray):
    plt.figure(figsize=(10, 7))
    plt.bar(range(1, 4), explained_var, alpha=0.5, align='center', label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.xticks([1, 2, 3])
    plt.title('Explained variance by top 3 PCA components')
    plt.legend(loc='best')
    plt.savefig("pca_explained_var.png", dpi=300)
    

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


def extract_and_save_rlct_data(
    data: dict, 
    callback_names: list[str], 
    sampler_type: str, 
    idx: int,
    **kwargs,
) -> dict:
    """Used to format data to log to WandB and save locally as csv.
    
    Needs a WandB instance to be initialised at time of function call.
    """
    log_dict = {}
    return_dict = {}
    # Always log mean and std
    log_dict[f"{sampler_type}/mean"] = return_dict[f"{sampler_type}/mean"] = data["llc/mean"]
    log_dict[f"{sampler_type}/std"] = return_dict[f"{sampler_type}/std"] = data["llc/std"]
    if kwargs["loss"]:
        log_dict["Train Acc"] = kwargs["loss"]
    if kwargs["acc"]:
        log_dict["Train Loss"] = kwargs["acc"]
    if "accept_ratio/mean" in data.keys():
        log_dict[f"{sampler_type}/accept_ratio"] = data["accept_ratio/mean"]
        return_dict[f"{sampler_type}/accept_ratio"] = data["accept_ratio/mean"]
    
    loss_p = plot_trace(data["loss/trace"], "loss")
    loss_filename = f"{sampler_type}_loss.png"
    loss_p.save(loss_filename, width=10, height=4, dpi=300)
    log_dict[f"{sampler_type}/loss"] = wandb.Image(loss_filename)
    
    weight_norm_filename = "weight_norm.png"
    noise_norm_filename = "noise_norm.png"
    gradient_norm_filename = "gradient_norm.png"
    
    if 'OnlineWBICEstimator' in callback_names:
        log_dict[f"{sampler_type}/wbic/means"] = data["wbic/means"]
        log_dict[f"{sampler_type}/wbic/stds"] = data["wbic/stds"]
        return_dict[f"{sampler_type}/wbic/means/mean"] = data["wbic/means"].mean().item()
        return_dict[f"{sampler_type}/wbic/std/mean"] = data["wbic/stds"].mean().item()

    if 'WeightNorm' in callback_names:
        weight_norm_p = plot_trace(data["weight_norm/trace"], "weight norm")
        return_dict[f"{sampler_type}/weight_norm/mean"] = data["weight_norm/trace"].mean().item()
        weight_norm_p.save(weight_norm_filename, width=10, height=4, dpi=300)
        log_dict["weight_norm"] = wandb.Image(weight_norm_filename)
    
    if 'NoiseNorm' in callback_names:
        noise_norm_p = plot_trace(data["noise_norm/trace"], "noise norm")
        return_dict[f"{sampler_type}/noise_norm/mean"] = data["noise_norm/trace"].mean().item()
        noise_norm_p.save(noise_norm_filename, width=10, height=4, dpi=300)
        log_dict["noise_norm"] = wandb.Image(noise_norm_filename)
    
    if 'GradientNorm' in callback_names:
        gradient_norm_p = plot_trace(data["gradient_norm/trace"], "gradient norm")
        return_dict[f"{sampler_type}/gradient_norm/mean"] = data["gradient_norm/trace"].mean().item()
        gradient_norm_p.save(gradient_norm_filename, width=10, height=4, dpi=300)
        log_dict["gradient_norm"] = wandb.Image(gradient_norm_filename)
    
    # TODO: additional metrics to be logged (e.g., GradientDistribution)
    
    wandb.log(log_dict, step=idx)
    
    for filename in [weight_norm_filename, gradient_norm_filename, noise_norm_filename]:
        if os.path.exists(filename):
            os.remove(filename)
    
    return return_dict


def extract_and_plot_rlct_data(
    data: dict, 
    callback_names: list[str], 
    sampler_type: str, 
    idx: int
) -> dict:
    """Used to format data to log to WandB and save locally as csv.
    
    Needs a WandB instance to be initialised at time of function call.
    """
    log_dict = {}
    return_dict = {}
    # Always log mean and std
    log_dict[f"{sampler_type}/mean_post"] = return_dict[f"{sampler_type}/mean"] = data["llc/mean"]
    log_dict[f"{sampler_type}/std_post"] = return_dict[f"{sampler_type}/std"] = data["llc/std"]
    if "accept_ratio/mean" in data.keys():
        log_dict[f"{sampler_type}/accept_ratio_post"] = data["accept_ratio/mean"]
        return_dict[f"{sampler_type}/accept_ratio_post"] = data["accept_ratio/mean"]
    
    loss_p = plot_trace(data["loss/trace"], "loss")
    loss_filename = f"{sampler_type}_loss.png"
    loss_p.save(loss_filename, width=10, height=4, dpi=300)
    log_dict[f"{sampler_type}/loss"] = wandb.Image(loss_filename)
    
    # weight_norm_filename = "weight_norm.png"
    # noise_norm_filename = "noise_norm.png"
    # gradient_norm_filename = "gradient_norm.png"
    
    # if 'OnlineWBICEstimator' in callback_names:
    #     log_dict[f"{sampler_type}/wbic/means"] = data["wbic/means"]
    #     log_dict[f"{sampler_type}/wbic/stds"] = data["wbic/stds"]
    #     return_dict[f"{sampler_type}/wbic/means/mean"] = data["wbic/means"].mean().item()
    #     return_dict[f"{sampler_type}/wbic/std/mean"] = data["wbic/stds"].mean().item()

    # if 'WeightNorm' in callback_names:
    #     weight_norm_p = plot_trace(data["weight_norm/trace"], "weight norm")
    #     return_dict[f"{sampler_type}/weight_norm/mean"] = data["weight_norm/trace"].mean().item()
    #     weight_norm_p.save(weight_norm_filename, width=10, height=4, dpi=300)
    #     log_dict["weight_norm"] = wandb.Image(weight_norm_filename)
    
    # if 'NoiseNorm' in callback_names:
    #     noise_norm_p = plot_trace(data["noise_norm/trace"], "noise norm")
    #     return_dict[f"{sampler_type}/noise_norm/mean"] = data["noise_norm/trace"].mean().item()
    #     noise_norm_p.save(noise_norm_filename, width=10, height=4, dpi=300)
    #     log_dict["noise_norm"] = wandb.Image(noise_norm_filename)
    
    # if 'GradientNorm' in callback_names:
    #     gradient_norm_p = plot_trace(data["gradient_norm/trace"], "gradient norm")
    #     return_dict[f"{sampler_type}/gradient_norm/mean"] = data["gradient_norm/trace"].mean().item()
    #     gradient_norm_p.save(gradient_norm_filename, width=10, height=4, dpi=300)
    #     log_dict["gradient_norm"] = wandb.Image(gradient_norm_filename)
    
    # TODO: additional metrics to be logged (e.g., GradientDistribution)
    
    wandb.log(log_dict, step=idx)
    
    for filename in [weight_norm_filename, gradient_norm_filename, noise_norm_filename]:
        if os.path.exists(filename):
            os.remove(filename)
    
    return return_dict


if __name__ == "__main__":
    """Test plotting functions."""
    np.random.seed(42)
    component_0 = np.random.randn(100)
    component_1 = np.random.randn(100)
    component_2 = np.random.randn(100)
    
    plot_pca_plotly(component_0, component_1, component_2, "test.png")