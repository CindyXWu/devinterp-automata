import os
import sys
from pathlib import Path
from dotenv.main import load_dotenv
import plotly.express as px
from typing import List, Union, Optional, Dict, Tuple
from jaxtyping import Int, Float
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import einops
from IPython.display import display

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# Plotting
import circuitsvis as cv
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from transformer_lens.utils import to_numpy
from transformer_lens.components import LayerNorm

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAIN = __name__ == "__main__"

import wandb
from pathlib import Path
import yaml
import s3fs
from omegaconf import OmegaConf

from di_automata.config_setup import *
from di_automata.constructors import (
    construct_model,
    create_dataloader_hf,
)
from di_automata.tasks.data_utils import take_n
import plotly.io as pio

# AWS
load_dotenv()
AWS_KEY, AWS_SECRET = os.getenv("AWS_KEY"), os.getenv("AWS_SECRET")
s3 = s3fs.S3FileSystem(key=AWS_KEY, secret=AWS_SECRET)


from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

CHAPTER = r"chapter1_transformers"
CHAPTER_DIR = r"./" if CHAPTER in os.listdir() else os.getcwd().split(CHAPTER)[0]
EXERCISES_DIR = CHAPTER_DIR + f"{CHAPTER}/exercises"
sys.path.append(EXERCISES_DIR)


# Plotting =====================================

def imshow_attention(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    """Used for visualising individual heads without circuitsvis."""
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)


def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)


# Used for function below
update_layout_set = {"xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor", "showlegend", "xaxis_tickmode", "yaxis_tickmode", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap", "coloraxis_showscale", "xaxis_tickangle"}

def imshow(tensor: torch.Tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    return_fig = kwargs_pre.pop("return_fig", False)
    text = kwargs_pre.pop("text", None)
    xaxis_tickangle = kwargs_post.pop("xaxis_tickangle", None)
    static = kwargs_pre.pop("static", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "color_continuous_midpoint" not in kwargs_pre:
        kwargs_pre["color_continuous_midpoint"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        # Weird thing where facet col wrap means labels are in wrong order
        if "facet_col_wrap" in kwargs_pre:
            facet_labels = reorder_list_in_plotly_way(facet_labels, kwargs_pre["facet_col_wrap"])
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    if text:
        if tensor.ndim == 2:
            # if 2D, then we assume text is a list of lists of strings
            assert isinstance(text[0], list)
            assert isinstance(text[0][0], str)
            text = [text]
        else:
            # if 3D, then text is either repeated for each facet, or different
            assert isinstance(text[0], list)
            if isinstance(text[0][0], str):
                text = [text for _ in range(len(fig.data))]
        for i, _text in enumerate(text):
            fig.data[i].update(
                text=_text,
                texttemplate="%{text}",
                textfont={"size": 12}
            )
    # Very hacky way of fixing the fact that updating layout with new tickangle only applies to first facet by default
    if xaxis_tickangle is not None:
        n_facets = 1 if tensor.ndim == 2 else tensor.shape[0]
        for i in range(1, 1+n_facets):
            xaxis_name = "xaxis" if i == 1 else f"xaxis{i}"
            fig.layout[xaxis_name]["tickangle"] = xaxis_tickangle
    return fig if return_fig else fig.show(renderer=renderer, config={"staticPlot": static})


def reorder_list_in_plotly_way(L: list, col_wrap: int):
    '''
    Helper function, because Plotly orders figures in an annoying way when there's column wrap.
    '''
    L_new = []
    while len(L) > 0:
        L_new.extend(L[-col_wrap:])
        L = L[:-col_wrap]
    print(f"Reordered labels: {L_new}")
    return L_new


def plot_tensor_heatmap(stacked_tensor, title='Comparison of Tensor Positions', figsize=(10, 6), cmap='coolwarm'):
    """
    Plots a heatmap of a stacked tensor.

    Args:
        stacked_tensor (Tensor): A stacked PyTorch tensor to plot.
        title (str, optional): Title of the heatmap. Defaults to 'Comparison of Tensor Positions'.
        figsize (tuple, optional): Figure size. Defaults to (10, 6).
        cmap (str, optional): Colormap for the heatmap. Defaults to 'coolwarm'.
        xticklabels (bool or list, optional): X-axis tick labels. Defaults to True (automatic).
        yticklabels (list, optional): Y-axis tick labels. If None, defaults to index numbers.

    Returns:
        None. Displays a matplotlib plot.
    """
    
    # Ensure the stacked_tensor is moved to CPU and converted to a numpy array
    tensor_matrix = stacked_tensor.cpu().numpy()

    # Plotting
    plt.figure(figsize=figsize)
    sns.heatmap(tensor_matrix, annot=True, cmap=cmap, fmt='d', linewidths=.5)
    plt.title('Comparison of Tensor Positions')
    plt.ylabel('Tensor')
    plt.xlabel('Position')
    
    plt.xticks(np.arange(0.5, stacked_tensor.shape[1], 1), np.arange(1, 26))
    plt.yticks(np.arange(0.5, len(stacked_tensor), 1), ['all_1_label', 'predicted_all_1s', 'all_0_label', 'predicted_all_0s', 'all_0s_except_1_label', 'predicted_all_0s_except_1', 'all_1s_except_1_label', 'predicted_all_1s_except_1'], rotation=0)
    plt.show()


# PCA =======================

def get_pca(tensor):
    mean = tensor.mean(dim=0)
    nonzero = torch.nonzero(mean).squeeze()
    mean = mean[nonzero]
    tensor = tensor[:, nonzero]
    std = tensor.std(dim=0)
    centered_tensor = (tensor - mean) / std
    U, S, V = torch.svd(centered_tensor)
    return U, S, V


def get_vars(singular_vals, truncate_idx=None):
    """Explained variances of PCA components."""
    total_var = (singular_vals ** 2).sum()
    return torch.cumsum(singular_vals ** 2 / total_var, dim=0)



# From 'Balanced Bracket Classifier' ================================================

def get_activations(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    names: Union[str, List[str]]
) -> Union[torch.Tensor, ActivationCache]:
    '''
    Uses hooks to return activations from the model.

    If names is a string, returns the activations for that hook name.
    If names is a list of strings, returns the cache containing only those activations.
    '''
    names_list = [names] if isinstance(names, str) else names
    _, cache = model.run_with_cache(
        toks,
        return_type=None,
        names_filter=lambda name: name in names_list,
    )

    return cache[names] if isinstance(names, str) else cache


def LN_hook_names(layernorm: LayerNorm) -> Tuple[str, str]:
    '''
    Returns the names of the hooks immediately before and after a given layernorm.
    e.g. LN_hook_names(model.final_ln) returns ["blocks.2.hook_resid_post", "ln_final.hook_normalized"]
    '''
    if layernorm.name == "ln_final":
        input_hook_name = utils.get_act_name("resid_post", 2)
        output_hook_name = "ln_final.hook_normalized"
    else:
        layer, ln = layernorm.name.split(".")[1:]
        input_hook_name = utils.get_act_name("resid_pre" if ln=="ln1" else "resid_mid", layer)
        output_hook_name = utils.get_act_name('normalized', layer, ln)

    return input_hook_name, output_hook_name


def get_ln_fit(
    model: HookedTransformer, data: torch.Tensor, layernorm: LayerNorm, seq_pos: Optional[int] = None
) -> Tuple[LinearRegression, float]:
    '''
    if seq_pos is None, find best fit aggregated over all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and the r^2 of the fit
    '''
    input_hook_name, output_hook_name = LN_hook_names(layernorm)
    # SOLUTION

    activations_dict = get_activations(model, data.toks, [input_hook_name, output_hook_name])
    inputs = utils.to_numpy(activations_dict[input_hook_name])
    outputs = utils.to_numpy(activations_dict[output_hook_name])

    if seq_pos is None:
        inputs = einops.rearrange(inputs, "batch seq d_model -> (batch seq) d_model")
        outputs = einops.rearrange(outputs, "batch seq d_model -> (batch seq) d_model")
    else:
        inputs = inputs[:, seq_pos, :]
        outputs = outputs[:, seq_pos, :]

    final_ln_fit = LinearRegression().fit(inputs, outputs)

    r2 = final_ln_fit.score(inputs, outputs)

    return (final_ln_fit, r2)


def cos_sim_with_MLP_weights(model: HookedTransformer, v: Float[Tensor, "d_model"], layer: int) -> Float[Tensor, "d_mlp"]:
    '''
    Returns a vector of length d_mlp, where the ith element is the cosine similarity between v and the
    ith in-direction of the MLP in layer `layer`.

    Recall that the in-direction of the MLPs are the columns of the W_in matrix.
    '''
    # SOLUTION
    v_unit = v / v.norm()
    W_in_unit = model.W_in[layer] / model.W_in[layer].norm(dim=0)

    return einops.einsum(v_unit, W_in_unit, "d_model, d_model d_mlp -> d_mlp")


def avg_squared_cos_sim(v: Float[Tensor, "d_model"], n_samples: int = 1000) -> float:
    '''
    Returns the average (over n_samples) cosine similarity between v and another randomly chosen vector.

    We can create random vectors from the standard N(0, I) distribution.
    '''
    # SOLUTION
    v2 = t.randn(n_samples, v.shape[0]).to(device)
    v2 /= v2.norm(dim=1, keepdim=True)

    v1 = v / v.norm()

    return (v1 * v2).pow(2).sum(1).mean().item()


# For activation patching ===================================

def hook_fn_patch_qk(
    value: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    new_value: Float[Tensor, "... seq d_head"],
    head_idx: Optional[int] = None
) -> None:
    if head_idx is not None:
        value[..., head_idx, :] = new_value[..., head_idx, :]
    else:
        value[...] = new_value[...]


def hook_fn_display_attn_patterns(
    pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    head_idx: int = 0
) -> None:
    avg_head_attn_pattern = pattern.mean(0)
    labels = ["[start]", *[f"{i+1}" for i in range(40)], "[end]"]
    display(cv.attention.attention_heads(
        tokens=labels,
        attention=avg_head_attn_pattern,
        attention_head_names=["0.0", "0.1"],
        max_value=avg_head_attn_pattern.max(),
        mask_upper_tri=False, # use for bidirectional models
    ))


def get_q_and_k_for_given_input(
    model: HookedTransformer,
    input: torch.Tensor,
    layer: int,
) -> Tuple[Float[Tensor, "seq n_head d_model"], Float[Tensor,  "seq n_head d_model"]]:
    '''
    Returns the queries and keys (both of shape [seq, d_head, d_model]) for the given parens string,
    for all attention heads in the given layer.
    '''
    # SOLUTION
    q_name = utils.get_act_name("q", layer)
    k_name = utils.get_act_name("k", layer)

    activations = get_activations(
        model,
        input,
        [q_name, k_name]
    )

    return activations[q_name][0], activations[k_name][0]