#%%
from architectures import mlp
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import pandas as pd
from collections import Counter
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Any
import plotly.graph_objects as go
import plotly
import os
import plotly.io as pio
from complexity import *


def increment_count(counter: Counter, key: Any) -> Counter:
    counter[key] += 1
    return counter


def heaviside(x) -> float:
    return (x >= 0).float()


def get_parameters(model: nn.Module) -> Tuple[List[float], List[float]]:
    """Extracts and returns the weights and biases of a PyTorch model.

    Parameters:
    model (torch.nn.Module): A PyTorch model from which parameters will be extracted.

    Returns:
    list, list: Two lists containing the flattened weights and biases of the model respectively.
    """
    weights = []
    biases = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight = param.data.flatten().numpy().tolist()
            weights += weight
        elif 'bias' in name:
            biases += param.data.flatten().numpy().tolist()
    return weights, biases


def init_model():
    pass


def all_binary_inputs(n: int, include_zero: bool = True) -> torch.Tensor:
    """Generate all binary inputs of length n.

    Args:
        n: The length of the binary inputs to generate.
        
    Returns:
        binary: A 2D tensor (2^n, n) where each row is a unique binary input.
    """
    # Tensor that represents the binary weights.
    weights = 2**torch.arange(n-1,-1,-1).unsqueeze(0)
    # Tensor that represents all integers from 0 to 2^n - 1.
    integers = torch.arange(2**n).unsqueeze(1)
    # Broadcasting to compute binary representation of each integer.
    binary = (integers & weights) / weights
    return binary if include_zero else binary[1:]


def plot_counts(
    df: pd.DataFrame, 
    title: str, 
    subfolder: str = 'images',
    probabilities: bool = True,
    log: bool = False,
    sample_size: Optional[int] = None) -> None:
    """Plot counts and complexity measures from a DataFrame as plots.

    Args:
        df (pd.DataFrame): The DataFrame.
        title (str): The title of the plot.W
    """
    os.makedirs(subfolder, exist_ok=True)
    fig = go.Figure()
    df_sorted = df.sort_values('Frequency', ascending=False)
    if probabilities:
        assert sample_size is not None
        df['Frequency'] = df['Frequency']/sample_size
    fig.add_trace(go.Bar(
        x=df_sorted['Boolean String'],
        y=df_sorted['Frequency'],
        name='Frequency',
    ))
    fig.update_layout(title=f'{title} Frequencies', yaxis_title='Boolean String', xaxis_title = 'Probability' if probabilities else 'Frequency')
    if log:
        fig.update_layout(yaxis_type='log')
    fig.show()
    plotly.offline.plot(fig, filename=os.path.join('images', subfolder, 'frequencies.html'), auto_open=False)

    # Plot complexity measures against frequency
    complexity_measures = ['Boolean Complexity', 'Entropy', 'Lempel-Ziv Complexity']
    for measure in complexity_measures:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[measure],
            y=df['Frequency'],
            mode='markers',
            name=measure,
            hovertemplate = 
            '<b>Boolean String</b>: %{text}<br><br>' +
            '<b>Frequency</b>: %{y}<br><br>' +
            f'<b>{measure}</b>: %{{x}}',
            text = df['Boolean String']
        ))
        fig.update_layout(title=f'{title} - {measure} vs Frequency', yaxis_title='Probability' if probabilities else 'Frequency', xaxis_title=measure)
        if log:
            fig.update_layout(yaxis_type='log')
        fig.show()
        plotly.offline.plot(fig, filename=os.path.join('images', subfolder, f'{measure}.html'), auto_open=False)


def get_samples() -> Tuple[Counter, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Testing functions produced by random weights")
    parser.add_argument("--n_inputs", type=int, required=True, help="Number of inputs to function produced")
    parser.add_argument("--n_sims", type=int, required=True, help="Number of initialisations to simulate")
    parser.add_argument("--mlp_hidden", type=int, nargs='+', required=True, help="Pass as many arguments as you want, in order of hidden layer sizes")
    parser.add_argument('--log_freq', action='store_true', help='Frequencies should be on a log axis if passed')
    args = parser.parse_args()
    
    inputs = all_binary_inputs(args.n_inputs)
    model = mlp.mlp_constructor(
        input_size=args.n_inputs,
        hidden_sizes=args.mlp_hidden,
        output_size=1,
        bias=False,
        flatten_input=False
    )
    # Dictionary of function truth tables on all possible inputs (key) and frequency (val)
    distribution = Counter()
    for _ in tqdm(range(args.n_sims)):
        output = heaviside(model(inputs))
        f_string = ''
        for bit in output:
            f_string += str(int(bit.item()))
        increment_count(distribution, f_string)

    return distribution, args


def calculate_complexity(counter: Counter, subfolder: str = 'images') -> pd.DataFrame:
    df = pd.DataFrame(list(counter.items()), columns=['Boolean String', 'Frequency'])
    df['Boolean Complexity'] = df['Boolean String'].apply(get_boolean_complexity)
    df['Entropy'] = df['Boolean String'].apply(get_entropy)
    df['Lempel-Ziv Complexity'] = df['Boolean String'].apply(get_lempel_ziv)
    
    os.makedirs(subfolder, exist_ok=True)
    # Save dataframe as a csv file in same directory as images
    csv_file = os.path.join(subfolder, 'complexity_measures.csv')
    df.to_csv(csv_file, index=False)
    
    return df


#%%

if __name__ == "__main__":
    distribution, args = get_samples()
    subfolder = f'plots_n_inputs_{args.n_inputs}_n_sims_{args.n_sims}_hidden_{"-".join(map(str, args.mlp_hidden))}'
    os.makedirs(f'images/{subfolder}', exist_ok=True)
    df = calculate_complexity(distribution)
    df.head()
    plot_counts(df, title = f'Input Size: {args.n_inputs}, Samples: {args.n_sims}, MLP Hidden Sizes: {args.mlp_hidden}', subfolder=subfolder)


