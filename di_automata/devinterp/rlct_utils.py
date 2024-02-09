import pandas as pd
import numpy as np
from plotnine import *

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


def plot_loss_trace(loss: np.ndarray):
    """Loss trace used to examine chain health."""
    df = pd.DataFrame(loss).reset_index().melt(id_vars="index", var_name="timestep", value_name="loss")
    df['timestep'] = df['timestep'].astype(int)
    
    p = (
        ggplot(df, aes(x='timestep', y='loss', color='factor(index)')) +
        geom_line()
    )
    return p