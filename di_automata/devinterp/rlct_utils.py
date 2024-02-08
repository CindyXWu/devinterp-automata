import pandas as pd
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
        geom_smooth(method='lm', se=False, color='black') + # Add line of best fit, without confidence interval
        scale_color_gradient(low='blue', high='red') +
        facet_wrap('~Comparison', scales='free') + 
        labs(title='Component Comparisons', x='Component Value', y='Component Value') +
        coord_fixed(ratio=1)
    )
    
    return p