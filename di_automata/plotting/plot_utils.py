import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from plotnine import ggplot
from torch import Tensor
from typing import Optional
from pathlib import Path

matplotlib.use('Agg') 


def visualise_seq_data(data: Tensor, idx: int, n: Optional[int] = 5):
    """
    Args:
        data: Shape [batch, sequence_length].
        n: Number of batch items to visualise.
    """
    Path('images').mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(data[:n, :].cpu().numpy(), cmap='coolwarm', linewidths=0.5, cbar=False)
    plt.title('Binary Sequence Data Visualization')
    plt.savefig(f'images/{idx}.png', bbox_inches='tight')
    plt.close()