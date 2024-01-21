import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Sequence
import numpy as np


class ParityDataset(Dataset):
    """Dataset with predecided parity."""
    def __init__(self, dataset_size: int, n_bits: int, subset_idxs: Optional[Sequence[int]] = None):
        self.dataset_size: int = dataset_size
        self.n_bits: int = n_bits
        self.y: torch.Tensor = torch.randint(0, 2, (dataset_size,))
        self.X: torch.Tensor = torch.tensor([self.generate_num_with_parity(y, n_bits) for y in self.y], dtype=torch.int)

        # Select a subset if indices are provided
        if subset_idxs is not None:
            self.X, self.y = self.X[subset_idxs], self.y[subset_idxs]

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx].float(), self.y[idx].float()

    def generate_num_with_parity(self, parity: int, n_bits: int) -> int:
        num_of_ones = np.random.choice(range(parity, n_bits + 1, 2))  # Choose an even or odd number of 1's based on the desired parity
        position_of_ones = np.random.choice(range(n_bits), num_of_ones, replace=False)  # Choose the positions for 1's randomly
        binary_str = [1 if i in position_of_ones else 0 for i in range(n_bits)]  # Create a binary string with 1's in the selected positions
        num = int(''.join(map(str, binary_str)), 2)  # Convert binary string back to decimal
        return num

