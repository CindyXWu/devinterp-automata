import torch
import random
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple
from ib_fcnn.config_setup import DatasetConfig


class PalindromeDataset(Dataset):
    def __init__(self, dataset_config: DatasetConfig):
        """Datapoints are binary strings. Labels are 0-1 depending on whether the string contains a palindrome of any length.
        Args:
            n: Length of the binary strings
            dataset_size: Number of samples in the dataset
            p_prob: Probability of inserting a palindrome
        """
        self.n: int = dataset_config.input_length
        self.dataset_size: int = dataset_config.dataset_size
        self.p_prob: float = dataset_config.p_prob
        self.X, self.y = torch.zeros((self.dataset_size, self.n)), torch.zeros(self.dataset_size)
        for i in range(self.dataset_size):
            self.X[i], self.y[i] = self.generate_random_string()
            
    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def generate_random_string(self) -> Tuple[str, torch.Tensor]:
        """Option to return s as a binary string or one-hot encoded tensor."""
        # Generate random binary string
        s = "".join(str(random.randint(0, 1)) for _ in range(self.n))
        # Generate and insert palindrome at random location
        if random.random() < self.p_prob:
            m = random.randint(1, self.n // 2)  # Palindrome length
            palindrome = "".join(str(random.randint(0, 1)) for _ in range(m))
            palindrome = palindrome + palindrome[::-1]
            insert_idx = random.randint(0, self.n - len(palindrome))
            s = s[:insert_idx] + palindrome + s[insert_idx + len(palindrome):] # Ensure total length remains self.n
            y = torch.tensor(1)
        else:
            y = torch.tensor(0)
        s = torch.tensor([int(char) for char in s], dtype=torch.long) # s to tensor

        return s, y


def save_to_csv(
    dataset: Dataset, 
    filename: str,
    input_col_name: str ='input',
    label_col_name: str ='label') -> None:
    data = [(str(x.numpy()), int(y.numpy())) for x, y in dataset]
    df = pd.DataFrame(data, columns=[input_col_name, label_col_name])
    df.to_csv(filename, index=False)
    
    
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    config = DatasetConfig(input_length=20, dataset_size=100000)
    dataset = PalindromeDataset(config)
    save_to_csv(dataset, PROJECT_ROOT / f'data/PalindromeDataset_{config.input_length}_{config.dataset_size}.csv')