import random
from typing import Protocol

import numpy as np
import torch

from devinfra.utils.device import DeviceOrDeviceLiteral


class Seedable(Protocol):
    def set_seed(self, seed: int):
        ...


def set_seed(seed: int, *seedables: Seedable, device: DeviceOrDeviceLiteral = "cpu"):
    """
    Sets the seed for the Learner.

    Args:
        seed (int): Seed to set.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    for seedable in seedables:
        seedable.set_seed(seed)

    if "cuda" in str(device):
        torch.cuda.manual_seed_all(seed)
    elif "xla" in str(device):
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed)
    elif "mps" in str(device):
        pass