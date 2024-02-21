from typing import Iterable, Literal, Union

import numpy as np
import torch

Reduction = Literal["mean", "sum", "none"]
ReturnTensor = Literal["pt", "tf", "np"]
TensorLike = Union[np.ndarray, torch.Tensor]


def convert_tensor(x: Iterable, return_tensor: ReturnTensor):
    """
    Converts a tensor to pytorch, tensorflow (not implemented yet) or numpy format.
    """
    if return_tensor == "pt":
        return torch.Tensor(x)
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    
    if return_tensor == "tf":
        raise NotImplementedError
        # return tf.convert_to_tensor(x)
    elif return_tensor == "np":
        return np.array(x)
    else:
        raise ValueError(f"Unknown return_tensor: {return_tensor}")


def reduce_tensor(xs: TensorLike, reduction: Reduction):
    """
    Reduces a tensor to a single value.
    """
    if reduction == "mean":
        return xs.mean()
    elif reduction == "sum":
        return xs.sum()
    elif reduction == "none":
        return xs
    else:
        raise ValueError(f"Unknown reduction: {reduction}")