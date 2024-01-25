import torch
import torch.nn as nn


def get_param_name(module: nn.Module, param: torch.Tensor) -> str:
    """
    Get the name of a parameter in a module.

    Args:
        module: The module containing the parameter.
        param: The parameter to get the name of.

    Returns:
        The name of the parameter.
    """
    for name, param_ in module.named_parameters():
        if param_ is param:
            return name

    raise ValueError(f"Parameter {param} not found in module {module}.")
