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


def get_submodules(model: nn.Module) -> dict[str, nn.Module]:
    """Get names of all submodules in model as dict. 
    Use this to find the names of the layers in the model for feature extractors.
    
    Args:
        model: The model.
    
    Returns:
        A dict mapping the names of the submodules to the names of the layers.
    """
    type_prefixes = {
        nn.Conv2d: 'conv',
        nn.BatchNorm2d: 'bn',
        nn.Linear: 'fc'
    }

    submodules = {
        name: f"{type_prefixes.get(type(module), '')}_{name.split('.')[-1]}" if type(module) in type_prefixes else name 
        for name, module in model.named_modules()
    }

    return submodules


def show_model(model):
    """Cursed little ad-hoc print function to check model modules."""
    print("List of model layers:")
    for idx, module in enumerate(model.children()):
        print(f"Layer {idx}: {module}")