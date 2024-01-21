from typing import Callable
from collections import OrderedDict
from torch import nn


def mlp_constructor(
    input_size: int,
    hidden_sizes: list[int],
    output_size: int,
    activation_constructor: Callable[[], nn.Module] = nn.ReLU,
    flatten_input: bool = True,
    bias: bool = True,
    weight_initialisation: Callable = nn.init.kaiming_normal_,
) -> nn.Sequential:
    """Default weight initialisation is Kaiming normal."""
    layers: list[tuple[str, nn.Module]] = [("input_layer", nn.Linear(input_size, hidden_sizes[0], bias=bias))]

    for i in range(len(hidden_sizes) - 1):
        in_size, out_size = hidden_sizes[i : i + 2]
        layers.append((f"activation{i}", activation_constructor()))
        layers.append((f"hidden_layer{i}", nn.Linear(in_size, out_size, bias=bias)))
    layers.append((f"activation{len(hidden_sizes)}", activation_constructor()))
    layers.append((f"output_layer", nn.Linear(hidden_sizes[-1], output_size, bias=False)))
    if flatten_input:
        layers.insert(0, ("input_flatten", nn.Flatten()))

    layers_dict = OrderedDict(layers)
    model = nn.Sequential(layers_dict)
    
    # Apply weight initialisation to each layer
    model.apply(lambda m: init_weights(m, weight_initialisation))
    return model


# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.kaiming_normal_(m.weight, a=1, mode="fan_in")


def init_weights(m, weight_initialisation=nn.init.kaiming_normal_):
    if type(m) == nn.Linear:
        weight_initialisation(m.weight)


def init_weights_uniform_constructor(a=-0.1, b=-0.1):
    def init_weights_uniform(m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, a=a, b=b) # a and b are the lower and upper bounds of the uniform distribution
            if m.bias is not None:
                nn.init.uniform_(m.bias, a=a, b=b)
    return init_weights_uniform