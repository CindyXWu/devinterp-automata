"""
The different "types" of parameters of the model. These will be initialised differently.,
and will have their optimiser's parameters scaled differently, depending on how they behave
in the infinite width limit.
"""

from enum import Enum
from operator import itemgetter
from typing import NamedTuple, Sequence, Union

import torch
import torch.nn as nn
from mechanistic_distillation.architectures.nano_gpt import LayerNorm

from mechanistic_distillation.mup.utils import get_param_name


class InfType(Enum):
    """
    See "Tensor Programs V: Tuning Large Neural Networks via Zero-shot Hyperparameter Transfer"
    - Table 3 by Greg Yang for details.
    """

    INPUT_OR_BIAS = "input_or_bias"
    HIDDEN_WEIGHT = "hidden_weight"
    OUTPUT_WEIGHT = "output_weight"
    INPUT_EMBEDDING = "input_embedding"


class InfParam(NamedTuple):
    type: InfType
    is_embedding: bool = False


def get_inf_types(
    model: torch.nn.Module, input_weights_names: Sequence[str], output_weights_names: Sequence[str]
) -> dict[str, InfParam]:
    """
    Given a model, and a manual specification by the user of which parameters are input weights and
    which are output weights, return a dictionary mapping the names of the parameters to their
    infinite width type.

    Return:
    - A dictionary mapping the names of the parameters to their infinite width type.
    """
    # Assert the input and output weight names are disjoint
    assert set(input_weights_names).isdisjoint(output_weights_names), "Input and output weight names must be disjoint"

    named_params: list[tuple[str, nn.Parameter]] = list(model.named_parameters())
    bias_names = {name for name, param in named_params if len(param.shape) <= 1}

    # Embedding layers in pytorch have fan-in, fan-out reversed, need to be treated differently
    embedding_weight_names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Embedding)):
            embedding_weight_names.update({get_param_name(model, param) for param in module.parameters()})


    # Get a mapping from inf. type to list of param. names of that inf. type
    inf_type_groups: dict[InfType, list[str]] = {
        InfType.INPUT_OR_BIAS: set(input_weights_names) | bias_names,
        InfType.OUTPUT_WEIGHT: set(output_weights_names),
        InfType.HIDDEN_WEIGHT: {name for name, param in named_params} - set(input_weights_names) - set(output_weights_names) - bias_names,
    }
    # Invert the mapping to be a mapping from param. name to inf. type
    return {
        name: InfParam(type=inf_type, is_embedding=name in embedding_weight_names) for inf_type, names in inf_type_groups.items() for name in names
    }


def get_params_without_init(model: torch.nn.Module) -> set[str]:
    """Get parameters that should not be initialised with mup (e.g. elementwise multipliers)"""
    param_names_without_init = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (torch.nn.LayerNorm, nn.GroupNorm, torch.nn.modules.batchnorm._BatchNorm, LayerNorm)):
            param_names_without_init.update(
                {
                    get_param_name(model, param)
                    for name, param in module.named_parameters()
                    if name.endswith("weight")
                }
            )
    return param_names_without_init