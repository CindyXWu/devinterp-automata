"""
Initialisation functions for mu-parameterisation. These follow the parameterisation in Table 3 of
"Tensor Programs V: Tuning Large Neural Networks via Zero-shot Hyperparameter Transfer". This is
because using the parameterisations in Table 8 requires altering the model with additional 
scalar multipliers.
"""
import math
from typing import Container, Sequence, Union
import torch.nn as nn
import logging
from mechanistic_distillation.config_schemas import DistributionType

from mechanistic_distillation.mup.inf_types import InfParam, InfType


def mup_initialise(
    named_params: Sequence[tuple[str, nn.Parameter]],
    param_inf_types: dict[str, InfParam],
    init_scale: Union[float, dict[str, float]],
    distribution: DistributionType,
    params_without_init: Container[str],
) -> None:
    """
    In-place initialise the parameters of a model using the MUP initialisation scheme described in
    "Tensor Programs V: Tuning Large Neural Networks via Zero-shot Hyperparameter Transfer".

    Note: The parameters are ASSUMED to have shape (fan_out, fan_in, ...) or (fan_out,)!! 
        The latter is the case for biases.

    Args:
        named_params: A sequence of (name, param) pairs, where name is the name of the parameter, and param is the parameter itself.
        param_inf_types: A dictionary mapping the names of the parameters to their infinite width type. The initialisation scheme
            will be different for each type of parameter.
        init_scale: The scale of the initialisation. This is a tunable hyperparameter constant that is independent of scale. If a dictionary,
            then the keys should be the names of the parameters, and the values should be the scale for that parameter.
    """
    # Input checks:
    if len(param_inf_types) != len(named_params):
        inf_type_nameset = set(param_inf_types.keys())
        named_params_nameset = set(name for name, _ in named_params)
        raise ValueError(
            f"The parameters in param_inf_types do not match the parameters in named_params.\n"
            f"The extra parameters in param_inf_types are: {inf_type_nameset - named_params_nameset}\n"
            f"The extra parameters in named_params are: {named_params_nameset - inf_type_nameset}"
        )

    # Initialise all params
    for name, param in named_params:
        if name in params_without_init:
            logging.info(f"Parameter with default initialization: {name}")
            continue
        inf_properties = param_inf_types[name]
        init_scale_for_param = init_scale[name] if isinstance(init_scale, dict) else init_scale

        mup_initialise_param(param, inf_properties=inf_properties, init_scale=init_scale_for_param, distribution=distribution)


def mup_initialise_param(param: nn.Parameter, inf_properties: InfParam, init_scale: float, distribution: DistributionType) -> None:
    """
    In-place initialise a parameter using the MUP initialisation scheme described in
    "Tensor Programs V: Tuning Large Neural Networks via Zero-shot Hyperparameter Transfer", as described
    in Table 3.

    Args:
        param: The parameter to initialise.
        inf_type: The infinite width type of the parameter.
        init_scale: The scale of the initialisation. This is a tunable hyperparameter constant that is independent of scale.
    """
    # Fan-in of a bias is 1
    if inf_properties.is_embedding:
        fan_in = 1
    else:
        fan_in = param.shape[1] if len(param.shape) >= 2 else 1

    match inf_properties.type:
        case InfType.INPUT_OR_BIAS | InfType.HIDDEN_WEIGHT:
            scale_multiplier = (1 / fan_in) ** 0.5
        case InfType.OUTPUT_WEIGHT:
            scale_multiplier = (1 / fan_in)
        case _:
            raise ValueError(f"Unrecognised infinite width type: {inf_properties}")
    # Initialise in place.
    initialise_param_with_std(param, scale=init_scale * scale_multiplier, distribution=distribution)


def scale_init_inplace(named_params: Sequence[tuple[str, nn.Parameter]], scales: Union[float, dict[str, float]]) -> None:
    """
    In-place scale the parameters of a model by a given scale.

    Args:
        named_params: A sequence of (name, param) pairs, where name is the name of the parameter, and param is the parameter itself.
        scale: The scale to multiply the parameters by.
    """
    for name, param in named_params:
        scale = scales if isinstance(scales, float) else scales[name]
        param.data *= scale
    

def standard_param_initialise(
    named_params: Sequence[tuple[str, nn.Parameter]],
    init_scale: Union[float, dict[str, float]],
    distribution: DistributionType,
    params_without_init: Container[str],
) -> None:
    """
    In-place initialise the parameters of a model using the standard_param initialisation scheme described in
    "Tensor Programs V: Tuning Large Neural Networks via Zero-shot Hyperparameter Transfer".

    Note: The parameters are ASSUMED to have shape (fan_out, fan_in, ...) or (fan_out,)!! 
        The latter is the case for biases.

    Args:
        named_params: A sequence of (name, param) pairs, where name is the name of the parameter, and param is the parameter itself.
        init_scale: The scale of the initialisation. This is a tunable hyperparameter constant that is independent of scale. If a dictionary,
            then the keys should be the names of the parameters, and the values should be the scale for that parameter.
        distribution: The distribution to use for the initialisation.
    """
    # Initialise all params
    for name, param in named_params:
        if name in params_without_init:
            logging.info(f"Parameter with default initialization: {name}")
            continue
        init_scale_for_param = init_scale[name] if isinstance(init_scale, dict) else init_scale
        standard_param_initialise_param(param, init_scale=init_scale_for_param, distribution=distribution)


def standard_param_initialise_param(param: nn.Parameter, init_scale: float, distribution: DistributionType) -> None:
    """
    In-place initialise a parameter using the standard_param initialisation scheme described in
    "Tensor Programs V: Tuning Large Neural Networks via Zero-shot Hyperparameter Transfer", as described
    in Table 3.

    Args:
        param: The parameter to initialise.
        init_scale: The scale of the initialisation. This is a tunable hyperparameter constant that is independent of scale.
    """
    # Fan-in of a bias is 1
    fan_in = param.shape[1] if len(param.shape) >= 2 else 1

    scale_multiplier = (1 / fan_in) ** 0.5
    # Initialise in place.
    initialise_param_with_std(param, scale=init_scale * scale_multiplier, distribution=distribution)


def torch_param_initialise(
    named_params: Sequence[tuple[str, nn.Parameter]],
    init_scale: Union[float, dict[str, float]],
    distribution: DistributionType,
    params_without_init: Container[str],
) -> None:
    """
    Args:
        named_params: A sequence of (name, param) pairs, where name is the name of the parameter, and param is the parameter itself.
        init_scale: The scale of the initialisation. This is a tunable hyperparameter constant that is independent of scale. If a dictionary,
            then the keys should be the names of the parameters, and the values should be the scale for that parameter.
        distribution: The distribution to use for the initialisation.
    """
    # Initialise all params
    for name, param in named_params:
        if name in params_without_init:
            logging.info(f"Parameter with default initialization: {name}")
            continue
        init_scale_for_param = init_scale[name] if isinstance(init_scale, dict) else init_scale
        # If parameter is a bias, find the matching weight to calculate fan_in with
        if name.endswith(".bias"):
            # Only replace the last instance of ".bias" at the end of the str with ".weight"
            weight_name = name.rsplit(".", 1)[0] + ".weight"
            assert weight_name in dict(named_params), f"Could not find a matching weight for bias {name}"
            weight = dict(named_params)[weight_name]
            layer_fan_in = weight.shape[1]
        elif param.ndim >= 2:
            layer_fan_in = param.shape[1]
        else:
            raise ValueError(f"Parameter {name} has a shape of {param.shape} and is not a bias, which is not supported by torch_param_initialise")
        scale_multiplier = (1 / layer_fan_in) ** 0.5
        # Initialise in place.
        initialise_param_with_std(param, scale=init_scale_for_param * scale_multiplier, distribution=distribution)


def initialise_param_with_std(param: nn.Parameter, scale: float, distribution: DistributionType) -> None:
    """

    Args:
        param: The parameter to initialise.
        init_scale: The scale of the initialisation. This is a tunable hyperparameter constant that is independent of scale.
        distribution: The distribution to initialise the parameters with ("NORMAL", "UNIFORM").
    """
    # Initialise in place.
    if distribution == DistributionType.NORMAL:
        nn.init.normal_(param, mean=0.0, std=scale)
    elif distribution == DistributionType.UNIFORM:
        nn.init.uniform_(param, a=-scale * math.sqrt(3), b=scale * math.sqrt(3))
    else:
        raise ValueError(f"Unrecognised distribution type: {distribution}")
