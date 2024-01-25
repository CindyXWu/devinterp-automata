import functools
from typing import Callable, Protocol
import torch
import torch.nn as nn

from torch import Tensor
from ib_fcnn.architectures.common import Identity, SplitAggregate


class NormalizationConstructorType(Protocol):
    def __call__(self, num_features: int) -> nn.Module:
        ...


def wide_resnet_constructor(
        blocks_per_stage: int,
        width_factor: int,
        activation_constructor: Callable[[], nn.Module] = functools.partial(nn.ReLU, inplace=True),
        normalization_constructor: NormalizationConstructorType = nn.BatchNorm2d,
    ) -> nn.Sequential:
    """
    Construct a Wide ResNet model.

    Follows the architecture described in Table 1 of the paper:
        Wide Residual Networks
        https://arxiv.org/pdf/1605.07146.pdf

    This is a Wide-ResNet with the block structure following the variant (sometimes known as ResNetV2) described in:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/abs/1603.05027

    Args:
        blocks_per_stage: Number of blocks per stage.
        width_factor: Width factor.

    Returns:
        The constructed model.
    """
    assert blocks_per_stage >= 1, f"blocks_per_stage must be >= 1, got {blocks_per_stage}"

    def block_constructor(in_channels: int, out_channels: int) -> nn.Module:
        return SplitAggregate(
            # Skip connection
            Identity(),
            # Conv. block
            nn.Sequential(
                normalization_constructor(in_channels),
                activation_constructor(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                normalization_constructor(out_channels),
                activation_constructor(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )
        )
    
    def downsample_block_constructor(in_channels: int, out_channels: int) -> nn.Module:
        return SplitAggregate(
            # Skip connection with 1x1 conv downsampling
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            # Conv. block
            nn.Sequential(
                normalization_constructor(in_channels),
                activation_constructor(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                normalization_constructor(out_channels),
                activation_constructor(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )
        )

    model = nn.Sequential(
        # The output width of the first conv. layer being 16 * width_factor is a slight
        # deviation from the paper repository 
        # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py
        nn.Conv2d(3, 16 * width_factor, kernel_size=3, stride=1, padding=1, bias=False),
        normalization_constructor(16 * width_factor),
        activation_constructor(),
        # Stage 1
        block_constructor(16 * width_factor, 16 * width_factor),
        *(block_constructor(16 * width_factor, 16 * width_factor) for _ in range(blocks_per_stage - 1)),
        # Stage 2
        downsample_block_constructor(16 * width_factor, 32 * width_factor),
        *(block_constructor(32 * width_factor, 32 * width_factor) for _ in range(blocks_per_stage - 1)),
        # Stage 3
        downsample_block_constructor(32 * width_factor, 64 * width_factor),
        *(block_constructor(64 * width_factor, 64 * width_factor) for _ in range(blocks_per_stage - 1)),
        # Output
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64 * width_factor, 10),
    )

    # Initialise
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    return model