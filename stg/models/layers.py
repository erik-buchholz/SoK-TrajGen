#!/usr/bin/env python3
"""Helpful layers for the models"""
import logging
from enum import Enum

from torch import nn

log = logging.getLogger()


class NormOption(str, Enum):
    """Enum for the different normalization options"""

    BATCH_1D = 'batch1d'
    BATCH = 'batch1d'
    LAYER = 'layer'
    NONE = 'none'
    DROPOUT = 'dropout'


class ActivationOption(str, Enum):
    """Enum for the different activation options"""

    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'


ACTIVATION_MAP = {
    ActivationOption.RELU: nn.ReLU,
    ActivationOption.LEAKY_RELU: nn.LeakyReLU,
    ActivationOption.TANH: nn.Tanh,
    ActivationOption.SIGMOID: nn.Sigmoid,
}


class Norm(nn.Module):
    """Abstraction Layer to quickly switch between different normalization layers"""

    def __init__(self, channels: int, norm_type: NormOption, channels_last: bool = False):
        """

        :param channels: Number of channels
        :param norm_type: see NORM_OPTIONS
        :param channels_last: If false, input expected to be (N, C, L) or (N, C), if true (N, L, C)
            Of course, in the case of (N, C), the channels are last, but in this case the option is ignored.
        """
        super().__init__()
        self.channels = channels
        self.norm_type = norm_type
        self.channels_last = channels_last
        if norm_type == NormOption.BATCH_1D or norm_type == NormOption.BATCH:
            self.norm = nn.BatchNorm1d(channels)
        elif norm_type == NormOption.LAYER:
            self.norm = nn.LayerNorm(channels)
        elif norm_type == NormOption.NONE:
            self.norm = nn.Identity()
        elif norm_type == NormOption.DROPOUT:
            self.norm = nn.Dropout(0.5)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

    def forward(self, x):
        log.debug(f"Norm: x.shape = {x.shape}")
        swap = self.norm_type == 'layer' and len(x.shape) > 2 and not self.channels_last
        if swap:
            # x might either have shape (N, C) or (N, C, L), but for layer norm, C needs to be in the end!
            x = x.transpose(2, 1)
        y = self.norm(x)
        if swap:
            y = y.transpose(2, 1)
        return y


class Activation(nn.Module):
    """Abstraction Layer to quickly switch between different activation functions"""

    def __init__(self, mode: ActivationOption):
        super().__init__()
        mode = mode.lower()
        if mode == ActivationOption.LEAKY_RELU:
            self.activation = nn.LeakyReLU(0.2)
        elif mode in ACTIVATION_MAP:
            self.activation = ACTIVATION_MAP[mode]()
        else:
            raise ValueError(f"Unknown activation: {mode}.")

    def forward(self, x):
        return self.activation(x)

    def __str__(self):
        return str(self.activation)


class CustomLinear(nn.Module):
    """
    Abstraction Layer to quickly switch between different linear and equivalent layers (conv1d).
    This was just an experiment, but it does not make much sense to use this as a Linear Layer
    and Conv1D layer with kernel size 1 are equivalent in terms of output.
    Of course, this only true for batches of shape (N, C).
    """

    def __init__(self, in_features: int, out_features: int, *args, mode: str = 'linear'):
        super().__init__()
        mode = mode.lower()
        self.mode = mode
        self.in_features = in_features
        self.out_features = out_features
        if mode == 'linear':
            self.layer = nn.Linear(in_features, out_features, *args)
        elif mode == 'conv1d':
            self.layer = nn.Conv1d(in_features, out_features, 1, *args)
        else:
            raise ValueError(f"Unknown Layer type: {mode}.")
        self.weight = self.layer.weight

    def forward(self, x):
        transpose = False
        if self.mode == 'conv1d' and x.dim() < 3:
            x = x.view(-1, self.in_features, 1)
            transpose = True
        x = self.layer(x)
        if transpose:
            x = x.view(-1, self.out_features)
        return x
