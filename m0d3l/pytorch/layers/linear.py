"""
Module for linear layer combinations
(c) 2023 tsm
"""
import torch
import torch.nn as nn

from ..layers.base import Layer

from collections import OrderedDict
from typing import Tuple


class LinLayer(Layer):
    """
    Layer that runs a sequence of Linear/Activation operations. The definition will determine how many
    layers there are.
    The first layer will have size <input_size> and will then have up or down scale depending on the ratio's provided

    Args:
        input_size: This is the size of the previous layer.
        layer_sizes: A Tuple of int numbers that will define the size of the layers. For instance (32, 16) would
            make a first layer with <in_features> = <input_size> and <out_features>= 32. Then a second layer would be
            added with input <in_features> = 32 and <out_features> = 16
        dropout: A percentage dropout to apply after the linear layer. Default 0.0. If set to 0.0 then no dropout is
            applied
        bn_interval: Adds a BatchNorm layer every <bn_interval> layers. Default = 0. If set to 0 no BatchNorm layers
            will be added.
    """
    def __init__(self, input_size: int, layer_sizes: Tuple[int, ...], dropout: float = 0.0, bn_interval: int = 0):
        super(LinLayer, self).__init__()
        ls = OrderedDict()
        prev_size = input_size
        for i, s in enumerate(layer_sizes):
            ls.update({f'lin_layer_{i+1:02d}': nn.Linear(prev_size, s)})
            if bn_interval != 0 and ((i+1) % bn_interval == 0):
                ls.update({f'lin_bn_{i+1:02d}': nn.BatchNorm1d(s)})
            ls.update({f'lin_act_{i + 1:02d}': nn.ReLU()})
            if dropout != 0.0:
                ls.update({f'lin_dropout_{i+1:02d}': nn.Dropout(dropout)})
            prev_size = s
        self.layers = nn.Sequential(ls)
        self._output_size = prev_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
