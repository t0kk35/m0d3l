"""
Module for Convolutional Layers
(c) 2023 tsm
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from ..layers.base import Layer

from collections import OrderedDict
from typing import Tuple, Type


class ConvolutionalBodyBase1d(Layer, ABC):
    def __init__(self,  conv_cls: Type[nn.Module], in_size: int, series_size: int,
                 conv_layers: Tuple[Tuple[int, int, int], ...],
                 drop_out: float, batch_norm_interval=2, activate_last=True):
        super(ConvolutionalBodyBase1d, self).__init__()
        self._p_conv_cls = conv_cls
        self._p_in_size = in_size
        self._p_conv_layers = conv_layers
        self._p_series_size = series_size
        self._p_drop_out = drop_out
        self._p_batch_norm_interval = batch_norm_interval
        self._p_activate_last = activate_last
        self._output_size = 0
        self._output_series_length = 0
        ly = OrderedDict()
        prev_size = self._p_series_size
        prev_channels = self._p_in_size
        for i, (out_channels, kernel, stride) in enumerate(conv_layers):
            ly.update({f'conv_{i+1:02d}': conv_cls(prev_channels, out_channels, kernel, stride)})
            # Don't activate last if requested
            if i + 1 != len(conv_layers) or activate_last:
                ly.update({f'relu_{i+1:02d}': nn.ReLU()})
            # Add BatchNorm Every 'batch_norm_interval' layer
            if (i+1) % batch_norm_interval == 0:
                ly.update({f'norm_{i+1:02d}': nn.BatchNorm1d(out_channels)})
            prev_channels = out_channels
            prev_size = self._layer_out_size(prev_size, kernel, stride, 0)
        if drop_out != 0.0:
            ly.update({f'dropout': nn.Dropout(drop_out)})
        self.conv_layers = nn.Sequential(ly)
        self.output_series_length = prev_size
        # Output size is the reduced series length times the # filters in the last layer
        self.output_size = self.output_series_length * conv_layers[-1][0]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def _layer_out_size(in_size: int, kernel: int, stride: int, padding: int) -> int:
        return int(((in_size - kernel + (2 * padding)) / stride) + 1)

    @property
    def output_size(self) -> int:
        return self._output_size

    @output_size.setter
    def output_size(self, size: int):
        self._output_size = size

    @property
    def output_series_length(self) -> int:
        return self._output_series_length

    @output_series_length.setter
    def output_series_length(self, series_length: int):
        self._output_series_length = series_length

    def copy(self) -> Layer:
        c = ConvolutionalBodyBase1d(
            self._p_conv_cls, self._p_in_size, self._p_series_size, self._p_conv_layers, self._p_drop_out,
            self._p_batch_norm_interval, self._p_activate_last
        )
        c.copy_state_dict(self)
        return c


class ConvolutionalBody1d(ConvolutionalBodyBase1d):
    def __init__(self, in_size: int, series_size: int, conv_layers: Tuple[Tuple[int, int, int], ...], drop_out: float):
        super(ConvolutionalBody1d, self).__init__(nn.Conv1d, in_size, series_size, conv_layers, drop_out)

    @staticmethod
    def _layer_out_size(in_size: int, kernel: int, stride: int, padding: int) -> int:
        return int(((in_size - kernel + (2 * padding)) / stride) + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Switch series and feature dim, Conv layers want the channel/feature dim as second, the series as third.
        y = x.transpose(1, 2)
        # Do convolutions + Potential Dropout
        y = self.conv_layers(y)
        # Flatten out to Rank-2 tensor
        y = torch.flatten(y, start_dim=1)
        return y
