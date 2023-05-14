"""
Module for common layers at the end of a NN
(c) 2023 tsm
"""
import torch
import torch.nn as nn
from abc import ABC

from math import sqrt
from ...common.modelconfig import TensorConfiguration
from ..layers.base import Layer

from typing import Tuple


class CategoricalLogSoftmax(Layer, ABC):
    def __init__(self, tensor_configuration: TensorConfiguration, input_rank: int, es_expr: str, use_mask=False):
        Layer.__init__(self)
        self.use_mask = use_mask
        self.input_rank = input_rank
        self.ein_sum_expression = es_expr
        self._i_features = tuple([n for n, _ in tensor_configuration.categorical_features])
        self._sizes = tuple([c for _, c in tensor_configuration.categorical_features])
        self._hidden_dim = max(self._sizes)
        self._class_dim = len(self._sizes)
        self.lsm = nn.LogSoftmax(dim=self.input_rank-1)

    @property
    def sizes(self) -> Tuple[int, ...]:
        return self._sizes

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def output_size(self) -> int:
        return self._class_dim

    @staticmethod
    def calculate_fan_in_and_fan_out(x: torch.Tensor):
        dimensions = x.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        num_input_maps = x.size()[1]
        num_output_maps = x.size()[0]
        receptive_field_size = 1
        if x.dim() > 2:
            receptive_field_size = x[0][0].numel()
        fan_in = num_input_maps * receptive_field_size
        fan_out = num_output_maps * receptive_field_size
        return fan_in, fan_out

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.f_weight, a=sqrt(5))
        fan_in, _ = self.calculate_fan_in_and_fan_out(self.f_weight)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.f_bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        x = torch.einsum(self.ein_sum_expression, x, self.f_weight)
        x = x + self.f_bias
        if self.mask is not None:
            x = x * self.mask
        x = self.lsm(x)
        return x

class CategoricalLogSoftmax1d(CategoricalLogSoftmax):
    def __init__(self, tensor_configuration: TensorConfiguration, input_size: int, use_mask=False):
        super(CategoricalLogSoftmax1d, self).__init__(tensor_configuration, 2, 'bi,ilc->blc', use_mask)
        self.f_weight = nn.parameter.Parameter(torch.zeros(input_size, self.hidden_dim, self.output_size))
        self.f_bias = nn.parameter.Parameter(torch.zeros(self.hidden_dim, self.output_size))
        mask = torch.zeros(self.hidden_dim, self.output_size)
        for i, s in enumerate(self._sizes):
            mask[:s+1, i] = 1.0
        self.register_buffer('mask', mask if use_mask else None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f'max_dim={self.hidden_dim}, classes={self.output_size}, use_mask={self.use_mask}'


class CategoricalLogSoftmax2d(CategoricalLogSoftmax):
    def __init__(self, tensor_configuration: TensorConfiguration, input_size: int, use_mask=False):
        super(CategoricalLogSoftmax2d, self).__init__(tensor_configuration, 3, 'bsi,ilc->bslc', use_mask)
        self.f_weight = nn.parameter.Parameter(torch.zeros(input_size, self.hidden_dim, self.class_dim))
        self.f_bias = nn.parameter.Parameter(torch.zeros(self.hidden_dim, self.class_dim))
        mask = torch.zeros(self.hidden_dim, self.class_dim)
        for i, s in enumerate(self.sizes):
            mask[:s+1, i] = 1.0
        self.register_buffer('mask', mask if use_mask else None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f'max_dim={self.hidden_dim}, classes={self.output_size}, use_mask={self.use_mask}'
