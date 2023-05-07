"""
Module that contains some autoencoder layers.
(c) 2023 tsm
"""
import torch
import torch.nn as nn

from ..layers.base import Layer
from ..layers.heads import TensorDefinitionHead, TensorConfiguration
from .linear import LinLayer

from f3atur3s import LearningCategory, LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CATEGORICAL

from collections import OrderedDict
from typing import Tuple


class LinearEncoder(Layer):
    def __init__(self, tensor_configuration: TensorConfiguration, layer_sizes: Tuple[int, ...],
                 dropout: float = 0.0, bn_interval: int = 0):
        super(Layer, self).__init__()
        ls = OrderedDict()
        ls.update({'head': TensorDefinitionHead(tensor_configuration, 0.5, 5, 10, 0.0)})
        ls.update({'linear_encoder': LinLayer(ls.get('head').output_size, layer_sizes, dropout, bn_interval, False)})
        self.layers = nn.Sequential(ls)
        self._output_size = layer_sizes[-1]

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class LinearDecoder(Layer):
    def __init__(self, tensor_configuration: TensorConfiguration, layer_sizes: Tuple[int, ...],
                 dropout: float = 0.0, bn_interval: int = 0):
        super(Layer, self).__init__()
        ls = OrderedDict()
        ls.update({'linear_decoder': LinLayer(layer_sizes[0], layer_sizes[1:], dropout, bn_interval)})
        if tensor_configuration.unique_learning_category == LEARNING_CATEGORY_BINARY:
            ls.update({'tail': nn.Linear(layer_sizes[-1], len(tensor_configuration.binary_features))})
            ls.update({'final-act': nn.Sigmoid()})
            self._output_size = len(tensor_configuration.binary_features)
        self.layers = nn.Sequential(ls)

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LinearVAEEncoder(Layer):
    def __init__(self, tensor_configuration: TensorConfiguration, layer_sizes: Tuple[int, ...],
                 dropout: float = 0.0, bn_interval: int = 0):
        super(Layer, self).__init__()
        ls = OrderedDict()
        ls.update({'head': TensorDefinitionHead(tensor_configuration, 0.5, 5, 10, 0.0)})
        prev_size = ls.get('head').output_size
        if len(layer_sizes) > 1:
            ls.update({'linear_encoder': LinLayer(prev_size, layer_sizes[:-1], dropout, bn_interval, False)})
            prev_size = layer_sizes[-2]
        self.layers = nn.Sequential(ls)
        self.mu = nn.Linear(prev_size, layer_sizes[-1])
        self.sigma = nn.Linear(prev_size, layer_sizes[-1])
        self._output_size = layer_sizes[-1]

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Run through the head and linear layers.
        x = self.layers(x)
        # Create the average latent dim.
        mu = self.mu(x)
        # Create the sigma latent dim
        s = self.sigma(x)
        # Return both the average and sigma latent dim.
        return mu, s

class LinearVAEDecoder(Layer):
    def __init__(self, tensor_configuration: TensorConfiguration, layer_sizes: Tuple[int, ...],
                 dropout: float = 0.0, bn_interval: int = 0):
        super(Layer, self).__init__()
        ls = OrderedDict()
        ls.update({'linear_decoder': LinLayer(layer_sizes[0], layer_sizes[1:], dropout, bn_interval)})
        if tensor_configuration.unique_learning_category == LEARNING_CATEGORY_BINARY:
            ls.update({'tail': nn.Linear(layer_sizes[-1], len(tensor_configuration.binary_features))})
            ls.update({'final-act': nn.Sigmoid()})
            self._output_size = len(tensor_configuration.binary_features)
        self.layers = nn.Sequential(ls)

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        # The input is not just a single Tensor, but a Tuple. That is what the LinearVAEEncoder has set.
        mu, s = x
        # first re-parameterize the latent mu and s.
        s = torch.exp(0.5*s)
        eps = torch.randn_like(s)
        x = mu + eps * s
        # Now run through the linear layers.
        x = self.layers(x)
        # Attention! We need to also return the mu and s for the loss calculation. See VAELoss functions.
        return x, mu, s
