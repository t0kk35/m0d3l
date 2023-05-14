"""
Module that contains some autoencoder layers.
(c) 2023 tsm
"""
import torch
import torch.nn as nn

from ..common.exception import PyTorchLayerException
from ..layers.base import Layer
from ..layers.heads import TensorConfiguration
from .linear import LinLayer
from .tails import CategoricalLogSoftmax1d

from f3atur3s import LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CATEGORICAL

from collections import OrderedDict
from typing import Tuple


class LinearEncoder(Layer):
    def __init__(self, input_size: int, layer_sizes: Tuple[int, ...],
                 dropout: float = 0.0, bn_interval: int = 0):
        super(Layer, self).__init__()
        self.enc_lin = LinLayer(input_size, layer_sizes, dropout, bn_interval, False)
        self._output_size = layer_sizes[-1]

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_lin(x)

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
        elif tensor_configuration.unique_learning_category == LEARNING_CATEGORY_CATEGORICAL:
            tail = CategoricalLogSoftmax1d(tensor_configuration, layer_sizes[-1])
            ls.update({'tail': tail})
            self._output_size = tail.hidden_dim * tail.output_size
        self.layers = nn.Sequential(ls)

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LinearVAEEncoder(Layer):
    def __init__(self, input_size: int, layer_sizes: Tuple[int, ...],
                 dropout: float = 0.0, bn_interval: int = 0):
        super(Layer, self).__init__()
        self._val_at_least_2_layers(layer_sizes)
        self.enc_lin = LinLayer(input_size, layer_sizes[:-1], dropout, bn_interval, False)
        self.mu = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.sigma = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self._output_size = layer_sizes[-1]

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Run through the head and linear layers.
        x = self.enc_lin(x)
        # Create the average latent dim.
        mu = self.mu(x)
        # Create the sigma latent dim
        s = self.sigma(x)
        # Return both the average and sigma latent dim.
        return mu, s

    @staticmethod
    def _val_at_least_2_layers(layer_sizes: Tuple[int, ...]):
        if len(layer_sizes) < 2:
            raise PyTorchLayerException(
                f'A Variational AutoEncoder should have at least 2 linear layer in the encoder. ' +
                f'Got layer-sizes {layer_sizes}.'
            )

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
        elif tensor_configuration.unique_learning_category == LEARNING_CATEGORY_CATEGORICAL:
            tail = CategoricalLogSoftmax1d(tensor_configuration, layer_sizes[-1])
            ls.update({'tail': tail})
            self._output_size = tail.hidden_dim * tail.output_size
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
