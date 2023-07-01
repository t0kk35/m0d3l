"""
Module that contains some autoencoder layers.
(c) 2023 tsm
"""
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from ..layers.base import Layer

from typing import Union, Tuple

class AttentionLastEntry(Layer):
    def __init__(self, in_size: int, project_size: int):
        super(AttentionLastEntry, self).__init__()
        self._in_size = in_size
        self._p_size = project_size
        self._heads = 1
        self.k_weight = nn.parameter.Parameter(torch.zeros(self._in_size, self._p_size))
        self.q_weight = nn.parameter.Parameter(torch.zeros(self._in_size, self._p_size))
        self.v_weight = nn.parameter.Parameter(torch.zeros(self._in_size, self._p_size))
        self.reset_parameters()

    def forward(self, x: torch.Tensor, return_weights=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Apply weights to the input for key
        k = torch.unsqueeze(torch.matmul(x[:, -1, :], self.k_weight), dim=1)
        # Use last entry of the series as query and apply weights
        q = torch.matmul(x, self.q_weight)
        # Apply weight to the input for the value
        v = torch.matmul(x, self.v_weight)
        # Dot product each entry with the transpose of  last entry.
        w = torch.squeeze(torch.bmm(q, k.transpose(1, 2)), dim=2)
        # Softmax along the series axis. Un-squeeze to make broadcast possible
        w = torch.unsqueeze(nnf.softmax(w, dim=1), dim=2)
        # Scale weights so they don't explode and kill the gradients
        w = w / sqrt(self._in_size)
        # Multiply each entry with the weights.
        x = torch.mul(v, w)
        if return_weights:
            return x, w
        else:
            return x

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.k_weight, a=sqrt(5))
        nn.init.kaiming_uniform_(self.q_weight, a=sqrt(5))
        nn.init.kaiming_uniform_(self.v_weight, a=sqrt(5))

    @property
    def output_size(self) -> int:
        return self._p_size

    def extra_repr(self) -> str:
        return f'heads={self._heads}, hidden_size={self._p_size}'

class Attention(Layer):
    def __init__(self, in_size: int, heads: int, dropout: float):
        super(Attention, self).__init__()
        self._in_size = in_size
        self.attn = nn.MultiheadAttention(in_size, heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head attention expects the batch to be the second dim. (Series, Batch, Feature). Need to transpose
        x = x.transpose(0, 1)
        x, _ = self.attn(x, x, x, need_weights=False)
        return x.transpose(0, 1)

    @property
    def output_size(self) -> int:
        return self._in_size
