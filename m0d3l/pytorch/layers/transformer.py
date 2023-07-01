"""
Module that contains some transformer Layers
(c) 2023 tsm
"""
from math import log
import torch
import torch.nn as nn

from ..layers.base import Layer

class PositionalEncoding(Layer):
    def __init__(self, in_size: int, series_size: int, positional_size: int):
        super(PositionalEncoding, self).__init__()
        self._series_size = series_size
        self._positional_size = positional_size
        # Compute the positional encodings in log space
        pe = torch.zeros(series_size, positional_size)
        position = torch.arange(0, series_size).unsqueeze(1)
        d_term = torch.exp(torch.arange(0, positional_size, 2) * (-log(10000.0) / positional_size))
        pe[:, 0::2] = torch.sin(position * d_term)
        if positional_size % 2 == 0:
            pe[:, 1::2] = torch.cos(position * d_term)
        else:
            pe[:, 1::2] = torch.cos(position * d_term)[:, :-1]
        # Register encodings as buffer, so they do not become parameters.
        self.register_buffer('pe', pe)
        self._out_size = in_size + self._positional_size

    def forward(self, x):
        # Repeat along batch axis
        y = self.pe.repeat(x.shape[0], 1, 1)
        # Concatenate along last axis. I.e. add to the feature axis.
        x = torch.cat([x, y], dim=2)
        return x

    @property
    def output_size(self) -> int:
        return self._out_size

    def extra_repr(self) -> str:
        return f'series_size={self._series_size}, positional_size={self._positional_size}'


class PositionalEmbedding(Layer):
    def __init__(self, in_size: int, series_size: int, positional_size: int):
        super(PositionalEmbedding, self).__init__()
        self._series_size = series_size
        self._positional_size = positional_size
        self.pos_embedding = nn.Embedding(series_size, positional_size)
        self._out_size = in_size + self._positional_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the embedding for the positions
        y = self.pos_embedding(torch.arange(self._series_size, device=x.device, dtype=torch.long))
        # Repeat along batch axis
        y = y.repeat(x.shape[0], 1, 1)
        # Concatenate position with the original features.
        x = torch.cat([x, y], dim=2)
        return x

    @property
    def output_size(self) -> int:
        return self._out_size

    def extra_repr(self) -> str:
        return f'series_size={self._series_size}, positional_size={self._positional_size}'


class TransformerBody(Layer):
    def __init__(self, in_size: int, series_size: int, positional_size: int,  positional_logic: str,
                 heads: int, feedforward_size: int, drop_out: float):
        super(TransformerBody, self).__init__()
        if positional_logic == 'encoding':
            self.pos = PositionalEncoding(in_size, series_size, positional_size)
        else:
            self.pos = PositionalEmbedding(in_size, series_size, positional_size)
        self.trans = nn.TransformerEncoderLayer(self.pos.output_size, heads, feedforward_size, drop_out, 'relu')
        self._output_size = self.pos.output_size * series_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(x)
        # Transformers want (S, B, F) and the input is (B, S, F).
        x = self.trans(x.transpose(0, 1))
        x = torch.flatten(x.transpose(1, 0), start_dim=1)
        return x

    @property
    def output_size(self) -> int:
        return self._output_size
