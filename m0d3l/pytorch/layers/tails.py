"""
Module for common layers at the end of a NN
(c) 2023 tsm
"""
import torch
import torch.nn as nn
from math import sqrt
from ..layers.base import Layer
from collections import OrderedDict
from typing import List, Tuple


class TailBinary(Layer):
    """
    Layer that runs a sequence of Linear/Drop-out/Activation operations. The definition will determine how many
    layers there are.
    For instance definition = [(128,0.0),(64,0.0),(32.0.1) will create 3 Linear Layers of 128, 64 and 32 features
    respectively. A dropout of 0.1 will be applied to the last layer.
    After the linear layers it runs a final linear layer of output size to and a sigmoid, to come to a binary output.
    Args:
        input_size: The size of the first layer. This must be the same as the output size of the previous layer
        definition: A List of Tuples. Each entry in the list will be turned into a layer. The Tuples must be
            of type [int, float]. The int is the number of features in that specific layer, the float is the dropout
            rate at that layer. If the dropout is 0.0 no dropout will be performed.
    """
    def __init__(self, input_size: int, definition: List[Tuple[int, float]], add_bn=True):
        super(TailBinary, self).__init__()
        ls = OrderedDict()
        prev_size = input_size
        for i, (o_size, dropout) in enumerate(definition[:-1]):
            ls.update({f'tail_lin_{i+1:02d}': nn.Linear(prev_size, o_size)})
            ls.update({f'tail_act_{i+1:02d}': nn.ReLU()})
            if dropout != 0:
                ls.update({f'tail_dropout_{i+1:02d}': nn.Dropout(dropout)})
            prev_size = o_size
        # Add batch norm is requested and there is more than 1 layer.
        if add_bn and len(definition) > 1:
            ls.update({f'tail_batch_norm': nn.BatchNorm1d(prev_size)})
        # Add Last Binary layer
        ls.update({f'tail_binary': nn.Linear(prev_size, definition[-1][0])})
        ls.update({f'tail_bin_act': nn.Sigmoid()})
        self.layers = nn.Sequential(ls)

    @property
    def output_size(self) -> int:
        return 1

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # If we received multiple Tensors, there were multiple streams, which we will concatenate before applying linear
        if isinstance(x, List) and len(x) > 1:
            x = torch.cat(x, dim=1)
        else:
            x = x[0]
        return self.layers(x)
