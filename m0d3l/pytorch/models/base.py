"""
Common classes for all Pytorch Models
(c) 2023 tsm
"""
import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from typing import Any, Tuple

from f3atur3s import TensorDefinition
from eng1n3.pandas import TensorInstanceNumpy

from ..common.exception import PyTorchModelException
from ..training.loss import Loss
from ..training.history import History
from ..layers.heads import TensorDefinitionHead


class Model(nn.Module, ABC):
    def __init__(self):
        nn.Module.__init__(self)

    def _forward_unimplemented(self, *inp: Any) -> None:
        raise NotImplemented('Abstract method _forward_unimplemented not implemented')

    @abstractmethod
    def get_x(self, ds: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Get the values that are considered the x values. I.e. the independent variables, I.e. NOT the label.

        Args:
            ds: A Tuple of tensors as read from a DataLoader object.

        Return:
            A Tuple of tensors to be used as input to a neural net.
        """
        pass

    @abstractmethod
    def get_y(self, ds: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Get the values that are considered the y values. I.e. the dependent variable, I.e. the label

        Args:
            ds: A Tuple of tensors as read from a DataLoader object.

        Return:
            A Tuple of tensors to be used as label for the neural net.
        """
        pass

    def forward(self, x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Forward method for the standard models. This is the logic of the forward pass through a Neural Net. It will
        become the 'forward' of the torch 'nn.Module'.

        Args:
            x: The input to the nn.Module. A Tuple of torch tenors.

        Returns:
            Output of the nn.Module as Tuple of tensors.
        """
        pass

    @property
    @abstractmethod
    def x_indexes(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def label_index(self) -> int:
        pass

    @property
    @abstractmethod
    def loss_fn(self) -> Loss:
        pass

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def history(self, batch_size: int, sample_number: int, step_number: int) -> History:
        pass

    def extra_repr(self) -> str:
        return f'Number of parameters : {self.num_parameters}. Loss : {self.loss_fn}'

    def save(self, path: str):
        if os.path.exists(path):
            raise PyTorchModelException(f'File {path} already exists. Not overriding model')
        torch.save(self._model.state_dict(), path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise PyTorchModelException(f'File {path} does not exist. Not loading model')
        self._model.load_state_dict(torch.load(path))


class ModelTensorDefinition(Model, ABC):
    def __init__(self, tensor_instance: TensorInstanceNumpy):
        Model.__init__(self)
        self._val_td_is_inference_ready(tensor_instance.target_tensor_def)
        self._tensor_definition: Tuple[TensorDefinition, ...] = tensor_instance.target_tensor_def
        self._x_indexes = tuple(
            [i for i, _ in enumerate(self._tensor_definition) if i not in tensor_instance.label_indexes]
        )
        self._x_ranks = tuple([self._tensor_definition[i].rank for i in self._x_indexes])

    @property
    def tensor_definitions(self) -> Tuple[TensorDefinition, ...]:
        """
        Property that return the TensorDefinitions this model was built on.

        Returns:
            Tuple of TensorDefinitions that were used to construct the model.
        """
        return self._tensor_definition

    def create_heads(self, dim_ratio: float = 0.5, min_dims: int = 5,
                     max_dims: int = 50, dropout: float = 0.1) -> nn.ModuleList:
        """
        Method from the ModelTensorDefinition class, it will create the heads based on the information provided in
        the input 'TensorInstanceNumpy'.

        Args:
            dim_ratio: This ratio defines the number of dimensions the embedding will have. The number of unique values
                is multiplied by the ratio to get the embedding dimension size. For instance with a ration of 0.5, and
                8 unique values, the embedding will have 8 * 0.5 or 4 dimension.
            min_dims: The minimum number of dimensions to allocate. If the number 'unique_values * dim_ratio' is
                smaller than min_dims, then the 'min_dims' number of dimensions will be allocated.
            max_dims: The maximum number of dimensions to allocate. If the number 'unique_values * dim_ratio' is
                bigger than max_dims, then the 'max_dims' number of dimensions will be allocated.
            dropout: The dropout factor to apply to the output of the embeddings.

        Returns:
            An 'nn.ModuleList' containing a 'TensorDefinition' Layer for each of the TensorDefinitions found in the
                'TensorInstance' used to create the model.

        """
        heads = [TensorDefinitionHead(td, dim_ratio, min_dims, max_dims, dropout)
                 for i, td in enumerate(self._tensor_definition) if i in self._x_indexes]
        return nn.ModuleList(heads)

    @property
    def x_indexes(self) -> Tuple[int, ...]:
        return self._x_indexes

    def get_x(self, ds: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        # A bit of logic to make sure that we always return tensors of the same rank. The batch dimension can get lost
        # if you select a single entry.
        ds = tuple(ds[xi] for xi in self._x_indexes)
        return ds

    def forward_captum(*args):
        """
        A Specific forward function for Captum. Captum wants a tuple of tensors as input, and it unpacks them before
        handing them to the forward of the model. That is not compatible with the standard Tuple[torch.Tensor, ...]
        m0d3l uses. This function takes a variable number of torch.Tensor as input packs them and forwards to the
        'self.forward'

        Args:
             A variable number of torch.Tensors.

        Returns:
             A single tensor. (Not a tuple like the standard forward).
        """
        self = args[0]
        o = self.forward(tuple(a for a in args[1:]))
        return o[0]

    @staticmethod
    def _val_td_is_inference_ready(tensor_definition: Tuple[TensorDefinition, ...]):
        for td in tensor_definition:
            if not td.inference_ready:
                raise PyTorchModelException(
                    f'Tensor Definition {td.name} is not read for inference. A Tensor Definition can be made ready ' +
                    f'for inference by using an engine method with the "inference=False" set'
                )
