"""
Module for Encoder Models
(c) 2023 tsm
"""

from abc import ABC

import torch
import torch.nn as nn

from f3atur3s import LEARNING_CATEGORY_CATEGORICAL, LEARNING_CATEGORY_BINARY, LearningCategory

from .base import ModelTensorDefinition
from ..layers.autoencoders import LinearEncoder, LinearDecoder, LinearVAEEncoder, LinearVAEDecoder
from ..common.exception import PyTorchModelException
from ..training.history import TunableHistory, History
from ...common.modelconfig import ModelConfiguration, TensorConfiguration
from ..training.loss import MultiLabelNLLLoss, SingleLabelBCELoss, SingleLabelVAELoss, Loss

from typing import Tuple, Dict

class AutoEncoderBase(ModelTensorDefinition, ABC):
    def __init__(self, model_configuration: ModelConfiguration):
        ModelTensorDefinition.__init__(self, model_configuration)
        self._lc = model_configuration.tensor_configurations[AutoEncoder.get_x_index(self)].unique_learning_category
        # Set-up label index. We only need this for testing. During testing, we might want know if it was fraud or not.
        self._y_index = self.get_label_index(model_configuration)

    def get_y(self, ds: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        # For encoders the in and output are the same
        return self.get_x(ds)

    def history(self, batch_size: int, sample_number: int, step_number: int) -> History:
        return AutoEncoderHistory(batch_size, sample_number, step_number)

    @property
    def learning_category(self) -> LearningCategory:
        return self._lc

    @property
    def label_index(self) -> int:
        return self._y_index

    @staticmethod
    def get_label_index(model_configuration: ModelConfiguration) -> int:
        if len(model_configuration.label_indexes) > 1:
            raise PyTorchModelException(
                f'An AutoEncoder is designed to have one label only. Got {model_configuration.label_indexes}'
            )
        return model_configuration.label_indexes[0]

    @staticmethod
    def get_x_index(model: ModelTensorDefinition) -> int:
        if len(model.x_indexes) > 1:
            raise PyTorchModelException(
                f'An AutoEncoder is designed to only have one x Tensor (Tensor with data). The input model has '
            )
        return model.x_indexes[0]

    @staticmethod
    def get_tensor_definition(model_configuration: ModelConfiguration) -> TensorConfiguration:
        di = model_configuration.data_indexes
        if len(di) > 1:
            raise PyTorchModelException(
                f'An auto-encoder can only have one data tensor definition. Found data indexes {di}'
            )
        else:
            return model_configuration.tensor_configurations[di[0]]

class AutoEncoder(AutoEncoderBase, ABC):
    def create_linear_encoder(self, layer_sizes: Tuple[int, ...], dropout: float = 0.0,
                              bn_interval: int = 0) -> nn.Module:
        tc = AutoEncoder.get_tensor_definition(self.model_configuration)
        return LinearEncoder(tc, layer_sizes, dropout, bn_interval)

    def create_linear_decoder(self, layer_sizes: Tuple[int, ...], dropout: float = 0.0,
                              bn_interval: int = 0) -> nn.Module:
        tc = AutoEncoder.get_tensor_definition(self.model_configuration)
        return LinearDecoder(tc, layer_sizes, dropout, bn_interval)

    @property
    def loss_fn(self) -> Loss:
        if self.learning_category == LEARNING_CATEGORY_BINARY:
            return SingleLabelBCELoss()
        elif self.learning_category == LEARNING_CATEGORY_CATEGORICAL:
            return MultiLabelNLLLoss()
        else:
            raise PyTorchModelException(
                f'Can not determine what the loss should be for LC {self.learning_category}'
            )

class VariationalAutoEncoder(AutoEncoderBase, ABC):

    def create_linear_encoder(self, layer_sizes: Tuple[int, ...], dropout: float = 0.0,
                              bn_interval: int = 0) -> nn.Module:
        tc = AutoEncoder.get_tensor_definition(self.model_configuration)
        return LinearVAEEncoder(tc, layer_sizes, dropout, bn_interval)

    def create_linear_decoder(self, layer_sizes: Tuple[int, ...], dropout: float = 0.0,
                              bn_interval: int = 0) -> nn.Module:
        tc = AutoEncoder.get_tensor_definition(self.model_configuration)
        return LinearVAEDecoder(tc, layer_sizes, dropout, bn_interval)

    @property
    def loss_fn(self) -> Loss:
        if self.learning_category == LEARNING_CATEGORY_BINARY:
            return SingleLabelVAELoss()
        elif self.learning_category == LEARNING_CATEGORY_CATEGORICAL:
            return MultiLabelNLLLoss()
        else:
            raise PyTorchModelException(
                f'Can not determine what the loss should be for LC {self.learning_category}'
            )

class AutoEncoderHistory(TunableHistory):
    loss_key = 'loss'

    def __init__(self, batch_size: int, sample_number: int, step_number: int):
        super(AutoEncoderHistory, self).__init__(batch_size, sample_number, step_number)
        self._history = {k: [] for k in [AutoEncoderHistory.loss_key]}
        self._running_loss = 0

    @property
    def history(self) -> Dict:
        return self._history

    def end_step(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...], loss: torch.Tensor):
        self._running_loss += loss.item()

    def end_epoch(self):
        self._history[AutoEncoderHistory.loss_key].append(round(self._running_loss/self.number_of_steps, 4))
        self._running_loss = 0
        super(AutoEncoderHistory, self).end_epoch()

    def early_break(self) -> bool:
        return False

    @property
    def tune_stats(self) -> Dict[str, float]:
        return {self.loss_key: self._history[self.loss_key][-1]}
