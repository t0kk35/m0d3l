"""
Module for classifier Models
(c) 2023 tsm
"""

import logging
import torch
import torch.nn as nn

from f3atur3s import TensorDefinition
from eng1n3.pandas import TensorInstanceNumpy

from .base import Model, ModelTensorDefinition
from ..common.exception import PyTorchModelException
from ..training.history import History

from ..training.loss import SingleLabelBCELoss, Loss

from typing import List, Dict, Union, Tuple

from ..training.optimizer import Optimizer

logger = logging.getLogger(__name__)


class BinaryClassifier(ModelTensorDefinition):
    def __init__(self, tensor_instance: TensorInstanceNumpy):
        super(BinaryClassifier, self).__init__(tensor_instance)
        self._y_index = self.get_label_index(tensor_instance)

    @staticmethod
    def create_tail() -> nn.Module:
        return nn.Sigmoid()

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        return ds[self._y_index: self._y_index+1]

    @property
    def label_index(self) -> int:
        return self._y_index

    @property
    def loss_fn(self) -> Loss:
        return SingleLabelBCELoss()

    def optimizer(self, lr=None, wd=None) -> Optimizer:
        pass

    def history(self, *args) -> History:
        pass

    @staticmethod
    def get_label_index(ti: TensorInstanceNumpy) -> int:
        if len(ti.label_indexes) > 1:
            raise PyTorchModelException(
                f'A Binary Classifier is designed to have one label only. Got {ti.label_indexes}'
            )
        return ti.label_indexes[0]
