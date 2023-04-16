"""
Module for classifier Models
(c) 2023 tsm
"""

import logging
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from eng1n3.pandas import TensorInstanceNumpy

from .base import ModelTensorDefinition
from ..common.exception import PyTorchModelException
from ..training.history import History

from ..training.loss import SingleLabelBCELoss, Loss

from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class BinaryClassifier(ModelTensorDefinition, ABC):
    def __init__(self, tensor_instance: TensorInstanceNumpy):
        ModelTensorDefinition.__init__(self, tensor_instance)
        self._y_index = self.get_label_index(tensor_instance)

    @staticmethod
    def create_tail() -> nn.Module:
        return nn.Sigmoid()

    def get_y(self, ds: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return ds[self._y_index: self._y_index+1]

    @property
    def label_index(self) -> int:
        return self._y_index

    @property
    def loss_fn(self) -> Loss:
        return SingleLabelBCELoss()

    def history(self, batch_size: int, sample_number: int, step_number: int) -> History:
        return ClassifierHistory(batch_size, sample_number, step_number)

    @staticmethod
    def get_label_index(ti: TensorInstanceNumpy) -> int:
        if len(ti.label_indexes) > 1:
            raise PyTorchModelException(
                f'A Binary Classifier is designed to have one label only. Got {ti.label_indexes}'
            )
        return ti.label_indexes[0]

class ClassifierHistory(History):
    loss_key = 'loss'
    acc_key = 'acc'
    auc_key = 'auc'

    def __init__(self, batch_size: int, sample_number: int, step_number: int):
        super(ClassifierHistory, self).__init__(batch_size, sample_number, step_number)
        self._history = {k: [] for k in [ClassifierHistory.loss_key, ClassifierHistory.acc_key]}
        self._running_loss = 0
        self._running_correct_cnt = 0
        self._running_count = 0

    @property
    def history(self) -> Dict:
        return self._history

    def end_step(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...], loss: torch.Tensor):
        pr, lb = y_prd[0], y[0]
        lb = self._reshape_label(pr, lb)
        self._running_loss += loss.item()
        self._running_correct_cnt += torch.sum(torch.eq(torch.ge(pr, 0.5), lb)).item()
        self._running_count += pr.shape[0]

    def early_break(self) -> bool:
        return False

    def end_epoch(self):
        self._history[self.loss_key].append(round(self._running_loss/self.number_of_steps, 4))
        self._history[self.acc_key].append(round(self._running_correct_cnt/self.number_of_samples, 4))
        # Reset for next epoch.
        self._running_correct_cnt = 0
        self._running_count = 0
        self._running_loss = 0

    @staticmethod
    def _reshape_label(pr: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
        if pr.shape == lb.shape:
            return lb
        elif len(pr.shape)-1 == len(lb.shape) and pr.shape[-1] == 1:
            return torch.unsqueeze(lb, dim=len(pr.shape)-1)
        else:
            raise PyTorchModelException(
                f'Incompatible shapes for prediction and label. Got {pr.shape} and {lb.shape}. Can not safely compare'
            )
