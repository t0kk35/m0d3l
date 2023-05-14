"""
Module for classifier Models
(c) 2023 tsm
"""

import logging
import torch
import torch.nn as nn
from abc import ABC

from .base import ModelTensorDefinition
from ..common.exception import PyTorchModelException
from ..training.history import TunableHistory
from ...common.modelconfig import ModelConfiguration

from ..training.loss import SingleLabelBCELoss, Loss

from collections import OrderedDict
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class BinaryClassifier(ModelTensorDefinition, ABC):
    def __init__(self, model_configuration: ModelConfiguration):
        ModelTensorDefinition.__init__(self, model_configuration)
        self._y_index = self.get_label_index(model_configuration)

    @staticmethod
    def create_tail(input_size: int) -> nn.Module:
        ls = OrderedDict()
        ls.update({'tail_lin': nn.Linear(input_size, 1)})
        ls.update({'tail_sig': nn.Sigmoid()})
        return nn.Sequential(ls)

    def get_y(self, ds: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return ds[self._y_index: self._y_index+1]

    @property
    def label_index(self) -> int:
        return self._y_index

    @property
    def loss_fn(self) -> Loss:
        return SingleLabelBCELoss()

    def history(self, batch_size: int, sample_number: int, step_number: int) -> TunableHistory:
        return ClassifierHistory(batch_size, sample_number, step_number)

    @staticmethod
    def get_label_index(model_configuration: ModelConfiguration) -> int:
        if len(model_configuration.label_indexes) > 1:
            raise PyTorchModelException(
                f'A Binary Classifier is designed to have one label only. Got {model_configuration.label_indexes}'
            )
        return model_configuration.label_indexes[0]

class ClassifierHistory(TunableHistory):
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

    @property
    def tune_stats(self) -> Dict[str, float]:
        return {self.loss_key: self._history[self.loss_key][-1]}
