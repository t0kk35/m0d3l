"""
Imports for Pytorch Loss functions
(c) 2023 tsm
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss as TorchLoss

from typing import Type, Any, List, Tuple


class Loss(ABC):
    """
    Loss function object provided to the model during training """
    def __init__(self, loss_fn: Type[TorchLoss], reduction: str):
        self._training_loss = loss_fn(reduction=reduction)
        self._score_loss = loss_fn(reduction='none')
        self._aggregator = torch.sum if reduction == 'sum' else torch.mean

    @abstractmethod
    def __call__(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def score(self, *args, **kwargs) -> torch.Tensor:
        pass

    def __repr__(self):
        return f'{self.__class__.__name__},  {str(self._training_loss.reduction)}'

    @property
    def train_loss(self) -> TorchLoss:
        return self._training_loss

    @property
    def score_loss(self) -> TorchLoss:
        return self._score_loss

    @property
    def score_aggregator(self) -> Any:
        return self._aggregator


class SingleLabelBCELoss(Loss):
    def __init__(self, reduction='mean'):
        super(SingleLabelBCELoss, self).__init__(nn.BCELoss, reduction)

    def __call__(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pr = torch.squeeze(y_prd[0])
        lb = torch.squeeze(y[0])
        loss = self.train_loss(pr, lb)
        return loss

    def score(self, *args, **kwargs) -> torch.Tensor:
        pr = torch.squeeze(args[0])
        lb = torch.squeeze(args[1][0])
        loss = self.score_loss(pr, lb)
        # Aggregate over all but batch dimension.
        return self.score_aggregator(loss, dim=list(range(1, len(pr.shape))))
