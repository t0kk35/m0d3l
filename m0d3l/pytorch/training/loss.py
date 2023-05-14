"""
Imports for Pytorch Loss functions
(c) 2023 tsm
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss as TorchLoss

from typing import Type, Any, Tuple


class Loss(ABC):
    def __init__(self, loss_fn: Type[TorchLoss], reduction: str):
        """
        Loss function object provided to the model during training

        Args:
            loss_fn: The class name of a torch compatible loss function. For instance 'nn.BCELoss'.
            reduction: The reduction logic to apply; can be 'mean', 'sum', 'none'.
        """
        self._training_loss = loss_fn(reduction=reduction)
        self._score_loss = loss_fn(reduction='none')
        self._aggregator = torch.sum if reduction == 'sum' else torch.mean

    @abstractmethod
    def __call__(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    @abstractmethod
    def score(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
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


class Loss1dBase(Loss, ABC):
    def __init__(self, loss_fn: Type[TorchLoss], reduction='mean'):
        super(Loss1dBase, self).__init__(loss_fn, reduction)

    def __call__(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pr = torch.squeeze(y_prd[0])
        lb = torch.squeeze(y[0])
        loss = self.train_loss(pr, lb)
        return loss

class SingleLabelBCELoss(Loss1dBase):
    def __init__(self, reduction='mean'):
        super(SingleLabelBCELoss, self).__init__(nn.BCELoss, reduction)

    def score(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pr = torch.squeeze(y_prd[0])
        lb = torch.squeeze(y[0])
        loss = self.score_loss(pr, lb)
        # Aggregate over all but batch dimension.
        return self.score_aggregator(loss, dim=list(range(1, len(pr.shape))))


class MultiLabelNLLLoss(Loss1dBase):
    """
    MultiLabelNLLLoss Negative Log Likely-hood Loss
    """
    def __init__(self, reduction='mean'):
        super(MultiLabelNLLLoss, self).__init__(nn.NLLLoss, reduction)

    def score(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pr = torch.squeeze(y_prd[0])
        lb = torch.squeeze(y[0])
        score = self.score_loss(pr, lb)
        score = self.score_aggregator(score, dim=list(range(1, len(pr.shape)-1)))
        return score

class MultiLabelNLLLoss2d(Loss):
    """
    MultiLabelNLLLoss Negative Log Likely-hood Loss
    """
    def __init__(self, reduction='mean'):
        super(MultiLabelNLLLoss2d, self).__init__(nn.NLLLoss, reduction)

    def __call__(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pr = torch.squeeze(y_prd[0])
        pr = pr.transpose(1, 2)
        lb = torch.squeeze(y[0])
        loss = self.train_loss(pr, lb)
        return loss

    def score(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pr = torch.squeeze(y_prd[0])
        pr = pr.transpose(1, 2)
        lb = torch.squeeze(y[0])
        score = self.score_loss(pr, lb)
        score = self.score_aggregator(score, dim=list(range(1, len(pr.shape)-1)))
        return score


class SingleLabelVAELoss(Loss):
    """ Variational Auto Encoder Loss
    Args:
        reduction : The reduction to use. One of 'mean', 'sum'. Do not use 'none'.
        kl_weight : A weight to apply to the kl divergence. The kl_divergence will be multiplied by the weight
        before adding to the BCE Loss. Default is 1. That means the full kl_divergence is added. The kl_divergence
        can be given a lower importance with values < 0.
    """
    def __init__(self, reduction='mean', kl_weight=1.0):
        super(SingleLabelVAELoss, self).__init__(nn.BCELoss, reduction)
        self._kl_weight = kl_weight

    def __call__(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # With a VAE the first argument is the predicted label dim, the second is the mu, the third the sigma.
        # See how the LinearVAEDecoder sets the output.
        pr = torch.squeeze(y_prd[0])
        mu = y_prd[1]
        s = y_prd[2]
        lb = torch.squeeze(y[0])
        recon_loss = self.train_loss(pr, lb)
        kl_divergence = -0.5 * torch.sum(1 + s - mu.pow(2) - s.exp())
        return recon_loss + (self._kl_weight * kl_divergence)

    def score(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pr = torch.squeeze(y_prd[0])
        mu = y_prd[1]
        s = y_prd[2]
        lb = torch.squeeze(y[0])
        # BCE Loss
        loss = self.score_loss(pr, lb)
        recon_loss = self.score_aggregator(loss, dim=1)
        # KL Divergence. Do not run over the batch dimension
        kl_divergence = -0.5 * torch.sum(1 + s - mu.pow(2) - s.exp(), tuple(range(1, len(mu.shape))))
        return recon_loss + (self._kl_weight * kl_divergence)

    def __repr__(self):
        return f'SingleLabelVAELoss(), {self._training_loss.reduction}, {self._kl_weight}'

class MultiLabelVAELoss(Loss):
    """ Variational Auto Encoder Loss
    Args:
        reduction : The reduction to use. One of 'mean', 'sum'. Do not use 'none'.
        kl_weight : A weight to apply to the kl divergence. The kl_divergence will be multiplied by the weight
        before adding to the BCE Loss. Default is 1. That means the full kl_divergence is added. The kl_divergence
        can be given a lower importance with values < 0.
    """
    def __init__(self, reduction='mean', kl_weight=1.0):
        super(MultiLabelVAELoss, self).__init__(nn.NLLLoss, reduction)
        self._kl_weight = kl_weight

    def __call__(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # With a VAE the first argument is the predicted label dim, the second is the mu, the third the sigma.
        # See how the LinearVAEDecoder sets the output.
        pr = torch.squeeze(y_prd[0])
        mu = y_prd[1]
        s = y_prd[2]
        lb = torch.squeeze(y[0])
        recon_loss = self.train_loss(pr, lb)
        kl_divergence = -0.5 * torch.sum(1 + s - mu.pow(2) - s.exp())
        return recon_loss + (self._kl_weight * kl_divergence)

    def score(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pr = torch.squeeze(y_prd[0])
        mu = y_prd[1]
        s = y_prd[2]
        lb = torch.squeeze(y[0])
        # BCE Loss
        loss = self.score_loss(pr, lb)
        recon_loss = self.score_aggregator(loss, dim=list(range(1, len(pr.shape)-1)))
        # KL Divergence. Do not run over the batch dimension
        kl_divergence = -0.5 * torch.sum(1 + s - mu.pow(2) - s.exp(), tuple(range(1, len(mu.shape))))
        return recon_loss + (self._kl_weight * kl_divergence)

    def __repr__(self):
        return f'MultiLabelVAELoss(), {self._training_loss.reduction}, {self._kl_weight}'
