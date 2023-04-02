"""
Module for custom optimizers
(c) 2020 tsm
"""
import torch.optim as opt
import torch.nn as nn

from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def zero_grad(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @property
    @abstractmethod
    def optimizer(self) -> opt.Optimizer:
        pass

    @property
    @abstractmethod
    def lr(self) -> float:
        pass


class AdamWOptimizer(Optimizer):
    def __init__(self, model: nn.Module, lr: float, wd: float = 1e-2):
        lr = lr if lr is not None else 1e-3
        wd = wd if wd is not None else 1e-2
        self._opt = opt.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def zero_grad(self):
        self._opt.zero_grad()

    def step(self):
        self._opt.step()

    @property
    def optimizer(self) -> opt.Optimizer:
        return self._opt

    @property
    def lr(self) -> float:
        return self._opt.param_groups[0]['lr']

    def __repr__(self):
        return f'AdamW Optimizer with learning rate {self.lr}'
