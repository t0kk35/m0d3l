"""
Definitions of some common objects we will use
(c) 2023 tsm
"""
import logging

import torch

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from ..common.exception import PyTorchTrainException

logger = logging.getLogger(__name__)


class History(ABC):
    """
    Base Class for all history objects, this is where we track training statistics.
    """
    def __init__(self, batch_size: int, sample_number: int, step_number: int):
        self._batch_size = batch_size
        self._sample_number = sample_number
        self._step = 0
        self._step_number = step_number
        self._epoch = 0

    @staticmethod
    def _val_is_tensor(arg):
        if not isinstance(arg, torch.Tensor):
            raise PyTorchTrainException(
                f'Expected this argument to be a Tensor. Got {type(arg)}'
            )

    @staticmethod
    def _val_is_tensor_tuple(arg):
        if not isinstance(arg, Tuple):
            raise PyTorchTrainException(
                f'Expected this argument to be List. Got {type(arg)}'
            )
        if not isinstance(arg[0], torch.Tensor):
            raise PyTorchTrainException(
                f'Expected this arguments list to contain tensors. Got {type(arg[0])}'
            )

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def number_of_samples(self) -> int:
        return self._sample_number

    @property
    def number_of_steps(self) -> int:
        return self._step_number

    @property
    @abstractmethod
    def history(self) -> Dict:
        pass

    @property
    def epoch(self):
        return self._epoch

    @property
    def step(self) -> int:
        return self._step

    def start_step(self):
        self._step += 1

    @abstractmethod
    def end_step(self, y_prd: Tuple[torch.Tensor, ...], y: Tuple[torch.Tensor, ...], loss: torch.Tensor):
        pass

    @abstractmethod
    def early_break(self) -> bool:
        pass

    def start_epoch(self):
        self._step = 0
        self._epoch += 1

    @abstractmethod
    def end_epoch(self):
        pass

class TunableHistory(History, ABC):
    """
    Base Class for histories which we can use during tuning. They need to report loss.
    """
    @property
    @abstractmethod
    def tune_stats(self) -> Dict[str, float]:
        pass
