"""
Definitions of some common objects we will use
(c) 2023 tsm
"""
import logging

import torch
import torch.utils.data as data

from abc import ABC, abstractmethod
from typing import Dict, List

from ..common.exception import PyTorchTrainException

logger = logging.getLogger(__name__)


class History(ABC):
    """ Base Class for all history objects, this is where we track training statistics.
    """
    def __init__(self, dl: data.DataLoader, history: Dict[str, List]):
        self._batch_size = dl.batch_size
        # noinspection PyTypeChecker
        self._samples = len(dl.dataset)
        self._step = 0
        self._steps = len(dl)
        self._epoch = 0
        self._history = history

    def _val_argument(self, args) -> data.DataLoader:
        if not isinstance(args[0], data.DataLoader):
            raise PyTorchTrainException(
                f'Argument during creation of {self.__class__.__name__} should have been a data loader. ' +
                f'Was {type(args[0])}'
            )
        else:
            return args[0]

    @staticmethod
    def _val_is_tensor(arg):
        if not isinstance(arg, torch.Tensor):
            raise PyTorchTrainException(
                f'Expected this argument to be a Tensor. Got {type(arg)}'
            )

    @staticmethod
    def _val_is_tensor_list(arg):
        if not isinstance(arg, List):
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
    def samples(self) -> int:
        return self._samples

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def history(self) -> Dict:
        return self._history

    @property
    def epoch(self):
        return self._epoch

    @property
    def step(self) -> int:
        return self._step

    def start_step(self):
        self._step += 1

    @abstractmethod
    def end_step(self, *args):
        pass

    def early_break(self) -> bool:
        pass

    def start_epoch(self):
        self._step = 0
        self._epoch += 1

    @abstractmethod
    def end_epoch(self):
        pass
