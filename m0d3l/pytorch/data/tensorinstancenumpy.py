"""
Classes that create a Pytorch data loader from a 'TensorInstanceNumpy' class
(c) 2023 tsm
"""
import logging
import torch
import torch.utils.data as data

from eng1n3.pandas import TensorInstanceNumpy

from .base import ModelDataSet
from ..common.exception import PyTorchTrainException
from typing import Tuple

logger = logging.getLogger(__name__)


class TensorInstanceNumpyDataSet(ModelDataSet):
    def __init__(self, ti: TensorInstanceNumpy):
        self._len = len(ti)
        self._dtypes = self.get_dtypes(ti)
        self._tensors = tuple(
            torch.as_tensor(array, dtype=dt) for array, dt in zip(ti.numpy_lists, self.get_dtypes(ti))
        )
        # Yes assign to CPU. We could directly allocate to the GPU, but then we can only use one worker :|
        self.device = torch.device('cpu')

    def __len__(self):
        return self._len

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        return tuple(t[item] for t in self._tensors)

    @property
    def tensors(self) -> Tuple[torch.Tensor, ...]:
        return self._tensors

    def data_loader(self, device: torch.device, batch_size: int, num_workers: int = 1,
                    shuffle: bool = False, sampler: data.Sampler = None) -> data.DataLoader:
        # Cuda does not support multiple workers. Override if GPU
        if num_workers > 1:
            if self.device.type == 'cuda':
                logger.warning(f'Defaulted to using the cpu for the data-loader ' +
                               f' Multiple workers not supported by "cuda" devices. ')
            self.device = torch.device('cpu')
            dl = data.DataLoader(
                self, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, pin_memory=True, sampler=sampler
            )
        else:
            # Only CPU Tensors can be pinned
            pin = False if device.type == 'cuda' else True
            self.device = device
            dl = data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, pin_memory=pin, sampler=sampler)
        return dl


class TensorInstanceNumpyLabelSampler:
    """
    Class for creating a sampler from a TensorInstanceNumpy
    """

    @classmethod
    def over_sampler(cls, ti: TensorInstanceNumpy, replacement=True,
                     target_balance: Tuple[float, ...] = None) -> data.Sampler:
        """
        Create a RandomWeightedSampler that balances out the classes. It'll by default more or less return an equal amount of
        each class. For a binary fraud label this would mean about as much fraud as non-fraud samples.

        Args:
            ti: The TensorInstanceNumpy on which to create the sampler.
            replacement: Bool flag to trigger sample with replacement. With replacement a row can be drawn more
                than once
            target_balance: Tuple of floats listing the target balance for each class. Can be used to override the
                standard equal target balance. For instance (.4,.6) will have approx. 40% of class 0 and 60% of class 1.
                The sum of balances should be 1.0 and there should be the same number of balances as there are actual
                classes in the label.

        Returns:
            A Pytorch Sampler
        """
        label_index = ti.label_indexes
        if len(label_index) > 1:
            raise PyTorchTrainException(
                f'The Class Sampler has only been implemented to have one label only. The TensorInstance had ' +
                f'{len(label_index)}'
            )
        else:
            label_index = label_index[0]
        if sum([t for t in target_balance]) is not 1.0:
            raise PyTorchTrainException(
                f'The sum of the target_balance parameter should be 1.0. It is {sum([t for t in target_balance])}'
            )
        _, class_balance = ti.unique(label_index)
        if target_balance is not None and len(class_balance) != len(target_balance):
            raise PyTorchTrainException(
                f'The number of target_balances should be the same as the number of classes in the label. There are ' +
                f'{len(class_balance)} classes and {len(target_balance)} balances'
            )
        if target_balance is None:
            balances = torch.ones([len(class_balance)], dtype=torch.float)
        else:
            balances = torch.as_tensor(target_balance)
        weights = balances*1./torch.tensor(class_balance, dtype=torch.float)
        sample_weights = weights[torch.as_tensor(ti.numpy_lists[label_index].astype(int))]
        sample_weights = torch.squeeze(sample_weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(ti),
            replacement=replacement
        )
        return train_sampler
