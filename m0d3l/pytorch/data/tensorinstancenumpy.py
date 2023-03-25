"""
Classes that create a Pytorch data loader from a 'TensorInstanceNumpy' class
(c) 2023 tsm
"""
import logging
import torch
import torch.utils.data as data

from f3atur3s import TensorDefinition
from eng1n3.pandas import TensorInstanceNumpy

from .base import ModelDataSet
from ..common.exception import PyTorchTrainException
from typing import List, Tuple

logger = logging.getLogger(__name__)


class TensorInstanceNumpyDataSet(ModelDataSet):
    def __init__(self, ti: TensorInstanceNumpy):
        self._ti = ti
        self._dtypes = self.get_dtypes(ti)
        # Yes assign to CPU. We could directly allocate to the GPU, but then we can only use one worker :|
        self.device = torch.device('cpu')

    def __len__(self):
        return len(self._ti)

    def __getitem__(self, item: int) -> List[torch.Tensor]:
        res = [torch.as_tensor(array[item], dtype=dt, device=self.device)
               for array, dt in zip(self._ti.numpy_lists, self._dtypes)]
        return res

    def tensor_definitions(self) -> Tuple[TensorDefinition, ...]:
        return self._ti.target_tensor_def

    def data_loader(self, device: torch.device, batch_size: int, num_workers: int = 1,
                    shuffle: bool = False, sampler: data.Sampler = None) -> data.DataLoader:
        # Cuda does not support multiple workers. Override if GPU
        if num_workers > 1:
            if self.device.type == 'cuda':
                logger.warning(f'Defaulted to using the cpu for the data-loader of '
                               f'<{[td.name for td in self.tensor_definitions()]}>.' +
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

    Args:
         ti: The TensorInstanceNumpy object ot Sample.
    """
    def __init__(self, ti: TensorInstanceNumpy):
        self._ti = ti

    def over_sampler(self, replacement=True) -> data.Sampler:
        """
        Create a RandomWeightedSampler that balances out the classes. It'll more or less return an equal amount of
        each class. For a binary fraud label this would mean about as much fraud as non-fraud samples.

        Args:
            replacement: Bool flag to trigger sample with replacement. With replacement a row can be drawn more
                than once

        Returns:
            A Pytorch Sampler
        """
        label_index = self._ti.label_indexes
        if len(label_index) > 1:
            raise PyTorchTrainException(
                f'The Class Sampler has only been implemented to have one label only. The TensorInstance had ' +
                f'{len(label_index)}'
            )
        else:
            label_index = label_index[0]
        _, class_balance = self._ti.unique(label_index)
        weights = 1./torch.tensor(class_balance, dtype=torch.float)
        sample_weights = weights[torch.as_tensor(self._ti.numpy_lists[label_index].astype(int))]
        sample_weights = torch.squeeze(sample_weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self._ti),
            replacement=replacement
        )
        return train_sampler
