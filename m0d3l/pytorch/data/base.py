"""
Base class to create Pytorch data-sets
(c) 2023 tsm
"""
from abc import ABC, abstractmethod
import torch
import torch.utils.data as data

from f3atur3s import TensorDefinition, LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CATEGORICAL
from f3atur3s import LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_LABEL

from eng1n3.pandas import TensorInstanceNumpy

from ..common.exception import PyTorchTrainException

from typing import List, Tuple


DEFAULT_TYPES_PER_LEARNING_CATEGORY = {
    LEARNING_CATEGORY_CONTINUOUS: torch.float32,
    LEARNING_CATEGORY_BINARY: torch.float32,
    LEARNING_CATEGORY_CATEGORICAL: torch.long,
    LEARNING_CATEGORY_LABEL: torch.float32
}


class ModelDataSet(ABC, data.Dataset):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item: int) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def data_loader(self, device: torch.device, batch_size: int, num_workers: int = 1,
                    shuffle: bool = False, sampler: data.Sampler = None) -> data.DataLoader:
        """
        Create a Pytorch Data-loader for the underlying Data-set.

        Args:
            device: The Pytorch device on which to create the data. Either CPU or GPU. Note that if the device is
                set to GPU only one worker can be used.
            batch_size: The batch size for the Data-loader.
            num_workers: Number of workers to use in the Data-loader. Default = 1. If more than one worker is
                defined the device will default to 'cpu' because 'cuda' devices do not support multiple workers.
            shuffle: Flag to trigger random shuffling of the dataset. Default = False
            sampler: Sampler to use. Optional. Needs to be an instance of a Sampler (from the Pytorch library).

        Returns:
            A Pytorch data-loader for this data-set. Ready to train.
        """
        pass

    @property
    @abstractmethod
    def tensor_definitions(self) -> Tuple[TensorDefinition, ...]:
        pass

    @staticmethod
    def get_dtypes(ti: TensorInstanceNumpy) -> Tuple[torch.dtype, ...]:
        dtypes = []
        for lc in ti.learning_categories:
            d_type = DEFAULT_TYPES_PER_LEARNING_CATEGORY.get(lc, None)
            if d_type is None:
                PyTorchTrainException(
                    f'Could not determine default Torch tensor data type for learning category <{lc}>'
                )
            else:
                dtypes.append(d_type)
        return tuple(dtypes)

