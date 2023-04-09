"""
Common class to wrap attribution results from Captum.
(c) 2023 tsm
"""
from abc import ABC, abstractmethod

import numpy as np

from f3atur3s import TensorDefinition

from typing import Tuple

class AttributionResults(ABC):
    def __init__(self, tensor_definitions: Tuple[TensorDefinition, ...], original_data: Tuple[np.ndarray, ...],
                 x_indexes: Tuple[int, ...]):
        self._tensor_definitions = tensor_definitions
        self._original_data = original_data
        self._original_x_indexes = x_indexes

    @property
    def tensor_definition(self) -> Tuple[TensorDefinition, ...]:
        return self._tensor_definitions

    @property
    @abstractmethod
    def attributions(self) -> Tuple[np.ndarray]:
        pass

    @property
    def original_data(self) -> Tuple[np.ndarray]:
        return self._original_data

    @property
    def original_x_indexes(self) -> Tuple[int, ...]:
        return self._original_x_indexes

    @property
    @abstractmethod
    def __len__(self):
        pass


class AttributionResultBinary(AttributionResults):
    def __init__(self, tensor_definitions: Tuple[TensorDefinition, ...],
                 original_data: Tuple[np.ndarray, ...], original_x_indexes: Tuple[int, ...], label_index: int,
                 classification_labels: np.ndarray, attr: Tuple[np.ndarray, ...]):
        super(AttributionResultBinary, self).__init__(tensor_definitions, original_data, original_x_indexes)
        self._attr = attr
        self._classification_labels = classification_labels
        self._label_index = label_index

    @property
    def attributions(self) -> Tuple[np.ndarray]:
        return self._attr

    @property
    def original_label_index(self) -> int:
        return self._label_index

    @property
    def classification_labels(self) -> np.ndarray:
        return self._classification_labels

    def __len__(self):
        return self._attr[0].shape[0]
