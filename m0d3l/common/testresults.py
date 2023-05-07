"""
Common class for test results
(c) 2023 tsm
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

class TestResults(ABC):
    @property
    @abstractmethod
    def y_prd(self) -> Tuple[np.ndarray, ...]:
        pass

    @property
    @abstractmethod
    def y(self) -> Tuple[np.ndarray, ...]:
        pass

class TestResultsBinary(TestResults):
    def __init__(self, y_prd: np.ndarray, y: np.ndarray):
        self._y_prd = y_prd
        self._y = y

    @property
    def y_prd(self) -> Tuple[np.ndarray, ...]:
        return (self._y_prd,)

    @property
    def y(self) -> Tuple[np.ndarray, ...]:
        return (self._y,)


class TestResultsLoss(TestResultsBinary):
    def __init__(self, y_prd: np.ndarray, y: np.ndarray, loss: np.ndarray):
        super(TestResultsLoss, self).__init__(y_prd, y)
        self._loss = loss

    @property
    def loss(self) -> np.ndarray:
        return self._loss
