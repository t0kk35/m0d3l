"""
Class that contains a model configuration.
We've split this out because we had challenges building nn.Modules that contain complex classes such as Feature,
TensorDefinition and TensorInstance. They can prevent the nn.Modules from being 'pickle-able'.

So we're making a class that only contains 'native' objects like strings and ints.

(c) 2023 TSM
"""

from f3atur3s import LearningCategory, FeatureLabel, FeatureCategorical
from f3atur3s import TensorDefinition

from typing import Tuple

class TensorConfiguration:
    def __init__(self, name: str, learning_categories: Tuple[Tuple[LearningCategory, int], ...],
                 categorical_features: Tuple[Tuple[str, int], ...], rank: int):
        self._name = name
        self._learning_categories = learning_categories
        self._categorical_features = categorical_features
        self._rank = rank

    @property
    def name(self) -> str:
        return self._name

    @property
    def categorical_features(self) -> Tuple[Tuple[str, int], ...]:
        return self._categorical_features

    @property
    def learning_categories(self) -> Tuple[Tuple[LearningCategory, int], ...]:
        return self._learning_categories

    @property
    def rank(self) -> int:
        return self._rank

class ModelConfiguration:
    def __init__(self, tensor_configuration: Tuple[TensorConfiguration, ...], label_indexes: Tuple[int, ...]):
        self._tensor_configurations = tensor_configuration
        self._label_indexes = label_indexes

    @property
    def tensor_configurations(self) -> Tuple[TensorConfiguration, ...]:
        return self._tensor_configurations

    @property
    def label_indexes(self) -> Tuple[int, ...]:
        return self._label_indexes

    @classmethod
    def from_tensor_definitions(cls, tensor_definitions: Tuple[TensorDefinition, ...]) -> 'ModelConfiguration':
        tcs = []
        for td in tensor_definitions:
            n = td.name
            lc = tuple([(lc, len(td.filter_features(lc, True))) for lc in td.learning_categories])
            r = td.rank
            cf = tuple([(f.name, len(f)) for f in td.categorical_features() if isinstance(f, FeatureCategorical)])
            tcs.append(TensorConfiguration(n, lc, cf, r))
        li = ModelConfiguration._find_label_indexes(tensor_definitions)
        return ModelConfiguration(tuple(tcs), li)

    @staticmethod
    def _find_label_indexes(target_tensor_def: Tuple[TensorDefinition, ...]) -> Tuple[int, ...]:
        """
        Local method that tries to find out what indexes are the labels in the data set (if any). Label
        TensorDefinitions are TensorDefinitions that only contain feature of type 'FeatureLabel'.

        Args:
             target_tensor_def: A tuple of TensorDefinitions to check

        Returns:
            A tuple of ints of (potentially empty) that are the indexes of the lists that hold the labels.
        """
        ind = [i for i, td in enumerate(target_tensor_def) if all([isinstance(f, FeatureLabel) for f in td.features])]
        return tuple(ind)
