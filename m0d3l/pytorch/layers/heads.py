"""
Module for common layers at the beginning of a NN
(c) 2023 tsm
"""
import torch
import torch.nn as nn

from f3atur3s import TensorDefinition, FeatureCategorical, LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CONTINUOUS
from f3atur3s import LEARNING_CATEGORY_CATEGORICAL, LearningCategory

from ...common.modelconfig import TensorConfiguration
from ..layers.base import Layer
from ..common.exception import PyTorchLayerException


class Embedding(Layer):
    """
    Layer that creates a set of torch embedding layers. One for each 'FeatureIndex' more specifically.
    The embeddings will be concatenated in the forward operation. So this will take a tensor of 'torch.long', run each
    through a torch embedding layer, concatenate the output, apply dropout and return.

    Args:
        tensor_configuration: A TensorConfiguration describing the input. Each FeatureIndex in this definition will be
            turned into an embedding layer
        dim_ratio: The ratio by which the size is multiplies to determine the embedding dimension.
        min_dims: The minimum dimension of an embedding.
        max_dims: The maximum dimension of an embedding.
        dropout: A float number that determines the dropout amount to apply. The dropout will be applied to the
            concatenated output layer
    """
    def __init__(self, tensor_configuration: TensorConfiguration, dim_ratio: float, min_dims: int,
                 max_dims: int, dropout: float):
        super(Embedding, self).__init__()
        self._i_features = tuple([n for n, _ in tensor_configuration.categorical_features])
        emb_dim = [
            (l, min(max(int(l*dim_ratio), min_dims), max_dims)) for _, l in tensor_configuration.categorical_features
        ]
        self._out_size = sum((y for _, y in emb_dim))
        self.embeddings = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dim])
        self.dropout = nn.Dropout(dropout)

    @property
    def output_size(self) -> int:
        return self._out_size

    def forward(self, x: torch.Tensor):
        rank = len(x.shape)
        if rank == 2:
            return self.dropout(torch.cat(tuple(emb(x[:, i]) for i, emb in enumerate(self.embeddings)), dim=1))
        elif rank == 3:
            return self.dropout(torch.cat(tuple(emb(x[:, :, i]) for i, emb in enumerate(self.embeddings)), dim=2))
        else:
            raise PyTorchLayerException(f'Don\'t know how to handle embedding with input tensor of rank {rank}')

    def embedding_weight(self, feature: FeatureCategorical) -> torch.Tensor:
        self._val_feature_is_categorical(feature)
        self._val_feature_in_embedding(feature)
        i = self._i_features.index(feature.name)
        w = self.embeddings[i].weight
        return w

    def _val_feature_in_embedding(self, feature: FeatureCategorical):
        if not isinstance(feature, FeatureCategorical):
            raise PyTorchLayerException(
                f'Feature <{feature.name}> is not of type {FeatureCategorical.__class__}. Embedding only work with '
                + f'Index Features'
            )
        if feature.name not in self._i_features:
            raise PyTorchLayerException(
                f'Feature <{feature.name}> is not known to this embedding layer. Please check the model was created ' +
                f'with a Tensor Definition than contains this feature'
            )

    @staticmethod
    def _val_feature_is_categorical(feature: FeatureCategorical):
        if not isinstance(feature, FeatureCategorical):
            raise PyTorchLayerException(
                f'The input to the "embedding_weight" method must be a feature of type "FeatureCategorical". ' +
                f'Got a <{feature.__class__}>'
            )


class TensorDefinitionHead(Layer):

    def __init__(self, tensor_configuration: TensorConfiguration, dim_ratio: float, emb_min_dim: int,
                 emb_max_dim: int, emb_dropout: float):
        # TensorDefinitionHead._val_has_bin_or_con_or_cat_features(tensor_def)
        super(TensorDefinitionHead, self).__init__()
        self._tc_name = tensor_configuration.name
        self._rank = tensor_configuration.rank
        self._lc = self._get_learning_category(tensor_configuration)
        self._captum_mode = False
        if self._lc == LEARNING_CATEGORY_CATEGORICAL:
            self.embedding = Embedding(tensor_configuration, dim_ratio, emb_min_dim, emb_max_dim, emb_dropout)
            self._output_size = self.embedding.output_size
        else:
            self.embedding = None
            self._output_size = sum(n for _, n in tensor_configuration.learning_categories)

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def learning_category(self) -> LearningCategory:
        return self._lc

    @property
    def captum_mode(self) -> bool:
        return self._captum_mode

    @captum_mode.setter
    def captum_mode(self, flag: bool) -> None:
        self._captum_mode = flag

    def extra_repr(self) -> str:
        return f'Name={self._tc_name}, lc={self.learning_category.name}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._lc == LEARNING_CATEGORY_CATEGORICAL:
            if not self._captum_mode:
                # Run embedding for Categorical features
                return self.embedding(x)
            else:
                # If in captum mode, then skip the embedding
                return x
        else:
            # For other Learning categories just forward.
            return x

    @staticmethod
    def _val_has_bin_or_con_or_cat_features(tensor_def: TensorDefinition) -> None:
        """
        Validation routine that check if the learning contains at least one of the following learning categories;
            - LEARNING_CATEGORY_BINARY
            - LEARNING_CATEGORY_CONTINUOUS
            - LEARNING_CATEGORY_CATEGORICAL

        Args:
            tensor_def: The TensorDefinition object of which we are checking

        Return:
            None

        Raises:
            PyTorchLayerException: If none of the listed LearningCategories is present in the input TensorDefinition.
        """
        if not (LEARNING_CATEGORY_BINARY in tensor_def.learning_categories
                or LEARNING_CATEGORY_CONTINUOUS in tensor_def.learning_categories
                or LEARNING_CATEGORY_CATEGORICAL in tensor_def.learning_categories):
            raise PyTorchLayerException(
                f'_StandardHead needs features of Learning category "Binary" or "Continuous" or "Categorical. '
                f'Tensor definition <{tensor_def.name} has none of these.'
            )

    @staticmethod
    def _get_learning_category(tensor_configuration: TensorConfiguration) -> LearningCategory:
        """
        Small Helper Method to retrieve the learning category. We would expect to find only one per TensorDefinition

        Args:
            tensor_configuration: The TensorConfiguration object of which we are trying to establish the Learning
                category

        Return:
            The learning category of the TensorConfiguration

        Raises:
            PyTorchLayerException: if there was more than on Learning Category in the tensor_configuration input
            parameter.
        """
        lcs = tuple(set(lc for lc, _ in tensor_configuration.learning_categories))

        if len(lcs) > 1:
            raise PyTorchLayerException(
                f'Expecting only one LearningCategory per each TensorDefinition. ' +
                f'Found {lcs} in TensorDefinition {tensor_configuration.name}'
            )
        else:
            return tensor_configuration.learning_categories[0][0]
