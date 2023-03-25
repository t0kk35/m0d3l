"""
Common layer class for all custom layers we create in our models
(c) 2023 tsm
"""
import logging
from abc import ABC, abstractmethod

import torch.nn as nn
from typing import Any

from ..common.exception import PyTorchLayerException

logger = logging.getLogger(__name__)


class Layer(nn.Module, ABC):
    def __init__(self):
        nn.Module.__init__(self)

    def _forward_unimplemented(self, *inp: Any) -> None:
        raise NotImplemented(f'Unimplemented <_forward_unimplemented>  in <{self.__class__.__name__}>')

    @property
    @abstractmethod
    def output_size(self) -> int:
        """
        Property that returns the output size of the layer

        Returns:
            Integer value for the size of the layer as it outputs
        """
        pass

    def freeze(self):
        """
        Method to freeze a layer. If a layer is frozen it will no longer be update in during training. Practically
        we do this by setting the parameters to not contribute during the backward gradient pass. In Pytorch this is
        done by setting the requires_grad = False on a parameter. A parameter is basically a Tensor object.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Method to un-freeze a layer. If a layer is un-frozen, which is the standard for parameters, the parameters
        will contribute to the gradient during the backward pass. In Pytorch this is done by setting the
        requires_grad = True. Again, this is the standard state for parameters. If there are parameters that were not
        explicitly frozen, they normally do not need to be unfrozen.
        """
        for param in self.parameters():
            param.requires_grad = True

    # @abstractmethod
    # def copy(self) -> 'Layer':
    #     """
    #     Function to copy a layer. This creates an exact duplicate of the layer. It will set up all parameters the
    #     layer needs and will copy the weights from one layer to the next. This does not reference the original weights
    #     so any updates to the copy will not affect the original layer.
    #
    #     Returns:
    #          The copied layer
    #     """
    #     pass

    def copy_state_dict(self, source_layer: 'Layer') -> None:
        """
        This function will copy the State dict from one model to another. The models have to have the exact same
        keys in the State dict for this operation to work.
        The parameters of the source_layer will be copied into the 'self' layer.

        Args:
             source_layer: The source layer which needs to be copied into the current layer.
        """
        # Get state of current layer
        self_state = self.state_dict()
        # Get state of the target layer
        source_state = source_layer.state_dict()
        # Check both state have the same keys
        if self_state.keys() != source_state.keys():
            raise PyTorchLayerException(
                f'Error copying state dicts. Keys do not match self_keys <{self_state.keys()}>.' +
                f'source_keys <{source_state.keys()}>'
            )
        # Copy keys across
        for sk in self_state.keys():
            self_state[sk].copy_(source_state[sk])
        # Load target Dict in the target layer.
        self.load_state_dict(self_state)
