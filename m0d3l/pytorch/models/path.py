"""
Common classes for all Pytorch Models
(c) 2023 tsm
"""
import logging
import torch.nn as nn
from collections import OrderedDict
from typing import Union, List

from ..common.exception import PyTorchModelException
from ..layers.base import Layer


logger = logging.getLogger(__name__)


class ModelPath:
    """
    Class used in models to set up a set of layers to be executed sequentially. This is not a nn.Module.
    It's just a place-holder class to bundle the layers. By calling the create method,
    a nn.Sequential or Layer will be returned which can be used in models.

    Args:
        name: A name for the Path.
        layer: Optional. This can be used to create a path and immediately add a layer. Default it is None.
    """
    def __init__(self, name: str, layer: Layer = None):
        self.name = name
        self._layers: OrderedDict[str, Layer] = OrderedDict()
        if layer is not None:
            self._layers.update({name: layer})
            self._out_size = layer.output_size
        else:
            self._out_size = -1

    def add(self, name: str, layer: Union[Layer, nn.Module], new_size: int):
        self._layers.update({name: layer})
        self._out_size = new_size

    def create(self) -> Union[nn.Sequential, Layer]:
        if len(self.layers) == 1:
            # There is just one layer, return the first item from the Dict.
            return [v for _, v in self._layers.items()][0]
        else:
            # There is more than one layer. Build a nn.Sequential.
            return nn.Sequential(self._layers)

    @property
    def out_size(self) -> int:
        if self._out_size == -1:
            raise PyTorchModelException(
                f'Outsize has not been set on stream {self.name}. Can not get the out_size'
            )
        return self._out_size

    @property
    def layers(self) -> List[Layer]:
        return [v for _, v in self._layers.items()]
