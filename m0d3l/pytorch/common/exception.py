"""
Definition of exceptions
(c) 2023 tsm
"""


class PyTorchTrainException(Exception):
    """Standard exception raised during training"""
    def __init__(self, message: str):
        super().__init__('Error in PyTorch Training: ' + message)


class PyTorchLayerException(Exception):
    """Standard exception raised in Layer Classes"""
    def __init__(self, message: str):
        super().__init__('Error in Layer: ' + message)


class PyTorchModelException(Exception):
    """Standard exception raised in Model construction"""
    def __init__(self, message: str):
        super().__init__('Error in Model: ' + message)
