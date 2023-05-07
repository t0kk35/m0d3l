"""
Definition of exceptions
(c) 2023 tsm
"""
class PlotException(Exception):
    """Standard exception raised during plotting of data"""
    def __init__(self, message: str):
        super().__init__('Error in PyTorch Training: ' + message)


class ModelConfigException(Exception):
    """Standard exception raised from the ModelConfig Class"""

    def __init__(self, message: str):
        super().__init__('Error in ModelConfig: ' + message)
