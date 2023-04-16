"""
Definition of exceptions
(c) 2023 tsm
"""
class PlotException(Exception):
    """Standard exception raised during plotting of data"""
    def __init__(self, message: str):
        super().__init__('Error in PyTorch Training: ' + message)