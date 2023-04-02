"""
Main class for testing m0d3l
(c) 2023 tsm
"""
import logging
import torch
import torch.utils.data as data
from tqdm import tqdm

from ..models.base import Model
from ...common.testresults import TestResultsBinary

from typing import List

logger = logging.getLogger(__name__)

class Tester:
    """
    Class to test a Neural net. Embeds some methods that hide the Pytorch logic.

    Args:
        model: The model to be tested. This needs to be a m0d3l model. Not a regular nn.Module
        device: A torch device (CPU or GPU) to use during training.
        test_dl: A torch DataLoader object containing the test data
    """
    def __init__(self, model: Model, device: torch.device, test_dl: data.DataLoader):
        self._model = model
        self._device = device
        self._test_dl = test_dl

    @property
    def model(self) -> Model:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @staticmethod
    def _test_step(model: Model, device: torch.device, test_dl: data.DataLoader) -> List[List[torch.Tensor]]:
        model.eval()
        out: List[List[torch.Tensor]] = []
        with torch.no_grad():
            with tqdm(total=len(test_dl), desc=f'Testing in {len(test_dl)} steps') as bar:
                for i, ds in enumerate(test_dl):
                    # All data-sets to the GPU if available
                    ds = tuple(d.to(device, non_blocking=True) for d in ds)
                    x = model.get_x(ds)
                    out.append((model(x)))
                    bar.update(1)
                    del ds
        return out

    def test_results_binary(self) -> TestResultsBinary:
        self.model.to(self.device)
        y_prd = Tester._test_step(self.model, self.device, self._test_dl)
        y_prd = torch.cat([e[0] for e in y_prd], dim=0)
        y_prd = torch.squeeze(y_prd).cpu().numpy()
        y = [self.model.get_y(ds) for ds in iter(self._test_dl)]
        y = torch.squeeze(torch.cat([e[0] for e in y], dim=0)).cpu().numpy()
        return TestResultsBinary(y_prd, y)
