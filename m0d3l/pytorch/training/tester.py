"""
Main class for testing m0d3l
(c) 2023 tsm
"""
import logging
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from .loss import Loss
from ..models.base import Model
from ...common.testresults import TestResultsBinary, TestResultsLoss

from typing import List, Tuple

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

    @staticmethod
    def _loss_step(model: Model, device: torch.device, test_dl: data.DataLoader,
                   loss_fn: Loss) -> List[torch.Tensor]:
        model.eval()
        loss_l: List[torch.Tensor] = []
        # Start loop
        with torch.no_grad():
            with tqdm(total=len(test_dl), desc=f'Calculating loss in {len(test_dl)} steps') as bar:
                for i, ds in enumerate(test_dl):
                    # All data-sets to the GPU if available
                    ds = tuple(d.to(device, non_blocking=True) for d in ds)
                    x = model.get_x(ds)
                    y = model.get_y(ds)
                    y_prd = model(x)
                    s = loss_fn.score(y_prd, y)
                    loss_l.append(s)
                    bar.update(1)
                    del ds
        return loss_l

    def test_results_binary(self) -> TestResultsBinary:
        self.model.to(self.device)
        y_prd = Tester._test_step(self.model, self.device, self._test_dl)
        y_prd = torch.cat([e[0] for e in y_prd], dim=0)
        y_prd = torch.squeeze(y_prd).cpu().numpy()
        y = [self.model.get_y(ds) for ds in iter(self._test_dl)]
        y = torch.squeeze(torch.cat([e[0] for e in y], dim=0)).cpu().numpy()
        return TestResultsBinary(y_prd, y)

    def test_results_loss(self) -> TestResultsLoss:
        self.model.to(self.device)
        loss = Tester._loss_step(self.model, self.device, self._test_dl, self.model.loss_fn)
        loss = torch.cat([e for e in loss], dim=0)
        loss = torch.squeeze(loss).cpu().numpy()
        li = self.model.label_index
        y = [ds[li] for ds in iter(self._test_dl)]
        y = torch.squeeze(torch.cat([e for e in y], dim=0)).cpu().numpy()
        return TestResultsLoss(loss, y, loss)

    def test_results_raw(self) -> Tuple[np.ndarray]:
        self.model.to(self.device)
        y_prd = Tester._test_step(self.model, self.device, self._test_dl)
        out: List[np.ndarray] = []
        s = len(y_prd[0])
        for i in range(s):
            o = torch.cat([e[i] for e in y_prd], dim=0)
            o = o.cpu().numpy()
            out.append(o)
        return tuple(out)
