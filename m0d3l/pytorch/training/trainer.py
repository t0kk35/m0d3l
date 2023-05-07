"""
Main training class for m0d3l
(c) 2023 tsm
"""
import logging
import torch
import torch.utils.data as data
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from ..training.history import History
from m0d3l.pytorch.training.loss import Loss
from m0d3l.pytorch.training.optimizer import Optimizer
from .schedule import LRHistory, LinearLR
# noinspection PyProtectedMember
from ..models.base import Model
from ..common.exception import PyTorchTrainException
# inspection PyProtectedMember
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class Trainer:
    """
    Class to train a Neural net. Embeds some methods that hide the Pytorch training logic/loop.

    Args:
        model: The model to be trained. This needs to be a m0d3l model. Not a regular nn.Module.
        device: A torch device (CPU or GPU) to use during training.
        train_dl: A torch DataLoader object containing the training data.
        val_dl: A torch DataLoader object containing the validation data.
    """
    def __init__(self, model: Model, device: torch.device, train_dl: data.DataLoader, val_dl: data.DataLoader):
        self._model = model
        self._device = device
        self._train_dl = train_dl
        self._val_dl = val_dl
        # noinspection PyTypeChecker
        self._train_history = model.history(train_dl.batch_size, len(train_dl.dataset), len(train_dl))
        # noinspection PyTypeChecker
        self._val_history = model.history(val_dl.batch_size, len(val_dl.dataset), len(val_dl))

    @property
    def model(self) -> Model:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def number_of_train_steps(self) -> int:
        return self._train_history.number_of_steps

    @staticmethod
    def _merge_histories(train: History, val: History, epoch: int) -> Dict:
        t = {f't_{k}': v[epoch] for k, v in train.history.items()}
        v = {f'v_{k}': v[epoch] for k, v in val.history.items()}
        r = t.copy()
        r.update(v)
        return r

    @staticmethod
    def _val_is_tensor_tuple(arg):
        if not isinstance(arg, Tuple):
            raise PyTorchTrainException(
                f'Expected this argument to be Tuple. Got {type(arg)}. Please make sure that during the forward, the ' +
                f'model returns a Tuple of torch.Tensors. i.e. Tuple[torch.Tensor, ...]'
            )
        if not isinstance(arg[0], torch.Tensor):
            raise PyTorchTrainException(
                f'Expected this arguments tuple to contain tensors. Got {type(arg[0])}. Please make sure that during ' +
                f' the forward, the model returns a List of torch.Tensors. i.e. Tuple[torch.Tensor, ...]'
            )

    @staticmethod
    def _train_step(bar: Optional[tqdm], model: Model, device: torch.device, train_dl: data.DataLoader,
                    loss_fn: Loss, optimizer: Optimizer, history: History, step_scheduler: LRScheduler):
        model.train()
        for i, ds in enumerate(train_dl):
            history.start_step()
            # All data-sets to the GPU if available
            ds = tuple(d.to(device, non_blocking=True) for d in ds)
            optimizer.zero_grad()
            x = model.get_x(ds)
            y = model.get_y(ds)
            y_prd = model(x)
            # Check the output
            Trainer._val_is_tensor_tuple(y_prd)
            loss = loss_fn(y_prd, y)
            loss.backward()
            optimizer.step()
            history.end_step(y_prd, y, loss)
            if step_scheduler is not None:
                step_scheduler.step()
            del ds
            del loss
            if bar is not None:
                bar.update(1)
            if history.early_break():
                f'Early Breaking Validation at step {history.step}'
                break

    @staticmethod
    def _validation_step(bar: Optional[tqdm], model: Model, device: torch.device, val_ds: data.DataLoader,
                         loss_fn: Loss, history: History):
        with torch.no_grad():
            model.eval()
            for i, ds in enumerate(val_ds):
                history.start_step()
                # All data-sets to the GPU if available
                ds = tuple(d.to(device, non_blocking=True) for d in ds)
                x = model.get_x(ds)
                y = model.get_y(ds)
                y_prd = model(x)
                loss = loss_fn(y_prd, y)
                history.end_step(y_prd, y, loss)
                del ds
                del loss
                if bar is not None:
                    bar.update(1)
                if history.early_break():
                    logger.info(f'Early Breaking Validation at step {history.step}')
                    break

    def _train(self, epochs: int, loss_fn: Loss, o: Optimizer, step_scheduler: LRScheduler) -> Tuple[History, History]:
        self.model.to(self.device)
        for epoch in range(epochs):
            self._train_history.start_epoch()
            self._val_history.start_epoch()
            with tqdm(total=self._train_history.number_of_steps+self._val_history.number_of_steps,
                      desc=f'Epoch {epoch+1:03d}/{epochs:03d}', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                      ncols=127) as bar:
                Trainer._train_step(bar, self.model, self.device, self._train_dl, loss_fn, o,
                                    self._train_history, step_scheduler)
                Trainer._validation_step(bar, self._model, self._device, self._val_dl, loss_fn, self._val_history)
                self._train_history.end_epoch()
                self._val_history.end_epoch()
                bar.set_postfix(self._merge_histories(self._train_history, self._val_history, epoch))
        return self._train_history, self._val_history

    def find_lr(self, optimizer: Optimizer, end_lr: float, max_steps: int = 100) -> LRHistory:
        self.model.to(self.device)
        # Set up a step schedule with new optimizer. It will adjust (increase) the LR at each step.
        lr_schedule = LinearLR(end_lr, max_steps, optimizer)
        # noinspection PyTypeChecker
        history = LRHistory(self._train_dl.batch_size, len(self._train_dl.dataset), len(self._train_dl),
                            lr_schedule, 5, 0.1, max_steps)
        # Run Loop
        with tqdm(total=min(max_steps, history.number_of_steps), desc=f'Finding LR in {max_steps} steps',
                  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', ncols=127) as bar:
            self._train_step(bar, self.model, self.device, self._train_dl, self.model.loss_fn,
                             optimizer, history, lr_schedule)
        return history

    def train(self, epochs: int, optimizer: Optimizer, scheduler: LRScheduler = None) -> Tuple[History, History]:
        """
        Train a model for a specific set of epochs using the provided optimizer.

        Args:
            epochs: Number of epoch to train for.
            optimizer: An m0d3l Optimizer class to use during training.
            scheduler: A Learning rate step scheduler from the torch package. See torch optim documentation.

        Returns:
             A tuple of 2 Objects of type LR_History. They contain the training statistics for the training and
                validation steps respectively.
        """
        return self._train(epochs, self.model.loss_fn, optimizer, scheduler)

    def __repr__(self):
        return f'Trainer for Model: {self.model.__class__.__name__}, on device: {self.device}'
