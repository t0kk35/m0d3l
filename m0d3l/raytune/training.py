
import os
import torch
import torch.utils.data as data
from torch.optim.lr_scheduler import LRScheduler

from ray.air import session
from ray.air.checkpoint import Checkpoint

from ..pytorch.common.exception import PyTorchTrainException
from ..pytorch.models.base import Model
from ..pytorch import Trainer
from ..pytorch.training.optimizer import Optimizer
from ..pytorch.training.history import History, TunableHistory

from typing import Tuple

class RayTrainer(Trainer):
    def __init__(self, model: Model, device: torch.device, train_dl: data.DataLoader, val_dl: data.DataLoader):
        super(RayTrainer, self).__init__(model, device, train_dl, val_dl)

    def train(self, epochs: int, optimizer: Optimizer, scheduler: LRScheduler = None) -> Tuple[History, History]:
        """
        Train a model for a specific set of epochs using the provided optimizer.

        Args:
            epochs: Number of epoch to train for.
            optimizer: An m0d3l Optimizer class to use during training.
            scheduler: A Learning rate step scheduler from the torch package. See torch optim documentation.

        Returns:
             A tuple of 2 Objects of type History. They contain the training statistics for the training and
                validation steps respectively.
        """
        # Load Potential Check Point
        loaded_checkpoint = session.get_checkpoint()
        if loaded_checkpoint:
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
            self.model.load_state_dict(model_state)
            optimizer.optimizer.load_state_dict(optimizer_state)

        self.model.to(self.device)
        for epoch in range(epochs):
            self._train_history.start_epoch()
            self._val_history.start_epoch()
            Trainer._train_step(
                None, self.model, self.device, self._train_dl, self.model.loss_fn, optimizer,
                self._train_history, scheduler
            )
            Trainer._validation_step(
                None, self._model, self._device, self._val_dl, self.model.loss_fn, self._val_history
            )
            self._train_history.end_epoch()
            self._val_history.end_epoch()

            if isinstance(self._val_history, TunableHistory):
                os.makedirs("my_model", exist_ok=True)
                torch.save((self.model.state_dict(), optimizer.optimizer.state_dict()), "my_model/checkpoint.pt")
                checkpoint = Checkpoint.from_directory("my_model")
                session.report(self._val_history.tune_stats, checkpoint=checkpoint)
            else:
                raise PyTorchTrainException(
                    f'The Validation History must be a subclass of Tunable History.'
                )

        return self._train_history, self._val_history
