"""
Helper Class for Training plotting.
(c) 2023 tsm
"""
import matplotlib.pyplot as plt

from ..pytorch.models.base import History

from typing import Tuple

class TrainPlot:

    @classmethod
    def plot_history(cls, history: Tuple[History, History], fig_size: Tuple[float, float] = None):
        train = history[0]
        val = history[1]
        _ = plt.figure(figsize=fig_size)
        plt.clf()
        plt.title('Training Metrics')
        plt.suptitle('Training and Validation')
        axis = []
        epochs = [i for i in range(1, train.epoch+1)]

        # This logic assumes the train and validate history contain the same metrics.
        for i, k in enumerate(train.history.keys()):
            if i == 0:
                ax = plt.subplot2grid((2, 1), (i, 0))
            else:
                ax = plt.subplot2grid((2, 1), (i, 0), sharex=axis[0])
            ax.set(xlabel='Epoch')
            ax.plot(epochs, train.history[k], label=f'train_{k}')
            ax.plot(epochs, val.history[k], label=f'val_{k}')
            if train.history[k][0] > train.history[k][-1]:
                ax.legend(loc=4)
            else:
                ax.legend(loc=2)
            ax.grid(color='0.95')
            axis.append(ax)

        plt.show()

    @classmethod
    def plot_lr(cls, history: History, fig_size: Tuple[float, float] = None):
        lr = history.history['lr']
        loss = history.history['loss']
        _ = plt.figure(figsize=fig_size)
        plt.clf()
        plt.plot(lr, loss)
        plt.title('Loss per Learning rate')
        plt.xlabel('Learning Rate (In log scale)')
        plt.xscale('log')
        plt.ylabel('Loss')
        plt.grid(color='0.95')
        plt.show()
