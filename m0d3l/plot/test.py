"""
Helper Class for Test plotting.
(c) 2023 tsm
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve

from .metrics import Metrics
from ..common.testresults import TestResultsBinary, TestResultsLoss

from typing import Tuple

class TestPlot:

    @classmethod
    def print_binary_classification_report(cls, results: TestResultsBinary, threshold=0.5):
        predictions = results.y_prd[0]
        labels = results.y[0]
        ap_score = average_precision_score(labels, predictions)
        auc_score = roc_auc_score(labels, predictions)
        predictions = (predictions > threshold)
        cr = classification_report(labels, predictions)
        print('------------- Classification report -----------------')
        print(cr)
        print()
        print(f'auc score : {auc_score:0.4f}')
        print(f'ap score  : {ap_score:0.4f}')
        print('-----------------------------------------------------')

    @classmethod
    def plot_binary_confusion_matrix(cls, results: TestResultsBinary, fig_size: Tuple[float, float] = None,
                                     threshold=0.5):
        predictions = results.y_prd[0]
        labels = results.y[0]
        predictions = (predictions > threshold)
        cm = confusion_matrix(labels, predictions)
        _ = plt.figure(figsize=fig_size)
        plt.clf()
        c_map = plt.get_cmap('Blues')
        plt.imshow(cm, interpolation='nearest', cmap=c_map)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        class_names = ['Non-Fraud', 'Fraud']
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        q = [['TN', 'FP'], ['FN', 'TP']]
        c_map_min, c_map_max = c_map(0), c_map(256)
        cut = (cm.max() + cm.min()) / 2.0
        for i in range(2):
            for j in range(2):
                color = c_map_max if cm[i, j] < cut else c_map_min
                plt.text(j, i, f'{str(q[i][j])} = {str(cm[i][j])}', color=color, ha="center", va="center")
        plt.show()

    @classmethod
    def plot_roc_curve(cls, results: TestResultsBinary, fig_size: Tuple[float, float] = None):
        predictions = results.y_prd[0]
        labels = results.y[0]
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        auc_score = roc_auc_score(labels, predictions)
        _ = plt.figure(figsize=fig_size)
        plt.clf()
        plt.plot(fpr, tpr, label=f'AUC Score = {auc_score:0.4f}')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc=4)
        plt.grid(color='0.95')
        plt.show()

    @classmethod
    def plot_precision_recall_curve(cls, results: TestResultsBinary, fig_size: Tuple[float, float] = None):
        predictions = results.y_prd[0]
        labels = results.y[0]
        p, r, _ = precision_recall_curve(labels, predictions)
        ap_score = average_precision_score(labels, predictions)
        _ = plt.figure(figsize=fig_size)
        plt.clf()
        plt.plot(r, p, label=f'AP Score = {ap_score:0.4f}')
        plt.title('Precision Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc=1)
        plt.grid(color='0.95')
        plt.show()

    @classmethod
    def plot_loss_binary(cls, results: TestResultsLoss, bins: int = 20, fig_size: Tuple[float, float] = None,
                         log_scale: bool = True):
        fraud_loss = results.loss[np.where(results.y[0] == 1.0)[0]]
        n_fraud_loss = results.loss[np.where(results.y[0] == 0.0)[0]]
        _ = plt.figure(figsize=fig_size)
        plt.clf()
        plt.hist([fraud_loss, n_fraud_loss], bins=bins, label=['Fraud', 'Non-Fraud'], color=['red', 'green'], alpha=0.8)
        plt.title(f'Loss Histogram')
        if log_scale:
            plt.yscale('log')
        plt.xlabel('Loss')
        plt.ylabel('#Entries')
        plt.legend(loc=1)
        plt.grid(color='0.95')
        plt.show()

    @classmethod
    def plot_loss_metrics(cls, results: TestResultsLoss, bins=20, fig_size: Tuple[float, float] = None):
        loss = results.loss
        labels = results.y[0]
        ths = np.linspace(np.amin(loss), np.amax(loss), bins)
        y = np.array([Metrics.score_metrics(loss, labels, t) for t in ths])
        _ = plt.figure(figsize=fig_size)
        plt.clf()
        for i, n in enumerate(Metrics.score_metrics_names()):
            plt.plot(ths, y[:, i], markersize=5, marker='o', label=f'{n}')
        plt.title('Metrics per Loss Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Metric')
        plt.legend(loc=1)
        plt.grid(color='0.95')
        plt.show()
