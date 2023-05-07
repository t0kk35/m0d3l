
"""
Helper for metric calculation
(c) tsm 2023
"""
import numpy as np
from typing import Tuple


class Metrics:
    @staticmethod
    def _base_metrics(pr: np.array, lb: np.array) -> Tuple[int, int, int, int]:
        tp = (pr & lb).sum()
        tn = ((~pr) & (~lb)).sum()
        fp = (pr & (~lb)).sum()
        fn = ((~pr) & lb).sum()
        return tp, tn, fp, fn

    @staticmethod
    def _f_beta_score(precision: float, recall: float, beta: float, eps=1e-12) -> float:
        f_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + eps)
        return f_beta

    @classmethod
    def score_metrics(cls, pr: np.array, lb: np.array, threshold=0.5, eps=1e-12) -> Tuple[float, ...]:
        pr = np.greater_equal(pr, threshold)
        lb = np.greater_equal(lb, 0.5)
        tp, tn, fp, fn = Metrics._base_metrics(pr, lb)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = Metrics._f_beta_score(precision, recall, 1)
        f2 = Metrics._f_beta_score(precision, recall, 2)
        return precision, recall, f1, f2

    @classmethod
    def score_metrics_names(cls) -> Tuple[str, ...]:
        return 'Precision', 'Recall', 'F1', 'F2'
