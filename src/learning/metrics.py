from abc import abstractmethod
from typing import Literal, Callable

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, log_loss, r2_score, roc_auc_score

from src.hgr import HGR, DoubleKernelHGR, AdversarialHGR, DensityHGR


class Metric:
    """Interface for a learning metric."""

    def __init__(self, name: str):
        """
        :param name:
            The name of the metric.
        """
        self.name: str = name

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> torch.Tensor:
        """Computes the metric between the input (x), output (y), and predictions (p)."""
        pass


class Correlation(Metric):
    """Correlation metric."""

    def __init__(self, excluded: int, algorithm: Literal['prs', 'kb', 'nn', 'kde']):
        """
        :param excluded:
            The index of the excluded feature.

        :param algorithm:
            The algorithm to use to compute the correlation.
        """
        if algorithm == 'prs':
            metric = DoubleKernelHGR(degree_a=1, degree_b=1)
            name = 'PEARSON'
        elif algorithm == 'kb':
            metric = DoubleKernelHGR()
            name = 'HGR-KB'
        elif algorithm == 'nn':
            metric = AdversarialHGR()
            name = 'HGR-NN'
        elif algorithm == 'kde':
            metric = DensityHGR()
            name = 'HGR-KDE'
        else:
            raise AssertionError(f"Invalid correlation algorithm '{algorithm}'")
        self._metric: HGR = metric
        self._excluded: int = excluded
        super(Correlation, self).__init__(name=name)

    def __call__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> torch.Tensor:
        return self._metric.correlation(a=x[:, self._excluded], b=y)


class Loss(Metric):
    """Loss metric."""

    def __init__(self, classification: bool):
        """
        :param classification:
            Whether the task is a regression or a classification one.
        """
        if classification:
            metric = log_loss
            name = 'BCE'
        else:
            metric = mean_squared_error
            name = 'MSE'
        self._metric: Callable = metric
        super(Loss, self).__init__(name=name)

    def __call__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> torch.Tensor:
        return self._metric(y, p)


class Accuracy(Metric):
    """Accuracy metric."""

    def __init__(self, classification: bool):
        """
        :param classification:
            Whether the task is a regression or a classification one.
        """
        if classification:
            metric = roc_auc_score
            name = 'AUC'
        else:
            metric = r2_score
            name = 'R2'
        self._metric: Callable = metric
        super(Accuracy, self).__init__(name=name)

    def __call__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> torch.Tensor:
        return self._metric(y, p)
