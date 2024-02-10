from abc import abstractmethod
from typing import Literal, Callable, Optional

import numpy as np
from sklearn.metrics import mean_squared_error, log_loss, r2_score, roc_auc_score

from src.hgr import HGR, DoubleKernelHGR, AdversarialHGR, DensityHGR, SingleKernelHGR


class Metric:
    """Interface for a learning metric."""

    def __init__(self, name: str):
        """
        :param name:
            The name of the metric.
        """
        self.name: str = name

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Computes the metric between the input (x), output (y), and predictions (p)."""
        pass


class Loss(Metric):
    """Loss metric."""

    def __init__(self, classification: bool, name: Optional[str] = None):
        """
        :param classification:
            Whether the task is a regression or a classification one.

        :param name:
            The name of the metric, or None for a default name.
        """
        if classification:
            metric = log_loss
            def_name = 'BCE'
        else:
            metric = mean_squared_error
            def_name = 'MSE'
        self._metric: Callable = metric
        super(Loss, self).__init__(name=def_name if name is None else name)

    def __call__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self._metric(y, p)


class Accuracy(Metric):
    """Accuracy metric."""

    def __init__(self, classification: bool, name: Optional[str] = None):
        """
        :param classification:
            Whether the task is a regression or a classification one.

        :param name:
            The name of the metric, or None for a default name.
        """
        if classification:
            metric = roc_auc_score
            def_name = 'AUC'
        else:
            metric = r2_score
            def_name = 'R2'
        self._metric: Callable = metric
        super(Accuracy, self).__init__(name=def_name if name is None else name)

    def __call__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self._metric(y, p)


class Correlation(Metric):
    """Correlation metric."""

    def __init__(self,
                 excluded: int,
                 algorithm: Literal['prs', 'sk', 'kb', 'nn', 'kde'],
                 name: Optional[str] = None):
        """
        :param excluded:
            The index of the excluded feature.

        :param algorithm:
            The algorithm to use to compute the correlation.

        :param name:
            The name of the metric, or None for a default name.
        """
        if algorithm == 'prs':
            metric = DoubleKernelHGR(degree_a=1, degree_b=1)
            def_name = 'PEARSON'
        elif algorithm == 'sk':
            metric = SingleKernelHGR(degree=5)
            def_name = 'HGR-SK'
        elif algorithm == 'kb':
            metric = DoubleKernelHGR(degree_a=5, degree_b=5)
            def_name = 'HGR-KB'
        elif algorithm == 'nn':
            metric = AdversarialHGR()
            def_name = 'HGR-NN'
        elif algorithm == 'kde':
            metric = DensityHGR()
            def_name = 'HGR-KDE'
        else:
            raise AssertionError(f"Invalid correlation algorithm '{algorithm}'")
        self._metric: HGR = metric
        self._excluded: int = excluded
        super(Correlation, self).__init__(name=def_name if name is None else name)

    def __call__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self._metric.correlation(a=x[:, self._excluded], b=p)[0]


class DIDI(Metric):
    def __init__(self, excluded: int, classification: bool, name: Optional[str] = None):
        """
        :param excluded:
            The index of the excluded feature (must be a binary feature).

        :param classification:
            Whether the task is a regression or a classification one.

        :param name:
            The name of the metric, or None for a default name.
        """
        self._excluded: int = excluded
        self._classification: bool = classification
        super(DIDI, self).__init__(name='DIDI' if name is None else name)

    def __call__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # binarize predictions in case of classification task
        p = p.round().astype(int) if self._classification else p
        # retrive the (binary) input and compute the partial DIDI for both groups
        x = x[:, self._excluded]
        avg = np.mean(p)
        avg0 = np.mean(p[x == 0.0])
        avg1 = np.mean(p[x == 1.0])
        # return the relative DIDI as the sum of absolute differences
        return abs(avg - avg0) + abs(avg - avg1)
