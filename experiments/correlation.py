import importlib.resources
import json
from typing import Dict, Any, List, Optional

import numpy as np

from experiments import utils
from src.datasets import Dataset
from src.hgr import KernelBasedHGR


class CorrelationExperiment:
    def __init__(self, dataset: str, degree_a: int, degree_b: int):
        """
        :param dataset:
            Either a Dataset instance or a valid alias.

        :param degree_a:
            The kernel degree for the first variable.

        :param degree_b:
            The kernel degree for the second variable.
        """
        self._dataset: Dataset = utils.DATASETS[dataset]
        self._metric: KernelBasedHGR = KernelBasedHGR(degree_a=degree_a, degree_b=degree_b)
        self._result: Optional[Dict[str, Any]] = None

    @staticmethod
    def cartesian_product(datasets: List[str], degrees_a: List[int], degrees_b: List[int]) -> List:
        """Returns a list of all the combinations of experiments given."""
        return [
            CorrelationExperiment(dataset=dataset, degree_a=degree_a, degree_b=degree_b)
            for dataset in datasets for degree_a in degrees_a for degree_b in degrees_b
        ]

    @property
    def filename(self) -> str:
        """The experiment name that can be used for file naming."""
        return f'correlation_d={self._dataset.name}_m={self._metric.name}.json'

    @property
    def a(self) -> np.ndarray:
        """The first variable vector."""
        return self._dataset.excluded(backend='numpy')

    @property
    def b(self) -> np.ndarray:
        """The second variable vector."""
        return self._dataset.target(backend='numpy')

    @property
    def degree_a(self) -> int:
        """The kernel degree for the first variable."""
        return self._metric.degree_a

    @property
    def degree_b(self) -> int:
        """The kernel degree for the second variable."""
        return self._metric.degree_b

    @property
    def result(self) -> Dict[str, Any]:
        """The result of the experiment."""
        if self._result is None:
            with importlib.resources.path('experiments.results', self.filename) as file:
                if not file.exists():
                    correlation, alpha, beta = self._metric.kbhgr(a=self.a, b=self.b)
                    output = {'correlation': correlation, 'alp': list(alpha), 'beta': list(beta)}
                    with open(file, mode='w') as out_file:
                        json.dump(output, out_file)
                    self._result = output
                else:
                    with open(file, mode='r') as in_file:
                        self._result = json.load(in_file)
        return self._result

    @property
    def correlation(self) -> float:
        """The value of the correlation computed in the experiment."""
        return self.result['correlation']

    @property
    def alpha(self) -> List[float]:
        """The alpha coefficients computed in the experiment."""
        return self.result['alpha']

    @property
    def beta(self) -> List[float]:
        """The alpha coefficients computed in the experiment."""
        return self.result['beta']
