from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from experiments.experiment import Experiment


@dataclass(frozen=True)
class CorrelationExperiment(Experiment):
    """An experiment where the correlation between two variables is computed."""

    @property
    def name(self) -> str:
        return 'correlation'

    def _run(self) -> Dict[str, Any]:
        return self.metric.correlation(a=self.a, b=self.b)

    @property
    def a(self) -> np.ndarray:
        """The first variable vector."""
        return self.dataset.excluded(backend='numpy')

    @property
    def b(self) -> np.ndarray:
        """The second variable vector."""
        return self.dataset.target(backend='numpy')

    @property
    def correlation(self) -> float:
        """The value of the correlation computed in the experiment."""
        return self._result['correlation']
