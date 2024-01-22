import importlib.resources
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from experiments.experiment import Experiment
from src.datasets import Dataset
from src.hgr import KernelBasedHGR


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

    @staticmethod
    def export_monotonicity(dataset: Dataset,
                            degrees_a: List[int],
                            degrees_b: List[int],
                            vmin: Optional[float] = None,
                            vmax: Optional[float] = None,
                            formats: Optional[List[str]] = None,
                            verbose: bool = False,
                            plot: bool = False):
        experiments = [
            CorrelationExperiment(dataset=dataset, metric=KernelBasedHGR(degree_a=da, degree_b=db))
            for da in degrees_a
            for db in degrees_b
        ]
        # run experiments and store results
        results = [experiment.correlation for experiment in tqdm(experiments)]
        results = np.reshape(results, (len(degrees_a), len(degrees_b)))
        # set graphics context
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        # plot results
        fig = plt.figure(figsize=(16, 9), tight_layout=True)
        ax = fig.gca()
        col = ax.imshow(results.transpose()[::-1], cmap=plt.colormaps['gray'], vmin=vmin, vmax=vmax)
        fig.colorbar(col, ax=ax)
        ax.set_xlabel('Degree A')
        ax.set_xticks(np.arange(len(degrees_a) + 1) - 0.5)
        ax.set_xticklabels([''] * (len(degrees_a) + 1))
        ax.set_xticks(np.arange(len(degrees_a)), minor=True)
        ax.set_xticklabels(degrees_a, minor=True)
        ax.set_ylabel('Degree B')
        ax.set_yticks(np.arange(len(degrees_b) + 1) - 0.5)
        ax.set_yticklabels([''] * (len(degrees_b) + 1))
        ax.set_yticks(np.arange(len(degrees_b)), minor=True)
        ax.set_yticklabels(degrees_b[::-1], minor=True)
        ax.grid(True, which='major')
        # store, print, and plot if necessary
        for extension in ([] if formats is None else formats):
            filename = f'monotonicity_{dataset.name}.{extension}'
            with importlib.resources.path('experiments.exports', filename) as file:
                fig.savefig(file)
        if verbose:
            print(results)
        if plot:
            fig.show()
