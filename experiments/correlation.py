import importlib.resources
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm

from experiments.experiment import Experiment
from src.datasets import Dataset
from src.datasets.deterministic import Deterministic
from src.hgr import KernelBasedHGR, HGR


@dataclass(frozen=True)
class CorrelationExperiment(Experiment):
    """An experiment where the correlation between two variables is computed."""

    @property
    def name(self) -> str:
        return 'correlation'

    def _compute(self) -> Dict[str, Any]:
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
    def monotonicity(dataset: Dataset,
                     degrees_a: Iterable[int] = (1, 2, 3, 4, 5, 6, 7),
                     degrees_b: Iterable[int] = (1, 2, 3, 4, 5, 6, 7),
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     formats: Iterable[str] = (),
                     verbose: bool = False,
                     plot: bool = False):
        # run experiments
        results = np.ndarray((len(degrees_a), len(degrees_b)))
        pbar = tqdm(total=len(degrees_a) * len(degrees_b))
        for i, da in enumerate(degrees_a):
            for j, db in enumerate(degrees_b):
                experiment = CorrelationExperiment(dataset=dataset, metric=KernelBasedHGR(degree_a=da, degree_b=db))
                results[i, j] = experiment.correlation
                pbar.update(n=1)
        pbar.close()
        # plot results
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        fig = plt.figure(figsize=(12, 9), tight_layout=True)
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
        for extension in formats:
            name = f'monotonicity_{dataset.fullname}.{extension}'
            with importlib.resources.path('experiments.exports', name) as file:
                fig.savefig(file, bbox_inches='tight')
        if verbose:
            print(results)
        if plot:
            fig.show()

    @staticmethod
    def correlations(datasets: Dict[str, Callable[[float], Deterministic]],
                     metrics: Dict[str, HGR],
                     noises: Iterable[float] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3),
                     seeds: Iterable[int] = (0, 1, 2, 3, 4),
                     columns: int = 2,
                     formats: Optional[List[str]] = None,
                     verbose: bool = False,
                     plot: bool = False):
        # run experiments
        results = []
        pbar = tqdm(total=len(noises) * len(datasets) * len(metrics) * len(seeds))
        for noise in noises:
            for dataset_name, dataset_fn in datasets.items():
                dataset = dataset_fn(noise)
                for metric_name, metric in metrics.items():
                    for seed in seeds:
                        results.append({
                            'correlation': CorrelationExperiment(dataset=dataset, metric=metric, seed=seed).correlation,
                            'dataset': dataset_name,
                            'metric': metric_name,
                            'noise': noise,
                            'seed': seed
                        })
                        pbar.update(n=1)
        pbar.close()
        results = pd.DataFrame(results)
        # plot results
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        rows = int(np.ceil((len(datasets) + 1) / columns))
        fig = plt.figure(figsize=(4 * columns, 4 * rows), tight_layout=True)
        handles, labels = [], []
        for i, (name, data) in enumerate(results.groupby('dataset')):
            ax = fig.add_subplot(rows, columns, i + 1)
            sns.lineplot(data=data, x='noise', y='correlation', hue='metric', style='metric')
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            ax.set_xticks(noises)
            # plot the original data without noise
            sub_ax = inset_axes(ax, width='30%', height='30%', loc='lower left')
            datasets[name](0.0).plot(ax=sub_ax, color='tab:blue')
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
        ax = fig.add_subplot(rows, columns, rows * columns)
        ax.legend(handles, labels, title='Metric', loc='upper center', borderpad=1.5, labelspacing=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        # store, print, and plot if necessary
        for extension in ([] if formats is None else formats):
            name = f'correlations.{extension}'
            with importlib.resources.path('experiments.exports', name) as file:
                fig.savefig(file, bbox_inches='tight')
        if verbose:
            print(results)
        if plot:
            fig.show()
