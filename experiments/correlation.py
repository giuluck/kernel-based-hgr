import importlib.resources
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from experiments.experiment import Experiment
from src.datasets import Dataset
from src.datasets.deterministic import Deterministic
from src.hgr import KernelBasedHGR, HGR

PALETTE: List[str] = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class CorrelationExperiment(Experiment):
    """An experiment where the correlation between two variables is computed."""

    def _compute(self) -> Dict[str, Any]:
        return self.metric.correlation(a=self.a, b=self.b)

    @property
    def name(self) -> str:
        return 'correlation'

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
                     plot: bool = False,
                     save_time: int = 60):
        # run experiments
        experiments = CorrelationExperiment.doe(
            file_name='monotonicity.json',
            save_time=save_time,
            dataset=dataset,
            seed=0,
            metric={(da, db): KernelBasedHGR(degree_a=da, degree_b=db) for da in degrees_a for db in degrees_b}
        )
        # build results
        results = np.zeros((len(degrees_a), len(degrees_b)))
        for i, da in enumerate(degrees_a[::-1]):
            for j, db in enumerate(degrees_b):
                results[i, j] = experiments[(db, da)].result['correlation']
        degrees_b = degrees_b[::-1]
        # plot results
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        ax = fig.gca()
        col = ax.imshow(results, cmap=plt.colormaps['gray'], vmin=vmin, vmax=vmax)
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
        ax.set_yticklabels(degrees_b, minor=True)
        ax.grid(True, which='major')
        # store, print, and plot if necessary
        for extension in formats:
            name = f'monotonicity_{dataset.key}.{extension}'
            with importlib.resources.path('experiments.results', name) as file:
                fig.savefig(file, bbox_inches='tight')
        if verbose:
            print(results)
        if plot:
            fig.show()

    @staticmethod
    def correlations(datasets: Dict[str, Callable[[float], Deterministic]],
                     metrics: Dict[str, HGR],
                     noises: Iterable[float] = np.linspace(0.0, 3.0, num=31, endpoint=True).round(2),
                     seeds: Iterable[int] = range(30),
                     columns: int = 3,
                     legend: int = 1,
                     formats: Optional[List[str]] = None,
                     verbose: bool = False,
                     plot: bool = False,
                     save_time: int = 60):
        # run experiments
        experiments = CorrelationExperiment.doe(
            file_name='correlations.json',
            save_time=save_time,
            dataset={(k, n): fn(n) for k, fn in datasets.items() for n in noises},
            metric=metrics,
            seed=list(seeds)
        )
        # build results
        results = []
        for key, experiment in experiments.items():
            results.append({
                'correlation': experiment.result['correlation'],
                'dataset': key[0][0],
                'noise': key[0][1],
                'metric': key[1],
                'seed': key[2]
            })
        results = pd.DataFrame(results)
        # plot results
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        plots = len(datasets) + 1
        rows = int(np.ceil(plots / columns))
        legend = (legend + plots) % plots
        fig = plt.figure(figsize=(4 * columns, 4 * rows), tight_layout=True)
        handles, labels, ax = [], [], None
        names = list(datasets.keys())[::-1]
        for i in np.arange(plots) + 1:
            if i == legend:
                continue
            name = names.pop()
            ax = fig.add_subplot(rows, columns, i, sharex=ax, sharey=ax)
            sns.lineplot(
                data=results[results['dataset'] == name],
                x='noise',
                y='correlation',
                hue='metric',
                style='metric',
                estimator='mean',
                errorbar='sd',
                palette=PALETTE[:len(metrics)],
                linewidth=2
            )
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            ax.set_xlabel('Noise Level $\sigma$')
            ax.set_ylabel(None)
            ax.set_ylim((-0.1, 1.1))
            # plot the original data without noise
            sub_ax = inset_axes(ax, width='30%', height='30%', loc='upper right')
            datasets[name](0.0).plot(ax=sub_ax, color='black')
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
        ax = fig.add_subplot(rows, columns, legend)
        ax.legend(handles, labels, loc='center', labelspacing=1.5, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        # store, print, and plot if necessary
        for extension in ([] if formats is None else formats):
            name = f'correlations.{extension}'
            with importlib.resources.path('experiments.results', name) as file:
                fig.savefig(file, bbox_inches='tight')
        if verbose:
            print(results)
        if plot:
            fig.show()
