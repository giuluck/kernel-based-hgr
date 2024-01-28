import importlib.resources
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from experiments.experiment import Experiment
from src.datasets import Dataset
from src.datasets.deterministic import Deterministic
from src.hgr import DoubleKernelHGR, HGR, KernelBasedHGR, AdversarialHGR

PALETTE: List[str] = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class CorrelationExperiment(Experiment):
    """An experiment where the correlation between two variables is computed."""

    def _compute(self) -> Experiment.Result:
        start = time.time()
        a = self.dataset.excluded(backend='numpy')
        b = self.dataset.target(backend='numpy')
        hgr, additional = self.metric.correlation(a=a, b=b)
        gap = time.time() - start
        if len(additional) > 0:
            external = f'{self.key}.pkl'
            with importlib.resources.path('experiments.results', external) as path:
                assert not path.exists(), f"File '{self.key}' is already present in package 'experiments.results'"
            with open(path, 'wb') as file:
                pickle.dump(additional, file=file)
        else:
            external = None
        return Experiment.Result(timestamp=start, execution=gap, external=external, correlation=hgr)

    @property
    def name(self) -> str:
        return 'correlation'

    @staticmethod
    def monotonicity(datasets: Iterable[Dataset],
                     degrees_a: Iterable[int] = (1, 2, 3, 4, 5, 6, 7),
                     degrees_b: Iterable[int] = (1, 2, 3, 4, 5, 6, 7),
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     formats: Iterable[str] = ('png',),
                     plot: bool = False,
                     save_time: int = 60):
        # run experiments
        experiments = CorrelationExperiment.doe(
            file_name='correlation',
            save_time=save_time,
            seed=0,
            dataset={dataset.key: dataset for dataset in datasets},
            metric={(da, db): DoubleKernelHGR(degree_a=da, degree_b=db) for da in degrees_a for db in degrees_b}
        )
        for dataset in datasets:
            # build results
            results = np.zeros((len(degrees_a), len(degrees_b)))
            for i, da in enumerate(degrees_a[::-1]):
                for j, db in enumerate(degrees_b):
                    results[i, j] = experiments[(dataset.key, (db, da))].result['correlation']
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
                with importlib.resources.path('experiments.exports', name) as file:
                    fig.savefig(file, bbox_inches='tight')
            if plot:
                config = dataset.configuration
                name = config.pop('name').title()
                info = ', '.join({f'{key}={value}' for key, value in config.items()})
                fig.suptitle(f'Monotonicity in {name}({info})')
                fig.show()

    @staticmethod
    def correlations(datasets: Dict[str, Callable[[float], Deterministic]],
                     metrics: Dict[str, HGR],
                     noises: Iterable[float] = np.linspace(0.0, 3.0, num=31, endpoint=True).round(2),
                     seeds: Iterable[int] = range(30),
                     columns: int = 3,
                     legend: int = 1,
                     formats: Iterable[str] = ('png',),
                     plot: bool = False,
                     save_time: int = 60):
        # run experiments
        experiments = CorrelationExperiment.doe(
            file_name='correlation',
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
                'execution': experiment.result['execution'],
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
        legend = (legend + plots - 1) % plots + 1
        fig_data = plt.figure(figsize=(4 * columns, 4 * rows), tight_layout=True)
        handles, labels, ax = [], [], None
        names = list(datasets.keys())[::-1]
        for i in np.arange(plots) + 1:
            if i == legend:
                continue
            name = names.pop()
            ax = fig_data.add_subplot(rows, columns, i, sharex=ax, sharey=ax)
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
            ax.set_ylabel('Correlation')
            ax.set_ylim((-0.1, 1.1))
            # plot the original data without noise
            sub_ax = inset_axes(ax, width='30%', height='30%', loc='upper right')
            datasets[name](0.0).plot(ax=sub_ax, color='black')
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
        # plot the legend
        ax = fig_data.add_subplot(rows, columns, legend)
        ax.legend(handles, labels, loc='center', labelspacing=1.5, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot the time in a different figure
        fig_time = plt.figure(figsize=(12, 12), tight_layout=True)
        ax = fig_time.gca()
        sns.barplot(
            data=results,
            x='metric',
            y='execution',
            hue='metric',
            estimator='mean',
            errorbar='sd',
            legend=False,
            ax=ax
        )
        ax.set_xlabel(None)
        ax.set_ylabel('Execution Time (s)')
        ax.set_yscale('log')
        # store, print, and plot if necessary
        for extension in formats:
            package = 'experiments.exports'
            fig_data.suptitle('Computed Correlations')
            fig_time.suptitle('Execution Times to Compute Correlations')
            with importlib.resources.path(package, f'correlations_data.{extension}') as file:
                fig_data.savefig(file, bbox_inches='tight')
            with importlib.resources.path(package, f'correlations_time.{extension}') as file:
                fig_time.savefig(file, bbox_inches='tight')
        if plot:
            fig_data.show()
            fig_time.show()

    @staticmethod
    def kernels(datasets: Iterable[Dataset],
                formats: Iterable[str] = ('png',),
                plot: bool = False):
        def standardize(v):
            return (v - v.mean()) / (v.std(ddof=0))

        # run experiments
        experiments = CorrelationExperiment.doe(
            file_name='correlation',
            save_time=0,
            dataset={dataset.key: dataset for dataset in datasets},
            metric=[DoubleKernelHGR(), AdversarialHGR()],
            seed=0
        )
        for dataset in datasets:
            # build and plot results
            a = dataset.excluded(backend='numpy')
            b = dataset.target(backend='numpy')
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)
            # retrieve hgr-kb
            exp_kb = experiments[(dataset.key, 0)]
            # noinspection PyUnresolvedReferences
            f_kb = KernelBasedHGR.kernel(a, degree=exp_kb.metric.degree_a, use_torch=False)
            # noinspection PyUnresolvedReferences
            g_kb = KernelBasedHGR.kernel(b, degree=exp_kb.metric.degree_b, use_torch=False)
            fa_kb = standardize(f_kb @ exp_kb.result['alpha'])
            gb_kb = standardize(g_kb @ exp_kb.result['beta'])
            # retrieve hgr-nn (switch sign to match the same orientation of the other kernels)
            exp_nn = experiments[(dataset.key, 1)]
            a_nn = torch.tensor(a, dtype=torch.float32).reshape((-1, 1))
            b_nn = torch.tensor(b, dtype=torch.float32).reshape((-1, 1))
            fa_nn = standardize(exp_nn.result['f'](a_nn).numpy(force=True).flatten())
            gb_nn = standardize(exp_nn.result['g'](b_nn).numpy(force=True).flatten())
            if fa_kb[0] * fa_nn[0] < 0:
                fa_nn, gb_nn = -fa_nn, -gb_nn
            # plot kernels
            for x, y, ax, kernel in zip([a, b], [(fa_kb, fa_nn), (gb_kb, gb_nn)], axes, ['X', 'Y']):
                sns.lineplot(x=x, y=y[0], sort=True, color=PALETTE[0], linestyle='-', label='HGR-KB', ax=ax)
                sns.lineplot(x=x, y=y[1], sort=True, color=PALETTE[1], linestyle='--', label='HGR-NN', ax=ax)
                ax.set_title(f'{kernel} Kernel')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.legend(loc='best')
            # store and plot if necessary
            for extension in formats:
                config = dataset.configuration
                name = config.pop('name').title()
                info = ', '.join({f'{key}={value}' for key, value in config.items()})
                fig.suptitle(f'Kernels for {name}({info})')
                name = f'kernels_{dataset.key}.{extension}'
                with importlib.resources.path('experiments.exports', name) as file:
                    fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.show()
