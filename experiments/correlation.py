import importlib.resources
import os.path
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Iterable, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm

from experiments.experiment import Experiment
from src.datasets import Dataset, Deterministic
from src.hgr import DoubleKernelHGR, HGR, AdversarialHGR, Oracle, KernelsHGR

PALETTE: List[str] = [
    '#000000',
    '#377eb8',
    '#ff7f00',
    '#4daf4a',
    '#f781bf',
    '#a65628',
    '#984ea3',
    '#999999',
    '#e41a1c',
    '#dede00'
]
"""The color palette for plotting data."""


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class CorrelationExperiment(Experiment):
    """An experiment where the correlation between two variables is computed."""

    dataset: Dataset = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The dataset used in the experiment."""

    metric: HGR = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The HGR metric used in the experiment."""

    seed: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The random seed used in the experiment."""

    def _compute(self) -> Experiment.Result:
        pl.seed_everything(self.seed, workers=True)
        start = time.time()
        a = self.dataset.excluded(backend='numpy')
        b = self.dataset.target(backend='numpy')
        hgr, additional = self.metric.correlation(a=a, b=b)
        gap = time.time() - start
        # store external files only for NN kernels, in the other cases include the additional results in the object
        if isinstance(self.metric, AdversarialHGR):
            external = os.path.join('correlation', f'{self.key}.pkl')
            with importlib.resources.files('experiments.results') as folder:
                filepath = os.path.join(folder, external)
                # overwrite files rather than asserting that they are not present since an abrupt interruption of the
                # DoE might cause leaking external files to be stored while the original results are not
                if os.path.exists(filepath):
                    print(f"Overwriting file '{self.key}' since it is already present in package 'experiments.results'")
                else:
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as file:
                pickle.dump(additional, file=file)
            return Experiment.Result(timestamp=start, execution=gap, external=external, correlation=hgr)
        else:
            return Experiment.Result(timestamp=start, execution=gap, external=None, correlation=hgr, **additional)

    @property
    def name(self) -> str:
        return 'correlation'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(
            experiment=self.name,
            dataset=self.dataset.configuration,
            metric=self.metric.configuration,
            seed=self.seed
        )

    @property
    def key(self) -> str:
        return f'{self.name}_{self.dataset.key}_{self.metric.key}_{self.seed}'

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
            verbose=False,
            seed=0,
            dataset={dataset.key: dataset for dataset in datasets},
            metric={(da, db): DoubleKernelHGR(degree_a=da, degree_b=db) for da in degrees_a for db in degrees_b}
        )
        # plot results
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        for dataset in datasets:
            # build results
            results = np.zeros((len(degrees_a), len(degrees_b)))
            for i, da in enumerate(degrees_a):
                for j, db in enumerate(degrees_b):
                    results[i, j] = experiments[(dataset.key, (da, db))].result['correlation']
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
                name = f'monotonicity_{dataset.key}.{extension}'
                with importlib.resources.path('experiments.exports', name) as file:
                    fig.savefig(file, bbox_inches='tight')
            if plot:
                config = dataset.configuration
                name = config.pop('name').title()
                info = ', '.join({f'{key}={value}' for key, value in config.items()})
                fig.suptitle(f'Monotonicity in {name}({info})')
                fig.show()
            plt.close(fig)

    @staticmethod
    def correlations(datasets: Dict[str, Callable[[float, int], Deterministic]],
                     metrics: Dict[str, HGR],
                     noises: Iterable[float] = np.linspace(0.0, 3.0, num=16, endpoint=True).round(2),
                     data_seeds: Iterable[int] = range(10),
                     algorithm_seeds: Iterable[int] = range(10),
                     test: bool = False,
                     columns: int = 2,
                     formats: Iterable[str] = ('png',),
                     plot: bool = False,
                     save_time: int = 60):
        assert len(data_seeds) > 1 or not test, "Tests cannot be performed only if more than one data seed is passed"
        # run experiments
        metrics = {'ORACLE': Oracle(), **metrics}
        experiments = CorrelationExperiment.doe(
            file_name='correlation',
            save_time=save_time,
            verbose=False,
            dataset={(k, n, s): fn(n, s) for k, fn in datasets.items() for n in noises for s in data_seeds},
            metric=metrics,
            seed=list(algorithm_seeds)
        )
        # build results
        results = []
        for key, experiment in tqdm(experiments.items(), desc='Storing Correlations'):
            dataset, noise, seed = key[0]
            config = dict(dataset=dataset, noise=noise, metric=key[1], data_seed=seed, algorithm_seed=key[2])
            results.append({
                'correlation': experiment.result['correlation'],
                'execution': experiment.result['execution'],
                'split': 'train',
                **config
            })
            # build results for test data (use all the data seeds but the training one)
            if test and isinstance(experiment.metric, KernelsHGR):
                for s in data_seeds:
                    if s == seed:
                        continue
                    dataset_seed = datasets[dataset](noise, s)
                    x = dataset_seed.excluded(backend='numpy')
                    y = dataset_seed.target(backend='numpy')
                    results.append({
                        'correlation': experiment.metric.kernels(a=x, b=y, experiment=experiment)[0],
                        'test_seed': s,
                        'split': 'test',
                        **config
                    })
        results = pd.DataFrame(results)
        # plot results
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        plots = len(datasets) + 1
        rows = int(np.ceil(plots / columns))
        fig = {'train': plt.figure(figsize=(4 * columns, 4 * rows), tight_layout=True)}
        if test:
            fig['test'] = plt.figure(figsize=(4 * columns, 4 * rows), tight_layout=True)
        for split, figure in fig.items():
            handles, labels, ax = [], [], None
            names = list(datasets.keys())[::-1]
            for i in np.arange(plots) + 2:
                name = names.pop()
                ax = figure.add_subplot(rows, columns, i, sharex=ax, sharey=ax)
                sns.lineplot(
                    data=results[np.logical_and(results['dataset'] == name, results['split'] == split)],
                    x='noise',
                    y='correlation',
                    hue='metric',
                    style='metric',
                    estimator='mean',
                    errorbar='sd',
                    palette=PALETTE[:len(metrics)],
                    linewidth=2,
                    ax=ax
                )
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()
                ax.set_xlabel('Noise Level $\sigma$')
                ax.set_ylabel('Correlation')
                ax.set_ylim((-0.1, 1.1))
                # plot the original data without noise
                sub_ax = inset_axes(ax, width='30%', height='30%', loc='upper right')
                datasets[name](0.0, 0).plot(ax=sub_ax, color='black')
                sub_ax.set_xticks([])
                sub_ax.set_yticks([])
            # plot the legend
            ax = figure.add_subplot(rows, columns, 1)
            ax.legend(handles, labels, loc='center', labelspacing=1.5, frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
        # plot the time in a different figure
        figure = plt.figure(figsize=(12, 12), tight_layout=True)
        fig['time'] = figure
        ax = figure.gca()
        sns.barplot(
            data=results[results['split'] == 'train'],
            x='metric',
            y='execution',
            hue='metric',
            estimator='mean',
            errorbar='sd',
            palette=PALETTE[:len(metrics)],
            legend=False,
            ax=ax
        )
        ax.set_xlabel(None)
        ax.set_ylabel('Execution Time (s)')
        ax.set_yscale('log')
        # store, print, and plot if necessary
        for extension in formats:
            package = 'experiments.exports'
            for key, figure in fig.items():
                with importlib.resources.path(package, f'correlations_{key}.{extension}') as file:
                    figure.savefig(file, bbox_inches='tight')
        if plot:
            titles = {
                'train': 'Computed Correlations (Train)',
                'test': 'Computed Correlations (Test)',
                'time': 'Execution Times to Compute Correlations',
            }
            for key, figure in fig.items():
                figure.suptitle(titles[key])
                figure.show()
        for figure in fig.values():
            plt.close(figure)

    @staticmethod
    def kernels(datasets: Iterable[Callable[[float], Deterministic]],
                metrics: Dict[str, KernelsHGR],
                tests: int = 30,
                formats: Iterable[str] = ('png',),
                plot: bool = False,
                save_time: int = 60):
        # run experiments
        metrics = {'ORACLE': Oracle(), **metrics}
        datasets_0 = [dataset_fn(0) for dataset_fn in datasets]
        experiments = CorrelationExperiment.doe(
            file_name='correlation',
            save_time=save_time,
            verbose=False,
            dataset={dataset.key: dataset for dataset in datasets_0},
            metric=metrics,
            seed=0
        )
        for dataset_fn, dataset in zip(datasets, datasets_0):
            # build and plot results
            a = dataset.excluded(backend='numpy')
            b = dataset.target(backend='numpy')
            fig, axes = plt.subplot_mosaic(
                mosaic=[['A', 'A', 'data', 'B', 'B'], ['A', 'A', 'hgr', 'B', 'B']],
                figsize=(15, 6),
                tight_layout=True
            )
            fa, gb = {'index': a}, {'index': b}
            # retrieve metric kernels
            for name, metric in metrics.items():
                _, fa_current, gb_current = metric.kernels(a=a, b=b, experiment=experiments[(dataset.key, name)])
                # for all the non-oracle kernels, switch sign to match kernel if necessary
                if name != 'ORACLE':
                    fa_signs = np.sign(fa_current * fa['ORACLE'])
                    fa_current = np.sign(fa_signs.sum()) * fa_current
                    gb_signs = np.sign(gb_current * gb['ORACLE'])
                    gb_current = np.sign(gb_signs.sum()) * gb_current
                fa[name], gb[name] = fa_current, gb_current
            fa, gb = pd.DataFrame(fa).set_index('index'), pd.DataFrame(gb).set_index('index')
            # plot kernels
            sns.set_context('notebook')
            sns.set_style('white')
            for data, kernel, labels in zip([fa, gb], ['A', 'B'], [('a', 'f(a)'), ('b', 'g(b)')]):
                ax = axes[kernel]
                sns.lineplot(
                    data=data,
                    sort=True,
                    estimator=None,
                    palette=PALETTE[:len(metrics)],
                    ax=ax
                )
                ax.set_title(f'{kernel} Kernel')
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1], rotation=0, labelpad=15)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.legend(loc='best')
            # plot data
            ax = axes['data']
            kwargs = dict() if dataset.noise == 0.0 else dict(alpha=0.4, edgecolor='black')
            dataset.plot(ax=ax, color='black', **kwargs)
            ax.set_title('Data')
            ax.set_xlabel('a')
            ax.set_ylabel('b', rotation=0, labelpad=7.5)
            ax.set_xticks([])
            ax.set_yticks([])
            # compute and plot correlations
            correlations = [{
                'metric': metric,
                'split': 'train',
                'hgr': experiments[(dataset.key, metric)].result['correlation']
            } for metric in metrics.keys()]
            for seed in np.arange(tests) + 1:
                dataset_seed = dataset_fn(seed)
                x = dataset_seed.excluded(backend='numpy')
                y = dataset_seed.target(backend='numpy')
                correlations += [{
                    'metric': name,
                    'split': 'test',
                    'hgr': metric.kernels(a=x, b=y, experiment=experiments[(dataset.key, name)])[0]
                } for name, metric in metrics.items()]
            ax = axes['hgr']
            sns.barplot(
                data=pd.DataFrame(correlations),
                y='hgr',
                x='split',
                hue='metric',
                estimator='median',
                errorbar='sd',
                palette=PALETTE[:len(metrics)]
            )
            ax.set_title('Correlation')
            ax.get_legend().set_title(None)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_ylim((0, 1))
            # store and plot if necessary
            for extension in formats:
                name = f'kernels_{dataset.key}.{extension}'
                with importlib.resources.path('experiments.exports', name) as file:
                    fig.savefig(file, bbox_inches='tight')
            if plot:
                config = dataset.configuration
                name = config.pop('name').title()
                info = ', '.join({f'{key}={value}' for key, value in config.items()})
                fig.suptitle(f'Kernels for {name}({info})')
                fig.show()
            plt.close(fig)
