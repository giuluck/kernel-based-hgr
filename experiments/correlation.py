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

    def _compute(self, folder: str) -> Experiment.Result:
        pl.seed_everything(self.seed, workers=True)
        start = time.time()
        a = self.dataset.excluded(backend='numpy')
        b = self.dataset.target(backend='numpy')
        hgr, additional = self.metric.correlation(a=a, b=b)
        gap = time.time() - start
        # store external files only for NN kernels, in the other cases include the additional results in the object
        if isinstance(self.metric, AdversarialHGR):
            external = os.path.join('correlation', f'{self.key}.pkl')
            filepath = os.path.join(folder, 'results', external)
            # overwrite files rather than asserting that they are not present since an abrupt interruption of the
            # DoE might cause leaking external files to be stored while the original results are not
            if os.path.exists(filepath):
                print(f"WARNING: overwriting file '{self.key}' since it is already in the expected folder")
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
    def monotonicity(folder: str,
                     datasets: Iterable[Dataset],
                     degrees_a: Iterable[int] = (1, 2, 3, 4, 5, 6, 7),
                     degrees_b: Iterable[int] = (1, 2, 3, 4, 5, 6, 7),
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     extensions: Iterable[str] = ('png',),
                     plot: bool = False,
                     save_time: int = 60):
        # run experiments
        experiments = CorrelationExperiment.doe(
            folder=folder,
            file_name='correlation',
            save_time=save_time,
            verbose=False,
            seed=0,
            dataset={dataset.key: dataset for dataset in datasets},
            metric={(da, db): DoubleKernelHGR(degree_a=da, degree_b=db) for da in degrees_a for db in degrees_b}
        )
        # plot results
        sns.set(context='poster', style='whitegrid', font_scale=1.8)
        for dataset in datasets:
            # build results
            results = np.zeros((len(degrees_a), len(degrees_b)))
            for i, da in enumerate(degrees_a):
                for j, db in enumerate(degrees_b):
                    results[i, j] = experiments[(dataset.key, (da, db))].result['correlation']
            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            ax = fig.gca()
            col = ax.imshow(results.transpose()[::-1], cmap=plt.colormaps['Greys'], vmin=vmin, vmax=vmax)
            fig.colorbar(col, ax=ax)
            ax.set_xlabel('h')
            ax.set_xticks(np.arange(len(degrees_a) + 1) - 0.5)
            ax.set_xticklabels([''] * (len(degrees_a) + 1))
            ax.set_xticks(np.arange(len(degrees_a)), minor=True)
            ax.set_xticklabels(degrees_a, minor=True)
            ax.set_ylabel('k', rotation=0, labelpad=20)
            ax.set_yticks(np.arange(len(degrees_b) + 1) - 0.5)
            ax.set_yticklabels([''] * (len(degrees_b) + 1))
            ax.set_yticks(np.arange(len(degrees_b)), minor=True)
            ax.set_yticklabels(degrees_b[::-1], minor=True)
            ax.grid(True, which='major')
            # store, print, and plot if necessary
            for extension in extensions:
                name = f'monotonicity_{dataset.key}.{extension}'
                file = os.path.join(folder, 'exports', name)
                fig.savefig(file, bbox_inches='tight')
            if plot:
                config = dataset.configuration
                name = config.pop('name').title()
                info = ', '.join({f'{key}={value}' for key, value in config.items()})
                fig.suptitle(f'Monotonicity in {name}({info})')
                fig.show()
            plt.close(fig)

    @staticmethod
    def correlations(folder: str,
                     datasets: Dict[str, Callable[[float, int], Deterministic]],
                     metrics: Dict[str, HGR],
                     noises: Iterable[float] = np.linspace(0.0, 3.0, num=16, endpoint=True).round(2),
                     noise_seeds: Iterable[int] = range(10),
                     algorithm_seeds: Iterable[int] = range(10),
                     test: bool = False,
                     columns: int = 2,
                     extensions: Iterable[str] = ('png',),
                     plot: bool = False,
                     save_time: int = 60):
        assert len(noise_seeds) > 1 or not test, "Tests cannot be performed only if more than one data seed is passed"
        # run experiments
        metrics = {'ORACLE': Oracle(), **metrics}
        experiments = CorrelationExperiment.doe(
            folder=folder,
            file_name='correlation',
            save_time=save_time,
            verbose=False,
            dataset={(k, n, s): fn(n, s) for k, fn in datasets.items() for n in noises for s in noise_seeds},
            metric=metrics,
            seed=list(algorithm_seeds)
        )
        # build results
        results = []
        for key, experiment in tqdm(experiments.items(), desc='Storing Correlations'):
            dataset, noise, seed = key[0]
            config = dict(dataset=dataset, noise=noise, metric=key[1], data_seed=seed, algorithm_seed=key[2])
            if not test:
                results.append({
                    'correlation': experiment.result['correlation'],
                    'execution': experiment.result['execution'],
                    **config
                })
            # build results for test data (use all the data seeds but the training one)
            elif isinstance(experiment.metric, KernelsHGR):
                for s in noise_seeds:
                    if s == seed:
                        continue
                    dataset_seed = datasets[dataset](noise, s)
                    x = dataset_seed.excluded(backend='numpy')
                    y = dataset_seed.target(backend='numpy')
                    results.append({
                        'correlation': experiment.metric.kernels(a=x, b=y, folder=folder, experiment=experiment)[0],
                        'test_seed': s,
                        **config
                    })
        results = pd.DataFrame(results)
        # plot results
        sns.set(context='poster', style='whitegrid', font_scale=1.7)
        # plot from 1 to D for test, while from 2 to D + 1 for train to leave the first subplot for the training times
        plots = np.arange(len(datasets)) + (1 if test else 2)
        rows = int(np.ceil(plots[-1] / columns))
        fig = plt.figure(figsize=(9 * columns, 8.5 * rows), tight_layout=True)
        handles, labels = [], []
        names = list(datasets.keys())[::-1]
        for i in plots:
            name = names.pop()
            ax = fig.add_subplot(rows, columns, i)
            sns.lineplot(
                data=results[results['dataset'] == name],
                x='noise',
                y='correlation',
                hue='metric',
                style='metric',
                estimator='mean',
                errorbar='sd',
                palette=PALETTE[:len(metrics)],
                linewidth=3,
                ax=ax
            )
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            ax.set_xlabel('Noise Level $\sigma$')
            ax.set_ylabel('Correlation')
            ax.set_ylim((-0.1, 1.1))
            # plot the original data without noise
            sub_ax = inset_axes(ax, width='30%', height='30%', loc='upper right')
            datasets[name](0.0, 0).plot(ax=sub_ax, linewidth=2, color='black')
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
        # if train, plot times in the first subplot
        if not test:
            ax = fig.add_subplot(rows, columns, 1)
            sns.barplot(
                data=results,
                x='metric',
                y='execution',
                hue='metric',
                estimator='mean',
                errorbar='sd',
                linewidth=3,
                palette=PALETTE[:len(metrics)],
                legend=False,
                ax=ax
            )
            for patch, handle in zip(ax.patches, handles):
                patch.set_linestyle(handle.__dict__['_dash_pattern'])
                color = patch.get_facecolor()
                patch.set_edgecolor(color)
                # fake transparency to white
                color = tuple([0.8 * c + 0.2 for c in color[:3]] + [1])
                patch.set_facecolor(color)
            ax.set_xticks(labels, labels=labels, rotation=45)
            ax.set_xlabel(None)
            ax.set_ylabel('Execution Time (s)')
            ax.set_yscale('log')
        # store, print, and plot if necessary
        key = 'test' if test else 'train'
        for extension in extensions:
            file = os.path.join(folder, 'exports', f'correlations_{key}.{extension}')
            fig.savefig(file, bbox_inches='tight')
        if plot:
            fig.suptitle(f'Computed Correlations ({key.title()})')
            fig.show()
        plt.close(fig)

    @staticmethod
    def kernels(folder: str,
                datasets: Iterable[Callable[[float], Deterministic]],
                metrics: Dict[str, KernelsHGR],
                tests: int = 30,
                extensions: Iterable[str] = ('png',),
                plot: bool = False,
                save_time: int = 60):
        # run experiments
        metrics = {'ORACLE': Oracle(), **metrics}
        datasets_0 = [dataset_fn(0) for dataset_fn in datasets]
        experiments = CorrelationExperiment.doe(
            folder=folder,
            file_name='correlation',
            save_time=save_time,
            verbose=False,
            dataset={dataset.key: dataset for dataset in datasets_0},
            metric=metrics,
            seed=0
        )
        sns.set(context='poster', style='white', font_scale=1.5)
        for dataset_fn, dataset in zip(datasets, datasets_0):
            # build and plot results
            a = dataset.excluded(backend='numpy')
            b = dataset.target(backend='numpy')
            fig, axes = plt.subplot_mosaic(
                mosaic=[['A', 'hgr'], ['data', 'B']],
                figsize=(16, 16),
                tight_layout=True
            )
            fa, gb = {'index': a}, {'index': b}
            # retrieve metric kernels
            for name, metric in metrics.items():
                _, fa_current, gb_current = metric.kernels(
                    a=a,
                    b=b,
                    folder=folder,
                    experiment=experiments[(dataset.key, name)]
                )
                # for all the non-oracle kernels, switch sign to match kernel if necessary
                if name != 'ORACLE':
                    fa_signs = np.sign(fa_current * fa['ORACLE'])
                    fa_current = np.sign(fa_signs.sum()) * fa_current
                    gb_signs = np.sign(gb_current * gb['ORACLE'])
                    gb_current = np.sign(gb_signs.sum()) * gb_current
                fa[name], gb[name] = fa_current, gb_current
            fa, gb = pd.DataFrame(fa).set_index('index'), pd.DataFrame(gb).set_index('index')
            # plot kernels
            handles = None
            for data, kernel, labels in zip([fa, gb], ['A', 'B'], [('a', 'f(a)'), ('b', 'g(b)')]):
                ax = axes[kernel]
                sns.lineplot(
                    data=data,
                    sort=True,
                    estimator=None,
                    linewidth=3,
                    palette=PALETTE[:len(metrics)],
                    ax=ax
                )
                handles, _ = ax.get_legend_handles_labels()
                ax.set_title(f'{kernel} Kernel')
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1], rotation=0, labelpad=37)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.legend(loc='best')
            # plot data
            ax = axes['data']
            kwargs = dict() if dataset.noise == 0.0 else dict(alpha=0.4, edgecolor='black', s=30)
            dataset.plot(ax=ax, color='black', **kwargs)
            ax.set_title('Data')
            ax.set_xlabel('a')
            ax.set_ylabel('b', rotation=0, labelpad=20)
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
                    'hgr': metric.kernels(a=x, b=y, folder=folder, experiment=experiments[(dataset.key, name)])[0]
                } for name, metric in metrics.items()]
            ax = axes['hgr']
            sns.barplot(
                data=pd.DataFrame(correlations),
                y='hgr',
                x='split',
                hue='metric',
                estimator='mean',
                errorbar='sd',
                linewidth=3,
                palette=PALETTE[:len(metrics)],
                legend=None,
                ax=ax
            )
            for patches, handle in zip(np.reshape(ax.patches, (-1, 2)), handles):
                for patch in patches:
                    patch.set_linestyle(handle.__dict__['_dash_pattern'])
                    color = patch.get_facecolor()
                    patch.set_edgecolor(color)
                    # fake transparency to white
                    color = tuple([0.8 * c + 0.2 for c in color[:3]] + [1])
                    patch.set_facecolor(color)
            ax.set_title('Correlation')
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            # store and plot if necessary
            for extension in extensions:
                name = f'kernels_{dataset.key}.{extension}'
                file = os.path.join(folder, 'exports', name)
                fig.savefig(file, bbox_inches='tight')
            if plot:
                config = dataset.configuration
                name = config.pop('name').title()
                info = ', '.join({f'{key}={value}' for key, value in config.items()})
                fig.suptitle(f'Kernels for {name}({info})')
                fig.show()
            plt.close(fig)
