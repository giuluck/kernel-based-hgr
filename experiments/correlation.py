import importlib.resources
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from experiments.experiment import Experiment
from src.datasets import Dataset
from src.datasets.deterministic import Deterministic
from src.hgr import DoubleKernelHGR, HGR, KernelBasedHGR, AdversarialHGR, Oracle

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


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class CorrelationExperiment(Experiment):
    """An experiment where the correlation between two variables is computed."""

    def _compute(self) -> Experiment.Result:
        pl.seed_everything(self.seed, workers=True)
        start = time.time()
        a = self.dataset.excluded(backend='numpy')
        b = self.dataset.target(backend='numpy')
        hgr, additional = self.metric.correlation(a=a, b=b)
        gap = time.time() - start
        # store external files only for NN kernels, in the other cases include the additional results in the object
        if isinstance(self.metric, AdversarialHGR):
            external = f'{self.key}.pkl'
            with importlib.resources.path('experiments.results', external) as path:
                # overwrite files rather than asserting that they are not present since an abrupt interruption of the
                # DoE might cause leaking external files to be stored while the original results are not
                if path.exists():
                    print(f"Overwriting file '{self.key}' since it is already present in package 'experiments.results'")
            with open(path, 'wb') as file:
                pickle.dump(additional, file=file)
            return Experiment.Result(timestamp=start, execution=gap, external=external, correlation=hgr)
        else:
            return Experiment.Result(timestamp=start, execution=gap, external=None, correlation=hgr, **additional)

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
            for i, da in enumerate(degrees_a):
                for j, db in enumerate(degrees_b):
                    results[i, j] = experiments[(dataset.key, (da, db))].result['correlation']
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
                     columns: int = 2,
                     legend: int = 1,
                     formats: Iterable[str] = ('png',),
                     plot: bool = False,
                     save_time: int = 60):
        # run experiments
        metrics = {'ORACLE': Oracle, **metrics}
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
            with importlib.resources.path(package, f'correlations_data.{extension}') as file:
                fig_data.savefig(file, bbox_inches='tight')
            with importlib.resources.path(package, f'correlations_time.{extension}') as file:
                fig_time.savefig(file, bbox_inches='tight')
        if plot:
            fig_data.suptitle('Computed Correlations')
            fig_data.show()
            fig_time.suptitle('Execution Times to Compute Correlations')
            fig_time.show()

    @staticmethod
    def kernels(datasets: Iterable[Deterministic],
                metrics: Dict[str, HGR],
                tests: int = 30,
                formats: Iterable[str] = ('png',),
                plot: bool = False,
                save_time: int = 60):
        def standardize(vv: np.ndarray) -> np.ndarray:
            return (vv - vv.mean()) / (vv.std(ddof=0))

        def hgr(xx: CorrelationExperiment, aa: np.ndarray, bb: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
            mtr = xx.metric
            if isinstance(mtr, Oracle):
                ff_aa = standardize(mtr.dataset.f(aa))
                gg_bb = standardize(mtr.dataset.g(bb))
            elif isinstance(mtr, AdversarialHGR):
                aa = torch.tensor(aa, dtype=torch.float32).reshape((-1, 1))
                bb = torch.tensor(bb, dtype=torch.float32).reshape((-1, 1))
                ff_aa = standardize(xx.result['f'](aa).numpy(force=True).flatten())
                gg_bb = standardize(xx.result['g'](bb).numpy(force=True).flatten())
            elif isinstance(mtr, KernelBasedHGR):
                # center the kernel with respect to the training data
                a_ref, b_ref = xx.dataset.excluded(backend='numpy'), xx.dataset.target(backend='numpy')
                ff = np.stack([aa ** d - np.mean(a_ref ** d) for d in np.arange(mtr.degree_a) + 1], axis=1)
                gg = np.stack([bb ** d - np.mean(b_ref ** d) for d in np.arange(mtr.degree_b) + 1], axis=1)
                ff_aa = standardize(ff @ xx.result['alpha'])
                gg_bb = standardize(gg @ xx.result['beta'])
            else:
                raise AssertionError(f"Unsupported metric {mtr}")
            return abs(np.mean(ff_aa * gg_bb)), ff_aa, gg_bb

        # run experiments
        metrics = {'ORACLE': Oracle, **metrics}
        experiments = CorrelationExperiment.doe(
            file_name='correlation',
            save_time=save_time,
            dataset={dataset.key: dataset for dataset in datasets},
            metric=metrics,
            seed=0
        )
        for dataset in datasets:
            # build and plot results
            a = dataset.excluded(backend='numpy')
            b = dataset.target(backend='numpy')
            fig, axes = plt.subplot_mosaic(
                mosaic=[['A', 'A', 'data', 'B', 'B'], ['A', 'A', 'hgr', 'B', 'B']],
                figsize=(15, 6),
                tight_layout=True
            )
            fa, gb = {'index': a}, {'index': b}
            # retrieve metric kernels (switch sign to match the same orientation of the optimal kernels)
            anchors = np.random.default_rng(0).choice(range(len(a)), size=10, replace=False)
            for name, metric in metrics.items():
                _, f_a, g_b = hgr(xx=experiments[(dataset.key, name)], aa=a, bb=b)
                if name != 'ORACLE':
                    kernel_anchors = f_a[anchors]
                    oracle_anchors = fa['ORACLE'][anchors]
                    anchor_signs = np.sign(kernel_anchors * oracle_anchors)
                    if anchor_signs.sum() < 0:
                        f_a, g_b = -f_a, -g_b
                fa[name], gb[name] = f_a, g_b
            fa, gb = pd.DataFrame(fa).set_index('index'), pd.DataFrame(gb).set_index('index')
            # plot kernels
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
            kwargs = dict() if dataset.noise == 0.0 else dict(alpha=0.6, edgecolor='black')
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
                x, y = dataset.from_seed(seed=seed)
                correlations += [{
                    'metric': metric,
                    'split': 'test',
                    'hgr': hgr(xx=experiments[(dataset.key, metric)], aa=x, bb=y)[0]
                } for metric in metrics.keys()]
            ax = axes['hgr']
            sns.barplot(
                data=pd.DataFrame(correlations),
                y='hgr',
                x='split',
                hue='metric',
                estimator='mean',
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
