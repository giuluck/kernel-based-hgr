import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Iterable, Literal, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr

from experiments.experiment import Experiment
from src.datasets import Dataset
from src.hgr import DoubleKernelHGR

SEED: int = 0
"""The random seed used in the experiment."""


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class AnalysisExperiment(Experiment):
    """An experiment where the correlation between two variables is computed."""

    dataset: Dataset = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The dataset used in the experiment."""

    metric: DoubleKernelHGR = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The HGR metric used in the experiment."""

    features: Tuple[str, str] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The features to analyze."""

    def _compute(self, folder: str) -> Experiment.Result:
        pl.seed_everything(SEED, workers=True)
        start = time.time()
        a, b = self.features
        hgr, additional = self.metric.correlation(a=self.dataset[a].values, b=self.dataset[b].values)
        gap = time.time() - start
        return Experiment.Result(timestamp=start, execution=gap, hgr=hgr, external=None, **additional)

    @property
    def name(self) -> str:
        return 'correlation'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(
            experiment=self.name,
            dataset=self.dataset.configuration,
            metric=self.metric.configuration,
            features=self.features
        )

    @property
    def key(self) -> str:
        return f'{self.name}_{self.dataset.key}_{self.metric.key}_{self.features[0]}_{self.features[1]}'

    @staticmethod
    def importance(folder: str,
                   datasets: Dict[str, Dataset],
                   on: Literal['target', 'protected', 'both'] = 'both',
                   top: int = 10,
                   extensions: Iterable[str] = ('png',),
                   plot: bool = False,
                   save_time: int = 60):
        targets = []
        if on in ['target', 'both']:
            targets.append(True)
        if on in ['protected', 'both']:
            targets.append(False)
        sns.set(context='poster', style='whitegrid', font_scale=2.5)
        # iterate over dataset and protected
        for ds, dataset in datasets.items():
            for target in targets:
                var = dataset.target_name if target else dataset.excluded_name
                # run experiments
                print(f"Running Experiments for Dataset {ds.title()} on variable '{var}':")
                experiments = AnalysisExperiment.doe(
                    folder=folder,
                    file_name='analysis',
                    save_time=save_time,
                    verbose=False,
                    dataset=dataset,
                    metric=DoubleKernelHGR(),
                    features={ft: (var, ft) for ft in dataset.input_names if ft != var}
                )
                print()
                # plot results
                correlations = {ft: exp.result['hgr'] for ft, exp in experiments.items()}
                correlations = pd.Series(correlations).sort_values(ascending=False).iloc[:top]
                fig = plt.figure(figsize=(13, 12))
                ax = fig.gca()
                sns.barplot(data=correlations, color='black', orient='h', ax=ax)
                ax.set_xlim((0, 1))
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(axis='x', which='major', length=0)
                ax.tick_params(axis='y', which='major', pad=10)
                # store and plot if necessary
                for extension in extensions:
                    name = f'importance_{ds}_{var}.{extension}'
                    file = os.path.join(folder, 'exports', name)
                    fig.savefig(file, bbox_inches='tight')
                if plot:
                    fig.suptitle(f"Importance Analysis for feature '{var}' of {ds.title()} dataset")
                    fig.show()

    @staticmethod
    def causality(folder: str,
                  datasets: Dict[str, Dataset],
                  on: Literal['target', 'surrogate', 'both'] = 'both',
                  extensions: Iterable[str] = ('png',),
                  plot: bool = False,
                  save_time: int = 60):
        targets = []
        if on in ['target', 'both']:
            targets.append(True)
        if on in ['surrogate', 'both']:
            targets.append(False)
        sns.set(context='poster', style='whitegrid', font_scale=1.5)
        # iterate over dataset and protected
        for ds, dataset in datasets.items():
            for target in targets:
                if target:
                    f1, f2 = dataset.excluded_name, dataset.target_name
                elif hasattr(dataset, 'surrogate_name'):
                    f1, f2 = dataset.surrogate_name, dataset.excluded_name
                else:
                    continue
                # run experiment
                x = dataset[f1].values
                y = dataset[f2].values
                metrics = {
                    True: DoubleKernelHGR(degree_b=1),
                    False: DoubleKernelHGR(degree_a=1)
                }
                experiments = AnalysisExperiment.doe(
                    folder=folder,
                    file_name='analysis',
                    save_time=save_time,
                    verbose=True,
                    dataset=dataset,
                    metric=metrics,
                    features=(f1, f2)
                )
                fig, axes = plt.subplot_mosaic(
                    mosaic=[[True, True, 'data', False, False], [True, True, 'hgr', False, False]],
                    figsize=(30, 12),
                    tight_layout=True
                )
                # plot kernels
                for direct, metric in metrics.items():
                    if direct:
                        t, xl, yl = (f'DIRECT: {f1} --> {f2}', f'K({f1})', f2)
                    else:
                        t, xl, yl = (f'INVERSE: {f2} --> {f1}', f1, f'K({f2})')
                    _, fx, gy = metric.kernels(a=x, b=y, folder=folder, experiment=experiments[direct])
                    ax = axes[direct]
                    sns.scatterplot(
                        x=fx,
                        y=y,
                        alpha=0.4,
                        color='black',
                        edgecolor='black',
                        ax=ax
                    )
                    ax.set_xlabel(xl)
                    ax.set_ylabel(yl)
                    ax.set_title(t)
                # plot original data
                ax = axes['data']
                sns.scatterplot(
                    x=x,
                    y=y,
                    alpha=0.4,
                    color='black',
                    edgecolor='black',
                    ax=ax
                )
                # plot correlations
                ax = axes['hgr']
                correlations = {f'DIRECT' if key else f'INVERSE': exp.result['hgr'] for key, exp in experiments.items()}
                sns.barplot(data=pd.Series(correlations), color='black', ax=ax)
                ax.set_ylim((0, 1))
                # store and plot if necessary
                for extension in extensions:
                    name = f'causality_{ds}_{f1}_{f2}.{extension}'
                    file = os.path.join(folder, 'exports', name)
                    fig.savefig(file, bbox_inches='tight')
                if plot:
                    fig.suptitle(f"Causal Analysis for {ds.title()} dataset")
                    fig.show()

    @staticmethod
    def example(folder: str,
                dataset: Dataset,
                degree_a: int = 2,
                degree_b: int = 2,
                extensions: Iterable[str] = ('png',),
                plot: bool = False):
        # compute correlations and kernels
        a, b = dataset.excluded(backend='numpy'), dataset.target(backend='numpy')
        hgr, kernels = DoubleKernelHGR(degree_a=degree_a, degree_b=degree_b).correlation(a, b)
        fa = DoubleKernelHGR.kernel(a, degree=degree_a, use_torch=False) @ kernels['alpha']
        gb = DoubleKernelHGR.kernel(b, degree=degree_b, use_torch=False) @ kernels['beta']
        # build canvas
        sns.set(context='poster', style='white', font_scale=1.3)
        fig = plt.figure(figsize=(20, 10))
        ax = fig.gca()
        ax.axis('off')
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        # build axes
        axes = {
            'data': ('center left', 'a', 'b', 14, f'Correlation: {abs(pearsonr(a, b)[0]):.3f}'),
            'fa': ('upper center', 'a', 'f(a)', 30, f"$\\alpha$ = {kernels['alpha'].round(2)}"),
            'gb': ('lower center', 'b', 'g(b)', 34, f"$\\beta$ = {kernels['beta'].round(2)}"),
            'proj': ('center right', 'f(a)', 'g(b)', 34, f'Correlation: {hgr:.3f}')
        }
        for key, (loc, xl, yl, lp, tl) in axes.items():
            x = inset_axes(ax, width='20%', height='40%', loc=loc)
            x.set_title(tl, pad=12)
            x.set_xlabel(xl, labelpad=8)
            x.set_ylabel(yl, rotation=0, labelpad=lp)
            x.set_xticks([])
            x.set_yticks([])
            axes[key] = x
        # build arrows
        ax.arrow(0.23, 0.57, 0.14, 0.1, color='black', linewidth=2, head_width=0.015)
        ax.arrow(0.23, 0.43, 0.14, -0.1, color='black', linewidth=2, head_width=0.015)
        ax.arrow(0.62, 0.70, 0.14, -0.1, color='black', linewidth=2, head_width=0.015)
        ax.arrow(0.62, 0.30, 0.14, 0.1, color='black', linewidth=2, head_width=0.015)
        # plot data, kernels, and projections
        dataset.plot(ax=axes['data'], color='black', edgecolor='black', alpha=0.6, s=10)
        sns.lineplot(x=a, y=fa, sort=True, linewidth=2, color='black', ax=axes['fa'])
        sns.lineplot(x=b, y=gb, sort=True, linewidth=2, color='black', ax=axes['gb'])
        axes['proj'].scatter(fa, gb, color='black', edgecolor='black', alpha=0.6, s=10)
        # store and plot if necessary
        for extension in extensions:
            name = f'example.{extension}'
            exports = os.path.join(folder, 'exports')
            os.makedirs(exports, exist_ok=True)
            file = os.path.join(exports, name)
            fig.savefig(file, bbox_inches='tight')
        if plot:
            fig.show()
