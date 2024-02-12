import importlib.resources
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Iterable, Literal, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns

from experiments import Experiment
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

    def _compute(self) -> Experiment.Result:
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
    def importance(datasets: Dict[str, Dataset],
                   on: Literal['target', 'protected', 'both'] = 'both',
                   top: int = 10,
                   formats: Iterable[str] = ('png',),
                   plot: bool = False,
                   save_time: int = 60):
        targets = []
        if on in ['target', 'both']:
            targets.append(True)
        if on in ['protected', 'both']:
            targets.append(False)
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        # iterate over dataset and protected
        for ds, dataset in datasets.items():
            for target in targets:
                var = dataset.target_name if target else dataset.excluded_name
                # run experiments
                print(f"Running Experiments for Dataset {ds.title()} on variable '{var}':")
                experiments = AnalysisExperiment.doe(
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
                fig = plt.figure(figsize=(16, 9), tight_layout=True)
                ax = fig.gca()
                sns.barplot(data=correlations, color='black', ax=ax)
                ax.set_xticklabels(correlations.index, rotation=45)
                ax.set_ylim((0, 1))
                # store and plot if necessary
                for extension in formats:
                    name = f'importance_{ds}_{var}.{extension}'
                    with importlib.resources.path('experiments.exports', name) as file:
                        fig.savefig(file, bbox_inches='tight')
                if plot:
                    fig.suptitle(f"Importance Analysis for feature '{var}' of {ds.title()} dataset")
                    fig.show()

    @staticmethod
    def causality(datasets: Dict[str, Dataset],
                  on: Literal['target', 'surrogate', 'both'] = 'both',
                  formats: Iterable[str] = ('png',),
                  plot: bool = False,
                  save_time: int = 60):
        targets = []
        if on in ['target', 'both']:
            targets.append(True)
        if on in ['surrogate', 'both']:
            targets.append(False)
        sns.set_context('notebook')
        sns.set_style('whitegrid')
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
                    file_name='analysis',
                    save_time=save_time,
                    verbose=True,
                    dataset=dataset,
                    metric=metrics,
                    features=(f1, f2)
                )
                fig, axes = plt.subplot_mosaic(
                    mosaic=[[True, True, 'data', False, False], [True, True, 'hgr', False, False]],
                    figsize=(15, 6),
                    tight_layout=True
                )
                # plot kernels
                for direct, metric in metrics.items():
                    if direct:
                        t, xl, yl = (f'DIRECT: {f1} --> {f2}', f'K({f1})', f2)
                    else:
                        t, xl, yl = (f'INVERSE: {f2} --> {f1}', f1, f'K({f2})')
                    _, fx, gy = metric.kernels(a=x, b=y, experiment=experiments[direct])
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
                for extension in formats:
                    name = f'causality_{ds}_{f1}_{f2}.{extension}'
                    with importlib.resources.path('experiments.exports', name) as file:
                        fig.savefig(file, bbox_inches='tight')
                if plot:
                    fig.suptitle(f"Causal Analysis for {ds.title()} dataset")
                    fig.show()
