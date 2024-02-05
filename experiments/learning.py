import importlib.resources
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.experiment import Experiment
from src.datasets import Dataset
from src.hgr import HGR
from src.learning import MultiLayerPerceptron, Data, Callback, Loss, Accuracy

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

EXTERNAL: List[str] = [
    'train_predictions',
    'train_inputs',
    'train_target',
    'val_predictions',
    'val_inputs',
    'val_target',
    'model'
]
"""The names of the results to be stored in the external file."""

SEED: int = 0
"""The random seed used in the experiment."""

FOLDS: int = 1  # TODO: replace with 5
"""The number of folds for k-fold cross-validation."""

MINI_BATCH: int = 128
"""The size of a mini batch."""

MINI_EPOCHS: int = 500
"""The number of epochs used during training with mini batches."""

FULL_EPOCHS: int = 200
"""The number of epochs used during training full batch."""


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class LearningExperiment(Experiment):
    """An experiment where a neural network is constrained so that the correlation between a protected variable and the
    target is reduced."""

    dataset: Dataset = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The dataset used in the experiment."""

    fold: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The fold that is used for training the model."""

    metric: Optional[HGR] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The metric to be used as penalty, or None for unconstrained model."""

    _alpha: Optional[float] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The alpha value for the penalizer."""

    full_batch: bool = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """Whether to train the model full batch or with mini batches."""

    entity: Optional[str] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """Either the Weights & Biases entity, or None for no logging."""

    def _compute(self) -> Experiment.Result:
        pl.seed_everything(SEED, workers=True)
        external = f'{self.key}.pkl'
        # retrieve train and validation data from splits and set parameters
        trn, val = self.dataset.data(folds=FOLDS, seed=SEED)[self.fold]
        trn_data = Data(x=trn[self.dataset.input_names], y=trn[self.dataset.target_name])
        val_data = Data(x=val[self.dataset.input_names], y=val[self.dataset.target_name])
        epochs = FULL_EPOCHS if self.full_batch else MINI_EPOCHS
        batch = len(trn) if self.full_batch else MINI_BATCH
        # build model
        model = MultiLayerPerceptron(
            units=[len(self.dataset.input_names), *self.dataset.units],
            classification=self.dataset.classification,
            feature=self.dataset.excluded_index,
            metric=self.metric,
            alpha=self.alpha
        )
        # build trainer and callback
        callback = Callback(experiment=self)
        trainer = pl.Trainer(
            min_epochs=epochs,
            max_epochs=epochs,
            check_val_every_n_epoch=epochs + 1,
            callbacks=[callback],
            deterministic=True,
            enable_progress_bar=True,
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=False
        )
        # run fitting
        start = time.time()
        trainer.fit(
            model=model,
            train_dataloaders=DataLoader(trn_data, batch_size=batch, num_workers=4, persistent_workers=True),
            val_dataloaders=DataLoader(val_data, batch_size=len(val), num_workers=4, persistent_workers=True)
        )
        gap = time.time() - start
        # store internal and external results
        int_results, ext_results = {}, {}
        for key, value in callback.results.items():
            structure = int_results if key in EXTERNAL else ext_results
            structure[key] = value
        with importlib.resources.path('experiments.results', external) as path:
            assert not path.exists(), f"File '{self.key}' is already present in package 'experiments.results'"
        with open(path, 'wb') as file:
            pickle.dump(ext_results, file=file)
        # return experiments
        return Experiment.Result(timestamp=start, execution=gap, external=external, **int_results)

    @property
    def alpha(self) -> Optional[float]:
        """The alpha value for the penalizer (defaults to None when no penalty is used)."""
        return None if self.metric == 'unc' else self._alpha

    @property
    def name(self) -> str:
        return 'learning'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(
            experiment=self.name,
            dataset=self.dataset.configuration,
            fold=self.fold,
            metric={'name': 'unconstrained'} if self.metric is None else self.metric.configuration,
            alpha=self.alpha,
            full_batch=self.full_batch
        )

    @property
    def key(self) -> str:
        metric = None if self.metric is None else self.metric.key
        return f'{self.name}_{self.dataset.key}_{metric}_{self.alpha}_{self.full_batch}_{self.fold}'

    @staticmethod
    def learning(datasets: Dict[str, Dataset],
                 metrics: Dict[str, HGR],
                 alpha: Optional[float] = None,
                 full_batch: bool = True,
                 entity: Optional[str] = None,
                 formats: Iterable[str] = ('png',),
                 plot: bool = False):
        # run experiments
        metrics = {'UNC': None, **metrics}
        experiments = LearningExperiment.doe(
            file_name='learning',
            save_time=0,
            verbose=True,
            dataset=datasets,
            metric=metrics,
            fold=list(range(FOLDS)),
            _alpha=alpha,
            full_batch=full_batch,
            entity=entity
        )
        # get results
        results = {dataset: [] for dataset in datasets.keys()}
        for (dataset, metric, fold), experiment in tqdm(experiments.items(), desc='Computing Metrics'):
            exp = experiment.result
            # retrieve input data
            data = datasets[dataset]
            xtr = exp['train_inputs'].numpy(force=True)
            ytr = exp['train_target'].numpy(force=True).flatten()
            xvl = exp['val_inputs'].numpy(force=True)
            yvl = exp['val_target'].numpy(force=True).flatten()
            # compute metrics for each epoch
            kpis = [
                Loss(classification=data.classification),
                Accuracy(classification=data.classification)
            ]
            res = results[dataset]
            for epoch in range(exp['epochs']):
                info = dict(dataset=dataset, metric=metric, fold=fold, epoch=epoch)
                ptr = exp['train_predictions'][epoch].numpy(force=True).flatten()
                pvl = exp['val_predictions'][epoch].numpy(force=True).flatten()
                # res.append({**info, 'kpi': 'alpha', 'split': 'Learning', 'value': exp['alpha'][epoch].numpy().item()})
                # res.append({**info, 'kpi': 'time', 'split': 'Learning', 'value': exp['time'][epoch]})
                for kpi in kpis:
                    res.append({**info, 'kpi': kpi.name, 'split': 'Train', 'value': kpi(x=xtr, y=ytr, p=ptr)})
                    res.append({**info, 'kpi': kpi.name, 'split': 'Val', 'value': kpi(x=xvl, y=yvl, p=pvl)})
        # plot results
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        for dataset, group in results.items():
            group = pd.DataFrame(group)
            kpis = group['kpi'].unique()
            fig, axes = plt.subplots(2, len(kpis), figsize=(5 * len(kpis), 8), tight_layout=True)
            for i, split in enumerate(['Train', 'Val']):
                for j, kpi in enumerate(kpis):
                    sns.lineplot(
                        data=group[np.logical_and(group['split'] == split, group['kpi'] == kpi)],
                        x='epoch',
                        y='value',
                        hue='metric',
                        style='metric',
                        estimator='mean',
                        errorbar='sd',
                        palette=PALETTE[:len(metrics)],
                        linewidth=2,
                        ax=axes[i, j]
                    )
                    axes[i, j].set_ylabel(f'{split} {kpi}')
            # store, print, and plot if necessary
            for extension in formats:
                name = f'learning_{dataset}_{full_batch}.{extension}'
                with importlib.resources.path('experiments.exports', name) as file:
                    fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f"Learning History for {dataset.title()} ({'Full' if full_batch else 'Mini'} Batch)")
                fig.show()
            plt.close(fig)
