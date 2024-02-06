import importlib.resources
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.experiment import Experiment
from src.datasets import Dataset
from src.hgr import HGR
from src.learning import MultiLayerPerceptron, Data, ResultsCallback, ProgressCallback, Loss, Accuracy

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

PROJECT: str = 'kernel-based-hgr'
"""The name of the wandb project."""

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

MINI_BATCH: int = 128
"""The size of a mini batch."""

MINI_EPOCHS: int = 100
"""The number of epochs used during training with mini batches."""

FULL_EPOCHS: int = 300
"""The number of epochs used during training full batch."""


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class LearningExperiment(Experiment):
    """An experiment where a neural network is constrained so that the correlation between a protected variable and the
    target is reduced."""

    dataset: Dataset = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The dataset used in the experiment."""

    fold: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The fold that is used for training the model."""

    folds: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The number of folds for k-fold cross-validation."""

    metric: Optional[HGR] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The metric to be used as penalty, or None for unconstrained model."""

    alpha: Optional[float] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The alpha value for the penalizer."""

    units: Optional[Iterable[int]] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The number of hidden units used to build the neural model, or None to use the dataset default value."""

    batch: Optional[int] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The batch size used during training, or None to train full batch."""

    epochs: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The number of training epochs."""

    wandb: bool = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """Whether to log on Weights & Biases."""

    def __post_init__(self):
        assert self.metric is not None or self.alpha is None, "If metric=None, alpha must be None as well."

    def _compute(self) -> Experiment.Result:
        pl.seed_everything(SEED, workers=True)
        external = f'{self.key}.pkl'
        # retrieve train and validation data from splits and set parameters
        trn, val = self.dataset.data(folds=self.folds, seed=SEED)[self.fold]
        trn_data = Data(x=trn[self.dataset.input_names], y=trn[self.dataset.target_name])
        val_data = Data(x=val[self.dataset.input_names], y=val[self.dataset.target_name])
        # build model
        units = self.dataset.units if self.units is None else self.units
        model = MultiLayerPerceptron(
            units=[len(self.dataset.input_names), *units],
            classification=self.dataset.classification,
            feature=self.dataset.excluded_index,
            metric=self.metric,
            alpha=self.alpha
        )
        # build trainer and callback
        callback = ResultsCallback()
        if self.wandb:
            logger = WandbLogger(
                project=PROJECT,
                name=self.key,
                log_model='all'
            )
            logger.experiment.config.update(self.configuration)
        else:
            logger = None
        trainer = pl.Trainer(
            deterministic=True,
            min_epochs=self.epochs,
            max_epochs=self.epochs,
            callbacks=[callback, ProgressCallback()],
            logger=logger,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            enable_progress_bar=True,
            enable_checkpointing=False,
            enable_model_summary=False
        )
        # run fitting
        start = time.time()
        batch_size = len(trn) if self.batch is None else self.batch
        trainer.fit(
            model=model,
            train_dataloaders=DataLoader(trn_data, batch_size=batch_size, num_workers=4, persistent_workers=True),
            val_dataloaders=DataLoader(val_data, batch_size=len(val), num_workers=4, persistent_workers=True)
        )
        gap = time.time() - start
        # store internal and external results
        int_results, ext_results = {}, {}
        for key, value in callback.results.items():
            structure = ext_results if key in EXTERNAL else int_results
            structure[key] = value
        with importlib.resources.path('experiments.results', external) as path:
            assert not path.exists(), f"File '{self.key}' is already present in package 'experiments.results'"
        with open(path, 'wb') as file:
            pickle.dump(ext_results, file=file)
        # return experiments
        return Experiment.Result(timestamp=start, execution=gap, external=external, **int_results)

    @property
    def name(self) -> str:
        return 'learning'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(
            experiment=self.name,
            dataset=self.dataset.configuration,
            fold=self.fold,
            folds=self.folds,
            metric={'name': 'unconstrained'} if self.metric is None else self.metric.configuration,
            alpha=self.alpha,
            units=self.units,
            batch=self.batch,
            epochs=self.epochs
        )

    @property
    def key(self) -> str:
        metric = None if self.metric is None else self.metric.key
        return (f'{self.name}_{self.dataset.key}_{metric}_{self.alpha}_{self.units}_{self.batch}_{self.epochs}_'
                f'{self.fold}_{self.folds}')

    @staticmethod
    def calibration(datasets: Dict[str, Dataset],
                    batches: Iterable[Optional[int]] = (None, 128),
                    units: Iterable[Iterable[int]] = ((256,), (32,) * 2, (256,) * 2, (32,) * 3),
                    wandb: bool = False,
                    formats: Iterable[str] = ('png',),
                    plot: bool = False):
        experiments = LearningExperiment.doe(
            file_name='learning',
            save_time=0,
            verbose=True,
            dataset=datasets,
            batch=list(batches),
            units=list(units),
            fold=0,
            folds=1,
            epochs=300,
            metric=None,
            alpha=None,
            wandb=wandb
        )

    @staticmethod
    def history(datasets: Dict[str, Dataset],
                metrics: Dict[str, HGR],
                alpha: Optional[float] = None,
                full_batch: bool = True,
                folds: int = 3,
                wandb: bool = False,
                formats: Iterable[str] = ('png',),
                plot: bool = False):
        # run experiments
        metrics = {'UNC': None, **metrics}
        epochs, batch = (FULL_EPOCHS, -1) if full_batch else (MINI_EPOCHS, MINI_BATCH)
        experiments = LearningExperiment.doe(
            file_name='learning',
            save_time=0,
            verbose=True,
            dataset=datasets,
            metric=metrics,
            fold=list(range(folds)),
            folds=folds,
            alpha=alpha,
            units=None,
            epochs=epochs,
            batch=batch,
            wandb=wandb
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
            col = len(kpis)
            fig, axes = plt.subplots(2, col, figsize=(5 * col, 8), sharex='all', sharey='col', tight_layout=True)
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
                    axes[i, j].set_ylim((0, None if kpi in ['MSE', 'BCE'] else 1))
            # store, print, and plot if necessary
            for extension in formats:
                name = f'learning_{dataset}_{full_batch}.{extension}'
                with importlib.resources.path('experiments.exports', name) as file:
                    fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f"Learning History for {dataset.title()} ({'Full' if full_batch else 'Mini'} Batch)")
                fig.show()
            plt.close(fig)
