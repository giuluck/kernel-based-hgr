import importlib.resources
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Iterable, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.experiment import Experiment
from src.datasets import Dataset
from src.hgr import HGR
from src.learning import MultiLayerPerceptron, Data, Loss, Accuracy, Metric, InternalLogger, Progress, History, \
    Correlation

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

SEED: int = 0
"""The random seed used in the experiment."""

MINI_BATCH: int = 512
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

    batch: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The batch size used during training, or -1 to train full batch."""

    epochs: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The number of training epochs."""

    wandb_project: Optional[str] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The name of the Weights & Biases project for logging, or None for no logging."""

    def __post_init__(self):
        assert self.metric is not None or self.alpha is None, "If metric=None, alpha must be None as well."

    def _compute(self) -> Experiment.Result:
        pl.seed_everything(SEED, workers=True)
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
        logger = InternalLogger()
        history = History(key=self.key)
        if self.wandb_project is not None:
            wandb_logger = WandbLogger(project=self.wandb_project, name=self.key, log_model='all')
            wandb_logger.experiment.config.update(self.configuration)
            loggers = [logger, wandb_logger]
        else:
            loggers = [logger]
        trainer = pl.Trainer(
            deterministic=True,
            min_epochs=self.epochs,
            max_epochs=self.epochs,
            logger=loggers,
            callbacks=[history, Progress()],
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False
        )
        # run fitting
        start = time.time()
        batch_size = len(trn) if self.batch == -1 else self.batch
        trainer.fit(
            model=model,
            train_dataloaders=DataLoader(trn_data, batch_size=batch_size),
            val_dataloaders=DataLoader(val_data, batch_size=len(val))
        )
        gap = time.time() - start
        # close wandb in case it was used in the logger
        if self.wandb_project is not None:
            wandb.finish()
        # store external files and return result
        external = f'{self.key}.pkl'
        with importlib.resources.path('experiments.results', external) as path:
            assert not path.exists(), f"Experiment '{self.key}' is already present in package 'experiments.results'"
            with open(path, 'wb') as file:
                pickle.dump({
                    'train_inputs': trn_data.x,
                    'train_target': trn_data.y,
                    'val_inputs': val_data.x,
                    'val_target': val_data.y
                }, file=file)
        return Experiment.Result(timestamp=start, execution=gap, history=history, external=external, **logger.results)

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
                    batches: Iterable[int] = (-1, 128, 512),
                    units: Iterable[Iterable[int]] = ((32,), (256,), (32,) * 2, (256,) * 2, (32,) * 3, (256,) * 3),
                    wandb_project: Optional[str] = None,
                    formats: Iterable[str] = ('png',),
                    plot: bool = False):
        def configuration(ds, bt, un, fl):
            classification = datasets[ds].classification
            return dict(dataset=ds, batch=bt, units=un, fold=fl), [
                Loss(classification=classification),
                Accuracy(classification=classification)
            ]

        units = [list(u) for u in units]
        experiments = LearningExperiment.doe(
            file_name='learning',
            save_time=0,
            verbose=True,
            dataset=datasets,
            batch={str(b): b for b in batches},
            units={str(u): u for u in units},
            fold=[0, 1, 2],
            folds=3,
            epochs=300,
            metric=None,
            alpha=None,
            wandb_project=wandb_project
        )
        # get metric results and add time
        results = LearningExperiment._metrics(experiments=experiments, configuration=configuration)
        times = []
        for index, experiment in experiments.items():
            info, _ = configuration(*index)
            times += [{
                **info,
                'kpi': 'Time',
                'split': 'Train',
                'epoch': epoch,
                'value': experiment.result['time'][epoch]
            } for epoch in range(experiment.epochs)]
        results = pd.concat((results, pd.DataFrame(times)))
        # plot results
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        for (dataset, kpi), data in results.groupby(['dataset', 'kpi']):
            cl = len(units)
            rw = len(batches)
            fig, axes = plt.subplots(rw, cl, figsize=(5 * cl, 4 * rw), sharex='all', sharey='all', tight_layout=True)
            # used to index the axes in case either or both units and batches have only one value
            axes = np.array(axes).reshape(rw, cl)
            for i, batch in enumerate(batches):
                for j, unit in enumerate(units):
                    sns.lineplot(
                        data=data[np.logical_and(data['batch'] == str(batch), data['units'] == str(unit))],
                        x='epoch',
                        y='value',
                        hue='split',
                        style='split',
                        estimator='mean',
                        errorbar='sd',
                        linewidth=2,
                        palette=['black'] if kpi == 'Time' else PALETTE[1:3],
                        ax=axes[i, j]
                    )
                    axes[i, j].set_ylabel(kpi)
                    axes[i, j].set_ylim((0, 1 if kpi in ['R2', 'ACC'] else data[data['epoch'] > 20]['value'].max()))
                    axes[i, j].set_title(f"Batch Size: {'Full' if batch == -1 else batch} - Units: {unit}")
            # store, print, and plot if necessary
            for extension in formats:
                name = f'calibration_{kpi}_{dataset}.{extension}'
                with importlib.resources.path('experiments.exports', name) as file:
                    fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f"Calibration {kpi} for {dataset.title()}")
                fig.show()
            plt.close(fig)

    @staticmethod
    def history(datasets: Dict[str, Dataset],
                metrics: Dict[str, HGR],
                alpha: Optional[float] = None,
                full_batch: bool = True,
                folds: int = 5,
                wandb_project: Optional[str] = None,
                formats: Iterable[str] = ('png',),
                plot: bool = False):
        def configuration(ds, mt, fl):
            d = datasets[ds]
            return dict(dataset=ds, metric=mt, fold=fl), [
                Loss(classification=d.classification),
                Accuracy(classification=d.classification),
                Correlation(excluded=d.excluded_index, algorithm='kb'),
                Correlation(excluded=d.excluded_index, algorithm='nn')
            ]

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
            wandb_project=wandb_project
        )
        # get and plot metric results
        results = LearningExperiment._metrics(experiments=experiments, configuration=configuration)
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        for dataset in datasets.keys():
            group = results[results['dataset'] == dataset]
            kpis = group['kpi'].unique()
            col = len(kpis)
            fig, axes = plt.subplots(2, col, figsize=(5 * col, 8), sharex='all', sharey='col', tight_layout=True)
            for i, split in enumerate(['Train', 'Val']):
                for j, kpi in enumerate(kpis):
                    sns.lineplot(
                        data=group[np.logical_and(group['split'] == split, group['kpi'] == kpi)],
                        x='epoch',
                        y='value',
                        estimator='mean',
                        errorbar='sd',
                        linewidth=2,
                        hue='metric',
                        style='metric',
                        palette=PALETTE[:len(metrics)],
                        ax=axes[i, j]
                    )
                    axes[i, j].set_ylabel(f'{split} {kpi}')
                    axes[i, j].set_ylim((0, None if kpi in ['MSE', 'BCE'] else 1))
            # store, print, and plot if necessary
            for extension in formats:
                name = f'history_{dataset}_{full_batch}.{extension}'
                with importlib.resources.path('experiments.exports', name) as file:
                    fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f"Learning History for {dataset.title()} ({'Full' if full_batch else 'Mini'} Batch)")
                fig.show()
            plt.close(fig)
        # get and plot history results
        history = []
        for index, experiment in experiments.items():
            info, _ = configuration(*index)
            for epoch in range(experiment.epochs):
                history.append({**info, 'epoch': epoch, 'kpi': 'alpha', 'value': experiment.result['alpha'][epoch]})
                history.append({**info, 'epoch': epoch, 'kpi': 'time', 'value': experiment.result['time'][epoch]})
        history = pd.DataFrame(history)
        row = len(datasets)
        fig, axes = plt.subplots(row, 2, figsize=(10, 4 * row), sharex='all', sharey=None, tight_layout=True)
        axes = np.array(axes).reshape(row, 2)
        for i, dataset in enumerate(datasets.keys()):
            for j, kpi in enumerate(['alpha', 'time']):
                sns.lineplot(
                    data=history[np.logical_and(history['dataset'] == dataset, history['kpi'] == kpi)],
                    x='epoch',
                    y='value',
                    estimator='mean',
                    errorbar='sd',
                    linewidth=2,
                    hue='metric',
                    style='metric',
                    palette=PALETTE[:len(metrics)],
                    ax=axes[i, j]
                )
                axes[i, j].set_ylabel(f'{dataset} {kpi}')
        # store, print, and plot if necessary
        for extension in formats:
            name = f'history_outputs_{full_batch}.{extension}'
            with importlib.resources.path('experiments.exports', name) as file:
                fig.savefig(file, bbox_inches='tight')
        if plot:
            fig.suptitle(f"Learning History Outputs for ({'Full' if full_batch else 'Mini'} Batch)")
            fig.show()
        plt.close(fig)

    @staticmethod
    def _metrics(experiments: Dict[Any, 'LearningExperiment'],
                 configuration: Callable[[tuple], Tuple[Dict[str, Any], Iterable[Metric]]]) -> pd.DataFrame:
        results = []
        for index, experiment in experiments.items():
            with importlib.resources.open_binary('experiments.results', experiment.result.external) as file:
                ext = pickle.load(file=file)
            # retrieve input data
            xtr = ext['train_inputs'].numpy(force=True)
            ytr = ext['train_target'].numpy(force=True).flatten()
            xvl = ext['val_inputs'].numpy(force=True)
            yvl = ext['val_target'].numpy(force=True).flatten()
            # compute metrics for each epoch (if they are present, load pre-computed values from external file)
            info, metrics = configuration(*index)
            outputs = {
                # **{f'train_{mtr.name}': ext.get(f'train_{mtr.name}', []) for mtr in metrics},
                # **{f'val_{mtr.name}': ext.get(f'val_{mtr.name}', []) for mtr in metrics}
                **{f'train_{mtr.name}': [] for mtr in metrics},
                **{f'val_{mtr.name}': [] for mtr in metrics}
            }
            reserialize = False
            for epoch in tqdm(range(experiment.epochs), desc=f'Computing Metrics for {experiment.key}'):
                info['epoch'] = epoch
                hst = experiment.result['history'][epoch]
                ptr = hst['train_predictions'].numpy(force=True).flatten()
                pvl = hst['val_predictions'].numpy(force=True).flatten()
                for mtr in metrics:
                    for split, (x, y, p) in zip(['train', 'val'], [(xtr, ytr, ptr), (xvl, yvl, pvl)]):
                        mtr_outputs = outputs[f'{split}_{mtr.name}']
                        # in case a value is already present for this metric use it, otherwise compute it
                        if len(mtr_outputs) > epoch:
                            mtr_value = mtr_outputs[epoch]
                        else:
                            reserialize = True
                            mtr_value = mtr(x=x, y=y, p=p)
                            mtr_outputs.append(mtr_value)
                        results.append({**info, 'kpi': mtr.name, 'split': split.title(), 'value': mtr_value})
            # if at least one value was re-computed, re-serialize the external file
            if reserialize:
                ext.update(outputs)
                with importlib.resources.path('experiments.results', experiment.result.external) as filepath:
                    with open(filepath, 'wb') as file:
                        pickle.dump(ext, file=file)
        return pd.DataFrame(results)
