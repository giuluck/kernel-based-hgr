from typing import Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
from matplotlib import pyplot as plt

from src.datasets import Polynomial, NonLinear, Dataset
from src.hgr import HGR, KernelBasedHGR

datasets = [
    lambda n: Polynomial(degree_x=1, degree_y=1, noise=n),
    lambda n: Polynomial(degree_x=2, degree_y=1, noise=n),
    lambda n: Polynomial(degree_x=3, degree_y=1, noise=n),
    lambda n: Polynomial(degree_x=1, degree_y=2, noise=n),
    lambda n: Polynomial(degree_x=2, degree_y=2, noise=n),
    lambda n: NonLinear(name='sign', noise=n),
    lambda n: NonLinear(name='relu', noise=n),
    lambda n: NonLinear(name='sin', noise=n),
    lambda n: NonLinear(name='tanh', noise=n)
]

metrics = [
    KernelBasedHGR(degree_a=1, degree_b=1, name='PEARS'),
    # KernelBasedHGR(degree_a=5, degree_b=5, name='HGR-KB'),
    # AdversarialHGR(name='HGR-NN'),
    # DensityHGR(name='HGR-KDE'),
    # ChiSquare(name='CHI^2'),
    # TODO: RDC
]

noises = np.arange(7) / 20.0
tests = 5


def correlation(metric: HGR, dataset: Callable[[float], Dataset]) -> pd.DataFrame:
    output = pd.DataFrame(index=range(tests), columns=noises, dtype=float)
    for noise in noises:
        # retrieve data
        data = dataset(noise)
        a = data.excluded(backend='numpy')
        b = data.target(backend='numpy')
        for seed in range(tests):
            pl.seed_everything(seed, workers=True)
            output[noise][seed] = metric.correlation(a=a, b=b)
    return output


if __name__ == '__main__':
    # set graphics context
    sns.set_context('notebook')
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), tight_layout=True)
    for ds, ax in zip(datasets, axes.flatten()):
        for mtr in metrics:
            results = correlation(metric=mtr, dataset=ds)
            sns.lineplot(data=results, estimator='mean', errorbar=results.std(), ax=ax)

    # if ds.name == 'sign':
    #     sns.lineplot(x=x[x < 0], y=y[x < 0], color='tab:blue', sort=False, estimator=None, ax=ax)
    #     sns.lineplot(x=x[x > 0], y=y[x > 0], color='tab:blue', sort=False, estimator=None, ax=ax)
    # else:
    #     sns.lineplot(x=x, y=y, color='tab:blue', sort=False, estimator=None, ax=ax)
    fig.show()
