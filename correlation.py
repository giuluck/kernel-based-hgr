from typing import Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
from matplotlib import pyplot as plt

from src.datasets import Polynomial, NonLinear, Dataset
from src.hgr import HGR, KernelBasedHGR, AdversarialHGR, ChiSquare, DensityHGR

datasets = dict(
    linear=lambda n: Polynomial(degree_x=1, degree_y=1, noise=n),
    x_square=lambda n: Polynomial(degree_x=2, degree_y=1, noise=n),
    x_cubic=lambda n: Polynomial(degree_x=3, degree_y=1, noise=n),
    y_square=lambda n: Polynomial(degree_x=1, degree_y=2, noise=n),
    circle=lambda n: Polynomial(degree_x=2, degree_y=2, noise=n),
    sign=lambda n: NonLinear(fn='sign', noise=n),
    relu=lambda n: NonLinear(fn='relu', noise=n),
    sin=lambda n: NonLinear(fn='sin', noise=n),
    tanh=lambda n: NonLinear(fn='tanh', noise=n)
)

metrics = dict(
    pears=KernelBasedHGR(degree_a=1, degree_b=1),
    kb=KernelBasedHGR(degree_a=5, degree_b=5),
    nn=AdversarialHGR(),
    kde=DensityHGR(),
    chi=ChiSquare(),
    # TODO: RDC
)

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
            output[noise][seed] = metric.correlation(a=a, b=b)['correlation']
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
