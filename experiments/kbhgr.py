import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.datasets import Communities, Adult, Polynomial, NonLinear
from src.hgr import KernelBasedHGR

DATASETS = dict(
    communities=Communities(continuous=True),
    adult=Adult(continuous=True),
    linear=Polynomial(degree_x=1, degree_y=1),
    x_squared=Polynomial(degree_x=2, degree_y=1),
    x_cubic=Polynomial(degree_x=3, degree_y=1),
    y_squared=Polynomial(degree_x=1, degree_y=2),
    y_cubic=Polynomial(degree_x=1, degree_y=3),
    circle=Polynomial(degree_x=2, degree_y=2),
    sign=NonLinear(name='sign'),
    relu=NonLinear(name='relu'),
    sin=NonLinear(name='sin'),
    tanh=NonLinear(name='tanh')
)

# build argument parser
parser = argparse.ArgumentParser(description='Test the Kernel-based HGR on a given dataset')
parser.add_argument(
    '-d',
    '--dataset',
    type=str,
    choices=list(DATASETS),
    default='communities',
    help='the dataset on which to run the experiment'
)
parser.add_argument(
    '-da',
    '--degrees_a',
    type=int,
    nargs='+',
    help='the degrees for the a variable',
    required=True
)
parser.add_argument(
    '-db',
    '--degrees_b',
    type=int,
    nargs='+',
    help='the degrees for the b variable',
    required=True
)

# parse arguments
args = parser.parse_args()
dataset = DATASETS[args.dataset]
degrees_a = args.degrees_a
degrees_b = args.degrees_b

# retrieve data
a = dataset.excluded(backend='numpy')
b = dataset.target(backend='numpy')

# build results list and iterate through degrees
results = []
pbar = tqdm(total=len(degrees_a) * len(degrees_b))
for i, da in enumerate(degrees_a):
    for j, db in enumerate(degrees_b):
        metric = KernelBasedHGR(degree_a=da, degree_b=db)
        # compute kernel matrices and retrieve the hgr value
        f = np.stack([a ** d - np.mean(a ** d) for d in np.arange(da) + 1], axis=1)
        g = np.stack([b ** d - np.mean(b ** d) for d in np.arange(db) + 1], axis=1)
        hgr, alp, bet = metric.kbhgr(a=a, b=b)
        results.append({
            'da': da,
            'db': db,
            'hgr': hgr,
            'alpha': alp,
            'beta': bet
        })
        # update progress bar
        pbar.update(n=1)
pbar.close()

# store results
df = pd.DataFrame(data=results)
print(df)
