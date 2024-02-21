import argparse
import logging
import re

import numpy as np

from experiments import CorrelationExperiment
from src.datasets import Polynomial, NonLinear
from src.hgr import DoubleKernelHGR, DensityHGR, ChiSquare, RandomizedDependenceCoefficient, SingleKernelHGR, \
    AdversarialHGR

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)


# function to retrieve the valid metric
def metrics(key):
    if key == 'nn':
        return 'HGR-NN', AdversarialHGR()
    elif key == 'kde':
        return 'HGR-KDE', DensityHGR()
    elif key == 'chi':
        return 'CHI^2', ChiSquare()
    elif key == 'rdc':
        return 'RDC', RandomizedDependenceCoefficient()
    elif key == 'prs':
        return 'PEARS', DoubleKernelHGR(degree_a=1, degree_b=1)
    elif key == 'kb':
        return 'HGR-KB', DoubleKernelHGR()
    elif key == 'sk':
        return 'HGR-SK', SingleKernelHGR()
    elif re.compile('kb-([0-9]+)').match(key):
        degree = int(key[3:])
        return f'HGR-KB ({degree})', DoubleKernelHGR(degree_a=degree, degree_b=degree)
    elif re.compile('sk-([0-9]+)').match(key):
        degree = int(key[3:])
        return f'HGR-SK ({degree})', SingleKernelHGR(degree=degree)
    else:
        raise KeyError(f"Invalid key '{key}' for metric")


# list all the valid datasets
datasets = dict(
    linear=lambda n, s: Polynomial(degree_x=1, degree_y=1, noise=n, seed=s),
    x_square=lambda n, s: Polynomial(degree_x=2, degree_y=1, noise=n, seed=s),
    x_cubic=lambda n, s: Polynomial(degree_x=3, degree_y=1, noise=n, seed=s),
    y_square=lambda n, s: Polynomial(degree_x=1, degree_y=2, noise=n, seed=s),
    y_cubic=lambda n, s: Polynomial(degree_x=1, degree_y=3, noise=n, seed=s),
    circle=lambda n, s: Polynomial(degree_x=2, degree_y=2, noise=n, seed=s),
    sign=lambda n, s: NonLinear(fn='sign', noise=n, seed=s),
    relu=lambda n, s: NonLinear(fn='relu', noise=n, seed=s),
    sin=lambda n, s: NonLinear(fn='sin', noise=n, seed=s),
    tanh=lambda n, s: NonLinear(fn='tanh', noise=n, seed=s)
)

# build argument parser
parser = argparse.ArgumentParser(description='Test multiple HGR metrics on multiple datasets')
parser.add_argument(
    '-f',
    '--folder',
    type=str,
    default='.',
    help='the path where to search and store the results and the exports'
)
parser.add_argument(
    '-d',
    '--datasets',
    type=str,
    nargs='+',
    choices=list(datasets),
    default=['circle', 'x_square', 'y_square', 'sign', 'sin'],
    help='the datasets on which to run the experiment'
)
parser.add_argument(
    '-m',
    '--metrics',
    type=str,
    nargs='*',
    default=['kb', 'sk', 'nn', 'kde', 'rdc'],
    help='the metric used to compute the correlations'
)
parser.add_argument(
    '-n',
    '--noises',
    type=float,
    nargs='+',
    default=list(np.linspace(0.0, 3.0, num=16, endpoint=True).round(2)),
    help='the noise values used in the experiments'
)
parser.add_argument(
    '-ns',
    '--noise-seeds',
    type=int,
    nargs='+',
    default=list(range(10)),
    help='the number of dataset variants per experiment'
)
parser.add_argument(
    '-as',
    '--algorithm-seeds',
    type=int,
    nargs='+',
    default=list(range(10)),
    help='the number of tests per experiment'
)
parser.add_argument(
    '--test',
    action='store_true',
    help='whether to compute the correlations on test data'
)
parser.add_argument(
    '-c',
    '--columns',
    type=int,
    default=2,
    help='the number of columns in the final plot'
)
parser.add_argument(
    '-e',
    '--extensions',
    type=str,
    nargs='*',
    default=['png'],
    help='the extensions of the files to save'
)
parser.add_argument(
    '--plot',
    action='store_true',
    help='whether to plot the results'
)
parser.add_argument(
    '-t',
    '--save-time',
    type=int,
    default=60,
    help='the number of seconds after which to store the computed results'
)

# parse arguments, build experiments, then export the results
args = parser.parse_args().__dict__
print("Starting experiment 'correlations'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['datasets'] = {k: datasets[k] for k in args['datasets']}
args['metrics'] = {k: v for k, v in [metrics(mt) for mt in args['metrics']]}
CorrelationExperiment.correlations(**args)
