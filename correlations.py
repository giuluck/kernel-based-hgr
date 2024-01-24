import argparse
import logging

import numpy as np

from experiments import CorrelationExperiment
from src.datasets import Polynomial, NonLinear
from src.hgr import KernelBasedHGR, DensityHGR, ChiSquare, RandomizedDependencyCoefficient, SingleKernelHGR, AdversarialHGR

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)

# list all the valid datasets
datasets = dict(
    linear=lambda n: Polynomial(degree_x=1, degree_y=1, noise=n),
    x_square=lambda n: Polynomial(degree_x=2, degree_y=1, noise=n),
    x_cubic=lambda n: Polynomial(degree_x=3, degree_y=1, noise=n),
    y_square=lambda n: Polynomial(degree_x=1, degree_y=2, noise=n),
    circle=lambda n: Polynomial(degree_x=2, degree_y=2, noise=n),
    sign=lambda n: NonLinear(fn='sign', noise=n),
    relu=lambda n: NonLinear(fn='relu', noise=n),
    sin=lambda n: NonLinear(fn='sin', noise=n)
)

# list all the valid metrics
metrics = {
    'dkn': ('HGR-KB', KernelBasedHGR(degree_a=5, degree_b=5)),
    'skn': ('HGR-SK', SingleKernelHGR(degree=5)),
    'adv': ('HGR-NN', AdversarialHGR()),
    'kde': ('HGR-KDE', DensityHGR()),
    'chi': ('CHI^2', ChiSquare()),
    'rdc': ('RDC', RandomizedDependencyCoefficient()),
    'prs': ('PEARS', KernelBasedHGR(degree_a=1, degree_b=1)),
}

# build argument parser
parser = argparse.ArgumentParser(description='Test multiple HGR metrics on multiple datasets')
parser.add_argument(
    '-d',
    '--datasets',
    type=str,
    nargs='+',
    choices=list(datasets),
    default=list(datasets),
    help='the datasets on which to run the experiment'
)
parser.add_argument(
    '-m',
    '--metrics',
    type=str,
    nargs='+',
    choices=list(metrics),
    default=list(metrics),
    help='the metrics used to compute the correlations'
)
parser.add_argument(
    '-n',
    '--noises',
    type=float,
    nargs='+',
    default=list(np.linspace(0.0, 3.0, num=31, endpoint=True).round(2)),
    help='the noise values used in the experiments'
)
parser.add_argument(
    '-s',
    '--seeds',
    type=int,
    nargs='+',
    default=list(range(30)),
    help='the number of tests per experiment'
)
parser.add_argument(
    '-c',
    '--columns',
    type=int,
    default=3,
    help='the number of columns in the final plot'
)
parser.add_argument(
    '-l',
    '--legend',
    type=int,
    default=1,
    help='where to position the legend in the final plot'
)
parser.add_argument(
    '-f',
    '--formats',
    type=str,
    nargs='*',
    default=['png'],
    help='the extensions of the files to save'
)
parser.add_argument(
    '--verbose',
    action='store_true',
    help='whether to print the results'
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
data = {k: datasets[k] for k in args.pop('datasets')}
metr = {k: v for k, v in [metrics[mt] for mt in args.pop('metrics')]}
CorrelationExperiment.correlations(datasets=data, metrics=metr, **args)
