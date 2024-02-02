import argparse
import logging

import numpy as np

from experiments import CorrelationExperiment
from src.datasets import Polynomial, NonLinear
from src.hgr import DoubleKernelHGR, DensityHGR, ChiSquare, RandomizedDependencyCoefficient, SingleKernelHGR, \
    AdversarialHGR

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)

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

# list all the valid metrics
metrics = {
    'dkn': ('HGR-KB', DoubleKernelHGR()),
    'skn': ('HGR-SK', SingleKernelHGR()),
    'adv': ('HGR-NN', AdversarialHGR()),
    'kde': ('HGR-KDE', DensityHGR()),
    'chi': ('CHI^2', ChiSquare()),
    'rdc': ('RDC', RandomizedDependencyCoefficient()),
    'prs': ('PEARS', DoubleKernelHGR(degree_a=1, degree_b=1)),
}

# build argument parser
parser = argparse.ArgumentParser(description='Test multiple HGR metrics on multiple datasets')
parser.add_argument(
    '-d',
    '--datasets',
    type=str,
    nargs='+',
    choices=list(datasets),
    default=['x_square', 'circle', 'y_square', 'sign', 'sin'],
    help='the datasets on which to run the experiment'
)
parser.add_argument(
    '-m',
    '--metrics',
    type=str,
    nargs='*',
    choices=list(metrics),
    default=['dkn', 'skn', 'adv', 'kde'],
    help='the metrics used to compute the correlations'
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
    '-ds',
    '--data-seeds',
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
    '-c',
    '--columns',
    type=int,
    default=2,
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
args['metrics'] = {k: v for k, v in [metrics[mt] for mt in args['metrics']]}
CorrelationExperiment.correlations(**args)
