import argparse
import logging

from experiments import CorrelationExperiment
from src.datasets import Polynomial, NonLinear, Communities, Adult

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)

# list all the valid datasets
datasets = dict(
    adult=Adult(continuous=True),
    communities=Communities(continuous=True),
    linear=Polynomial(degree_x=1, degree_y=1, noise=0.0),
    x_square=Polynomial(degree_x=2, degree_y=1, noise=0.0),
    y_square=Polynomial(degree_x=1, degree_y=2, noise=0.0),
    circle=Polynomial(degree_x=2, degree_y=2, noise=0.0),
    sign=NonLinear(fn='sign', noise=0.0),
    relu=NonLinear(fn='relu', noise=0.0),
    sin=NonLinear(fn='sin', noise=0.0),
    tanh=NonLinear(fn='tanh', noise=0.0)
)

# build argument parser
parser = argparse.ArgumentParser(description='Test the Kernel-based HGR on a given dataset')
parser.add_argument(
    '-d',
    '--dataset',
    type=str,
    choices=list(datasets),
    default='communities',
    help='the dataset on which to run the experiment'
)
parser.add_argument(
    '-da',
    '--degrees_a',
    type=int,
    nargs='+',
    default=[1, 2, 3, 4, 5, 6, 7],
    help='the degrees for the a variable'
)
parser.add_argument(
    '-db',
    '--degrees_b',
    type=int,
    nargs='+',
    default=[1, 2, 3, 4, 5, 6, 7],
    help='the degrees for the b variable'
)
parser.add_argument(
    '-m',
    '--vmin',
    type=float,
    nargs='?',
    help='the min value used in the color bar (or None if empty)'
)
parser.add_argument(
    '-M',
    '--vmax',
    type=float,
    nargs='?',
    help='the max value used in the color bar (or None if empty)'
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
print('Starting Experiment: MONOTONICITY')
for k, v in args.items():
    print('  >', k, '-->', v)
args['dataset'] = datasets[args['dataset']]
CorrelationExperiment.monotonicity(**args)
