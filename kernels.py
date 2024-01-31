import argparse
import logging
import re

from experiments import CorrelationExperiment
from src.datasets import Polynomial, NonLinear
from src.hgr import AdversarialHGR, DoubleKernelHGR, SingleKernelHGR

# noinspection DuplicatedCode
log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)

# list all the valid datasets
datasets = dict(
    linear=lambda n: Polynomial(degree_x=1, degree_y=1, noise=n),
    x_square=lambda n: Polynomial(degree_x=2, degree_y=1, noise=n),
    x_cubic=lambda n: Polynomial(degree_x=3, degree_y=1, noise=n),
    y_square=lambda n: Polynomial(degree_x=1, degree_y=2, noise=n),
    y_cubic=lambda n: Polynomial(degree_x=1, degree_y=3, noise=n),
    circle=lambda n: Polynomial(degree_x=2, degree_y=2, noise=n),
    sign=lambda n: NonLinear(fn='sign', noise=n),
    relu=lambda n: NonLinear(fn='relu', noise=n),
    sin=lambda n: NonLinear(fn='sin', noise=n),
    tanh=lambda n: NonLinear(fn='tanh', noise=n)
)


# function to retrieve the valid metrics
def metrics(key: str):
    if key == 'adv':
        return 'HGR-NN', AdversarialHGR()
    elif key == 'dkn':
        return 'HGR-KB', DoubleKernelHGR()
    elif key == 'skn':
        return 'HGR-SK', SingleKernelHGR()
    elif re.compile('dkn-([0-9]+)').match(key):
        degree = int(key[4:])
        return f'HGR-KB ({degree})', DoubleKernelHGR(degree_a=degree, degree_b=degree)
    elif re.compile('skn-([0-9]+)').match(key):
        degree = int(key[4:])
        return f'HGR-SK ({degree})', SingleKernelHGR(degree=degree)
    else:
        raise KeyError(f"Invalid key '{key}' for metrics")


# build argument parser
parser = argparse.ArgumentParser(description='Test the Kernel-based HGR on a given dataset')
parser.add_argument(
    '-d',
    '--datasets',
    type=str,
    nargs='+',
    choices=list(datasets),
    default=['circle'],
    help='the dataset on which to run the experiment'
)
parser.add_argument(
    '-m',
    '--metrics',
    type=str,
    nargs='*',
    default=['dkn', 'adv'],
    help='the metrics used to compute the correlations'
)
parser.add_argument(
    '-n',
    '--noises',
    type=float,
    nargs='+',
    default=[1.0],
    help='the noise values used in the experiments'
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
print("Starting experiment 'kernels'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
noises = args.pop('noises')
args['datasets'] = [datasets[ds](n) for ds in args['datasets'] for n in noises]
args['metrics'] = {k: v for k, v in [metrics(key=mt) for mt in args['metrics']]}
CorrelationExperiment.kernels(**args)
