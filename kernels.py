import argparse
import logging
import re

from experiments import CorrelationExperiment
from src.datasets import Polynomial, NonLinear
from src.hgr import AdversarialHGR, DoubleKernelHGR, SingleKernelHGR

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)


# function to retrieve the valid dataset
def datasets(key, noise):
    if key == 'linear':
        return lambda s: Polynomial(degree_x=1, degree_y=1, noise=noise, seed=s)
    elif key == 'x_square':
        return lambda s: Polynomial(degree_x=2, degree_y=1, noise=noise, seed=s)
    elif key == 'x_cubic':
        return lambda s: Polynomial(degree_x=3, degree_y=1, noise=noise, seed=s)
    elif key == 'y_square':
        return lambda s: Polynomial(degree_x=1, degree_y=2, noise=noise, seed=s)
    elif key == 'y_cubic':
        return lambda s: Polynomial(degree_x=1, degree_y=3, noise=noise, seed=s)
    elif key == 'circle':
        return lambda s: Polynomial(degree_x=2, degree_y=2, noise=noise, seed=s)
    elif key == 'sign':
        return lambda s: NonLinear(fn='sign', noise=noise, seed=s)
    elif key == 'relu':
        return lambda s: NonLinear(fn='relu', noise=noise, seed=s)
    elif key == 'sin':
        return lambda s: NonLinear(fn='sin', noise=noise, seed=s)
    elif key == 'tanh':
        return lambda s: NonLinear(fn='tanh', noise=noise, seed=s)
    else:
        raise KeyError(f"Invalid key '{key}' for dataset")


# function to retrieve the valid metric
def metrics(key):
    if key == 'nn':
        return 'HGR-NN', AdversarialHGR()
    elif key == 'kb':
        return 'HGR-KB', DoubleKernelHGR(degree_a=7, degree_b=7)
    elif key == 'sk':
        return 'HGR-SK', SingleKernelHGR(degree=7)
    elif re.compile('kb-([0-9]+)').match(key):
        degree = int(key[4:])
        return f'HGR-KB ({degree})', DoubleKernelHGR(degree_a=degree, degree_b=degree)
    elif re.compile('sk-([0-9]+)').match(key):
        degree = int(key[4:])
        return f'HGR-SK ({degree})', SingleKernelHGR(degree=degree)
    else:
        raise KeyError(f"Invalid key '{key}' for metric")


# build argument parser
parser = argparse.ArgumentParser(description='Inspect the HGR kernels on a given dataset')
parser.add_argument(
    '-d',
    '--datasets',
    type=str,
    nargs='+',
    default=['circle'],
    help='the dataset on which to run the experiment'
)
parser.add_argument(
    '-m',
    '--metrics',
    type=str,
    nargs='*',
    default=['kb-2', 'kb', 'nn'],
    help='the metric used to compute the correlations'
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
args['datasets'] = [datasets(key=ds, noise=n) for ds in args['datasets'] for n in noises]
args['metrics'] = {k: v for k, v in [metrics(key=mt) for mt in args['metrics']]}
CorrelationExperiment.kernels(**args)
