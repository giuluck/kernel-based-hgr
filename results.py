import argparse
import logging
import os
import warnings

from experiments import LearningExperiment
from src.datasets import Communities, Adult, Census, Students
from src.hgr import DoubleKernelHGR, SingleKernelHGR, AdversarialHGR, DensityHGR

# noinspection DuplicatedCode
os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", ".*does not have many workers.*")
for name in ["lightning_fabric", "pytorch_lightning.utilities.rank_zero", "pytorch_lightning.accelerators.cuda"]:
    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(logging.ERROR)

# list all the valid datasets
datasets = dict(
    communities=Communities(),
    adult=Adult(),
    census=Census(),
    students=Students()
)

# list all the valid metrics
metrics = dict(
    sk=('HGR-SK', SingleKernelHGR()),
    kb=('HGR-KB', DoubleKernelHGR()),
    nn=('HGR-NN', AdversarialHGR()),
    kde=('HGR-KDE', DensityHGR())
)

# build argument parser
parser = argparse.ArgumentParser(description='Train multiple neural networks using different HGR metrics as penalizers')
parser.add_argument(
    '-d',
    '--datasets',
    type=str,
    nargs='+',
    choices=list(datasets),
    default=['communities', 'adult', 'census'],
    help='the datasets on which to run the experiment'
)
parser.add_argument(
    '-m',
    '--metrics',
    type=str,
    nargs='*',
    choices=list(metrics),
    default=['sk', 'kb', 'nn'],
    help='the metrics used as penalties'
)
parser.add_argument(
    '-s',
    '--steps',
    type=int,
    default=500,
    help='the number of steps to run for each network'
)
parser.add_argument(
    '-k',
    '--folds',
    type=int,
    default=5,
    help='the number of folds to be used for cross-validation'
)
parser.add_argument(
    '-p',
    '--wandb-project',
    type=str,
    nargs='?',
    help='the name of the Weights & Biases project for logging, or None for no logging'
)
parser.add_argument(
    '-f',
    '--formats',
    type=str,
    nargs='*',
    choices=['csv', 'tex'],
    default=['csv'],
    help='the extensions of the files to save'
)

# parse arguments, build experiments, then export the results
args = parser.parse_args().__dict__
print("Starting experiment 'history'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['datasets'] = {k: datasets[k] for k in args['datasets']}
args['metrics'] = {k: v for k, v in [metrics[m] for m in args['metrics']]}
LearningExperiment.results(**args)
