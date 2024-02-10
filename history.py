import argparse
import logging
import os
import warnings

from experiments import LearningExperiment
from src.datasets import Communities, Adult, Census
from src.hgr import DoubleKernelHGR, SingleKernelHGR, AdversarialHGR, DensityHGR

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
    census=Census()
)

# list all the valid metrics
metrics = dict(
    unc=('UNC', None),
    sk=('HGR-SK', SingleKernelHGR(degree=5)),
    kb=('HGR-KB', DoubleKernelHGR(degree_a=5, degree_b=5)),
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
    default=list(datasets),
    help='the datasets on which to run the experiment'
)
parser.add_argument(
    '-m',
    '--metrics',
    type=str,
    nargs='+',
    choices=list(metrics),
    default=['unc', 'sk', 'nn', 'kb'],
    help='the metrics used as penalties'
)
parser.add_argument(
    '-s',
    '--split',
    type=float,
    default=0.3,
    help='the train/test split value'
)
parser.add_argument(
    '-b',
    '--batches',
    type=str,
    choices=['mini', 'full', 'both'],
    default='both',
    help='whether to train the networks with mini batches, full batch, or both'
)
parser.add_argument(
    '-wp',
    '--wandb-project',
    type=str,
    nargs='?',
    help='the name of the Weights & Biases project for logging, or None for no logging.'
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

# parse arguments, build experiments, then export the results
args = parser.parse_args().__dict__
print("Starting experiment 'history'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['datasets'] = {k: datasets[k] for k in args['datasets']}
args['metrics'] = {k: v for k, v in [metrics[m] for m in args['metrics']]}
LearningExperiment.history(**args)
