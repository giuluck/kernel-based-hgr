import argparse
import logging
import os
import warnings

from experiments import LearningExperiment
from src.hgr import DoubleKernelHGR, SingleKernelHGR, AdversarialHGR, DensityHGR

# noinspection DuplicatedCode
os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", ".*does not have many workers.*")
for name in ["lightning_fabric", "pytorch_lightning.utilities.rank_zero", "pytorch_lightning.accelerators.cuda"]:
    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(logging.ERROR)

# build argument parser
parser = argparse.ArgumentParser(description='Run the practical use case tests')
parser.add_argument(
    '-s',
    '--steps',
    type=int,
    default=500,
    help='the number of steps to run for each network'
)
parser.add_argument(
    '-p',
    '--wandb-project',
    type=str,
    nargs='?',
    help='the name of the Weights & Biases project for logging, or None for no logging'
)
parser.add_argument(
    '-u',
    '--units',
    type=int,
    nargs='*',
    help='the hidden units of the neural networks (if not passed, uses the dataset default choice)'
)
parser.add_argument(
    '-b',
    '--batch',
    type=int,
    nargs='?',
    help='the batch size used during training (if not passed, uses the dataset default choice)'
)
parser.add_argument(
    '-t',
    '--threshold',
    type=float,
    nargs='?',
    help='the penalty threshold used during training (if not passed, uses the dataset default choice)'
)
parser.add_argument(
    '-a',
    '--alpha',
    type=float,
    nargs='?',
    help='the alpha value used for the penalty constraint (if not passed, uses automatic tuning)'
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
print("Starting experiment 'usecase'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
LearningExperiment.usecase(**args)
