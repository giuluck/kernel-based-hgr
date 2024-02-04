import argparse
import logging

from experiments.learning import LearningExperiment
from src.datasets import Communities, Adult
from src.hgr import DoubleKernelHGR, SingleKernelHGR, AdversarialHGR

for name in ["lightning_fabric", "pytorch_lightning.utilities.rank_zero", "pytorch_lightning.accelerators.cuda"]:
    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(logging.ERROR)

# list all the valid datasets
datasets = dict(
    communities=Communities(continuous=True),
    adult=Adult(continuous=True)
)

# list all the valid penalties
penalties = dict(
    dkn=('HGR-KB', DoubleKernelHGR()),
    skn=('HGR-SK', SingleKernelHGR()),
    adv=('HGR-NN', AdversarialHGR())
)


# build argument parser
parser = argparse.ArgumentParser(description='Train multiple neural networks using different HGR penalties')
parser.add_argument(
    '-d',
    '--datasets',
    type=str,
    nargs='+',
    choices=list(datasets),
    default=['communities'],
    help='the datasets on which to run the experiment'
)
parser.add_argument(
    '-p',
    '--penalties',
    type=str,
    nargs='*',
    choices=list(penalties),
    default=['dkn', 'skn'],
    help='the penalties used to compute the correlations'
)
parser.add_argument(
    '-a',
    '--alpha',
    type=float,
    nargs='?',
    help='the alpha value used in the experiments'
)
parser.add_argument(
    '--full-batch',
    action='store_true',
    help='whether to train the networks full batch or with mini batches'
)
parser.add_argument(
    '--warm-start',
    action='store_true',
    help='whether to train the networks starting from a pretrained non-constrained network or from scratch'
)
parser.add_argument(
    '-e',
    '--entity',
    type=str,
    nargs='?',
    help='the Weights & Biases entity, or None for no Weights & Biases logging'
)

# parse arguments, build experiments, then export the results
args = parser.parse_args().__dict__
print("Starting experiment 'learning'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['datasets'] = {k: datasets[k] for k in args['datasets']}
args['penalties'] = {k: v for k, v in [penalties[p] for p in args['penalties']]}
LearningExperiment.learning(**args)
