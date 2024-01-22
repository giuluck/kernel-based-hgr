import argparse
import importlib.resources

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from experiments import utils, CorrelationExperiment

# build argument parser
parser = argparse.ArgumentParser(description='Test the Kernel-based HGR on a given dataset')
parser.add_argument(
    '-d',
    '--dataset',
    type=str,
    choices=list(utils.DATASETS),
    default='communities',
    help='the dataset on which to run the experiment'
)
parser.add_argument(
    '-da',
    '--degrees_a',
    type=int,
    nargs='+',
    required=True,
    help='the degrees for the a variable'
)
parser.add_argument(
    '-db',
    '--degrees_b',
    type=int,
    nargs='+',
    required=True,
    help='the degrees for the b variable'
)
parser.add_argument(
    '-vmin',
    '--vmin',
    type=float,
    nargs='?',
    help='the min value used in the color bar (or None if empty)'
)
parser.add_argument(
    '-vmax',
    '--vmax',
    type=float,
    nargs='?',
    help='the max value used in the color bar (or None if empty)'
)
parser.add_argument(
    '-f',
    '--format',
    type=str,
    nargs='*',
    help='the extensions of the files to save'
)
parser.add_argument(
    '--print',
    action='store_true',
    help='whether to print the results'
)
parser.add_argument(
    '--plot',
    action='store_true',
    help='whether to plot the results'
)

# parse arguments and build experiments
args = parser.parse_args()
experiments = CorrelationExperiment.cartesian_product(
    datasets=[args.dataset],
    degrees_a=args.degrees_a,
    degrees_b=args.degrees_b
)

# run experiments and store results
results = [experiment.correlation for experiment in tqdm(experiments)]
results = np.reshape(results, (len(args.degrees_a), len(args.degrees_b)))

# set graphics context
sns.set_context('notebook')
sns.set_style('whitegrid')
# plot results
fig = plt.figure(figsize=(16, 9), tight_layout=True)
ax = fig.gca()
col = ax.imshow(results.transpose()[::-1], cmap=plt.colormaps['gray'], vmin=args.vmin, vmax=args.vmax)
fig.colorbar(col, ax=ax)
ax.set_xlabel('Degree A')
ax.set_xticks(np.arange(len(args.degrees_a) + 1) - 0.5)
ax.set_xticklabels([''] * (len(args.degrees_a) + 1))
ax.set_xticks(np.arange(len(args.degrees_a)), minor=True)
ax.set_xticklabels(args.degrees_a, minor=True)
ax.set_ylabel('Degree B')
ax.set_yticks(np.arange(len(args.degrees_b) + 1) - 0.5)
ax.set_yticklabels([''] * (len(args.degrees_b) + 1))
ax.set_yticks(np.arange(len(args.degrees_b)), minor=True)
ax.set_yticklabels(args.degrees_b[::-1], minor=True)
ax.grid(True, which='major')

for extension in args.format:
    filename = f'monotonicity_{args.dataset}.{extension}'
    with importlib.resources.path('experiments.exports', filename) as file:
        fig.savefig(file)
if args.print:
    print(results)
if args.plot:
    fig.show()
