import argparse
import importlib.resources
import os
import re

from experiments import utils

EXPERIMENTS = ['monotonicity', 'correlation', 'learning']

# build argument parser
parser = argparse.ArgumentParser(description='Test the Kernel-based HGR on a given dataset')
parser.add_argument(
    '-p',
    '--pattern',
    type=str,
    nargs='?',
    help='provides a pattern to identify which files to clear (the other parameters are ignored)'
)
parser.add_argument(
    '-e',
    '--experiment',
    type=str,
    nargs='+',
    choices=EXPERIMENTS,
    help='the experiment (or list of such) to clear'
)
parser.add_argument(
    '-d',
    '--dataset',
    type=str,
    nargs='+',
    choices=list(utils.DATASETS),
    help='the dataset (or list of such) to clear'
)
parser.add_argument(
    '-m',
    '--metric',
    type=str,
    nargs='+',
    choices=list(utils.METRICS),
    help='the metric (or list of such) to clear'
)
parser.add_argument(
    '-f',
    '--format',
    type=str,
    nargs='+',
    help='the file extension (or list of such) to clear'
)
parser.add_argument(
    '--no-results',
    action='store_true',
    help='excludes the results folder from the cleaning process'
)
parser.add_argument(
    '--no-exports',
    action='store_true',
    help='excludes the exports folder from the cleaning process'
)

# parse arguments and build pattern
#   - if a custom pattern is passed, use that one
#   - otherwise, build the pattern with matches of experiment, dataset, metric, and format
#   - finally, if none of the arguments is passed (the pattern is still empty), remove all the files
args = parser.parse_args()
if args.pattern is not None:
    pattern = args.pattern
else:
    pattern = ''
    if args.experiment is not None:
        pattern += f".*({'|'.join(args.experiment)}).*"
    if args.dataset is not None:
        pattern += f".*({'|'.join(args.dataset)}).*"
    if args.metric is not None:
        pattern += f"({'|'.join(args.metric)}).*"
    if args.format is not None:
        pattern += f".({'|'.join(args.format)})"
    if pattern == '':
        pattern = '.*'
pattern = re.compile(pattern)

# remove all files that match the pattern (excluding python files) in the two sub-packages
if not args.no_exports:
    with importlib.resources.files('experiments.exports') as folder:
        for file in os.listdir(folder):
            if pattern.match(file) and file != '__pycache__' and not file.endswith('.py'):
                os.remove(f'{folder}/{file}')
if not args.no_results:
    with importlib.resources.files('experiments.results') as folder:
        for file in os.listdir(folder):
            if pattern.match(file) and file != '__pycache__' and not file.endswith('.py'):
                os.remove(f'{folder}/{file}')
