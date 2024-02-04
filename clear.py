import argparse
from typing import List

from experiments.experiment import Experiment

FILES: List[str] = ['learning', 'correlation']

# build argument parser
parser = argparse.ArgumentParser(description='Clears the results in the experiment files')
parser.add_argument(
    '-f',
    '--file',
    type=str,
    nargs='+',
    default=FILES,
    help='the name of the experiment (or list of such) to clear'
)
parser.add_argument(
    '-d',
    '--dataset',
    type=str,
    nargs='*',
    help='the name of the dataset (or list of such) to clear'
)
parser.add_argument(
    '-m',
    '--metric',
    type=str,
    nargs='*',
    help='the name of the metric (or list of such) to clear'
)
parser.add_argument(
    '-s',
    '--seed',
    type=int,
    nargs='*',
    help='the value of the seed (or list of such) to clear'
)
parser.add_argument(
    '-p',
    '--pattern',
    type=str,
    nargs='?',
    help='provides a pattern to be matched with the experiment identifier'
)
parser.add_argument(
    '--force',
    action='store_true',
    help='clears everything without asking for confirmation at the end'
)
parser.add_argument(
    '--exports',
    action='store_true',
    help='clears only the export files while keeping the experiment results (all the other parameters are ignored)'
)

# parse arguments and decide what to clear
args = parser.parse_args().__dict__
print('Starting exports clearing procedure...')
Experiment.clear_exports()
if args.pop('exports') is False:
    print('Starting experiments clearing procedure...')
    for k, v in args.items():
        print('  >', k, '-->', v)
    print()
    Experiment.clear_results(**args)
