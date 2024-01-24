import argparse
import importlib.resources
import os
import pickle
import re
from typing import List

EXPERIMENTS: List[str] = ['monotonicity', 'correlation']

# build argument parser
parser = argparse.ArgumentParser(description='Clears the results in the experiment files')
parser.add_argument(
    '-e',
    '--experiment',
    type=str,
    nargs='+',
    default=EXPERIMENTS,
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
    '--exports',
    action='store_true',
    help='clears the export files rather than the experiments (all the other parameters are ignored)'
)

# parse arguments and decide what to clear
args = parser.parse_args()
if args.exports:
    print('Starting exports clearing procedure...\n')
    with importlib.resources.files('experiments.results') as folder:
        for file in os.listdir(folder):
            if file != '__pycache__' and not file.endswith('.py') and not file.endswith('.pkl'):
                print(f'CLEAR: export file {file}')
                os.remove(os.path.join(folder, file))
else:
    # build sets and pattern
    pattern = None if args.pattern is None else re.compile(args.pattern)
    datasets = None if args.dataset is None else set(args.dataset)
    metrics = None if args.metric is None else set(args.metric)
    seeds = None if args.seed is None else set(args.seed)
    print('Starting experiments clearing procedure...')
    print(f'  > datasets --> {datasets}')
    print(f'  > metrics --> {metrics}')
    print(f'  > seeds --> {seeds}')
    print(f'  > pattern --> {pattern}')
    print()
    # iterate over all the experiments
    for experiment in args.experiment:
        # try to access the experiment file
        with importlib.resources.path('experiments.results', f'{experiment}.pkl') as path:
            pass
        # if it does not exist, there is nothing to clear
        if not path.exists():
            continue
        # otherwise, retrieve the dictionary of results
        with open(path, 'rb') as file:
            results = pickle.load(file=file)
        # build a dictionary of results to keep
        # a result must if there is at least a non-null matcher that does not match
        output = {}
        for idx, res in results.items():
            if datasets is not None and res['dataset']['name'] not in datasets:
                # print(f"LEAVE: '{identifier}' from '{experiment}.json' (unmatch dataset '{res['dataset']['name']}')")
                output[idx] = res
            elif metrics is not None and res['metric']['name'] not in metrics:
                # print(f"LEAVE: '{identifier}' from '{experiment}.json' (unmatch metric '{res['metric']['name']}')")
                output[idx] = res
            elif seeds is not None and res['seed'] not in seeds:
                # print(f"LEAVE: '{identifier}' from '{experiment}.json' (unmatch seed {res['seed']})")
                output[idx] = res
            elif pattern is not None and not pattern.match(idx):
                # print(f"LEAVE: '{identifier}' from '{experiment}.json' (unmatch pattern)")
                output[idx] = res
            else:
                print(f"CLEAR: '{idx}' from '{experiment}.pkl'")
        # dump the file before writing to check if it is json-compliant
        dump = pickle.dumps(output)
        with open(path, 'wb') as file:
            file.write(dump)
