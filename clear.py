import argparse
import importlib.resources
import json
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

# parse arguments and build pattern
args = parser.parse_args()
pattern = None if args.pattern is None else re.compile(args.pattern)
datasets = None if args.dataset is None else set(args.dataset)
metrics = None if args.metric is None else set(args.metric)
seeds = None if args.seed is None else set(args.seed)
print('Starting clearing procedure...')
print(f'  > datasets --> {datasets}')
print(f'  > metrics --> {metrics}')
print(f'  > seeds --> {seeds}')
print(f'  > pattern --> {pattern}')
print()

# iterate over all the experiments
for experiment in args.experiment:
    # try to access the experiment file
    with importlib.resources.path('experiments.results', f'{experiment}.json') as path:
        pass
    # if it does not exist, there is nothing to clear
    if not path.exists():
        continue
    # otherwise, retrieve the dictionary of results
    with open(path, 'r') as file:
        results = json.load(fp=file)
    # build a dictionary of results to keep
    # a result must if there is at least a non-null matcher that does not match
    output = {}
    for identifier, result in results.items():
        if datasets is not None and result['dataset']['name'] not in datasets:
            # print(f"LEAVE: '{identifier}' from '{experiment}.json' (unmatch dataset '{result['dataset']['name']}')")
            output[identifier] = result
        elif metrics is not None and result['metric']['name'] not in metrics:
            # print(f"LEAVE: '{identifier}' from '{experiment}.json' (unmatch metric '{result['metric']['name']}')")
            output[identifier] = result
        elif seeds is not None and result['seed'] not in seeds:
            # print(f"LEAVE: '{identifier}' from '{experiment}.json' (unmatch seed {result['seed']})")
            output[identifier] = result
        elif pattern is not None and not pattern.match(identifier):
            # print(f"LEAVE: '{identifier}' from '{experiment}.json' (unmatch pattern)")
            output[identifier] = result
        else:
            print(f"CLEAR: '{identifier}' from '{experiment}.json'")
    # dump the file before writing to check if it is json-compliant
    dump = json.dumps(output, indent=2)
    with open(path, 'w') as file:
        file.write(dump)
