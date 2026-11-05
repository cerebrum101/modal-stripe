"""Follow failure branches, labeled separately from the initial comparison."""
import json
import subprocess
import sys
from run import ROOT

jobs = [('recall', 'hybrid', 0, 32, 1000), ('recall', 'spectral', 0, 32, 1000)]
jobs += [(task, 'striped', seed, 8, steps)
         for task, steps in [('recall', 1000), ('text', 600)] for seed in range(3)]
for task, kind, seed, modes, steps in jobs:
    path = ROOT/f'results/{task}_{kind}_s{seed}_r{modes}.json'
    if path.exists() and json.loads(path.read_text())['steps'] == steps:
        continue
    print('START', task, kind, seed, modes, flush=True)
    command = [sys.executable, str(ROOT/'run.py'), 'train', '--task', task,
               '--kind', kind, '--seed', str(seed), '--steps', str(steps), '--modes', str(modes)]
    with path.with_suffix('.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('DONE', task, kind, seed, modes, json.loads(path.read_text())['final'], flush=True)
