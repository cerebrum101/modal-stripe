"""Run parameter-matched, equal-token comparisons sequentially to avoid contention."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for task, steps in [('recall', 1000), ('text', 600)]:
    for seed in range(3):
        for kind in ['full', 'local', 'spectral', 'hybrid']:
            result = root/f'results/{task}_{kind}_s{seed}_r8.json'
            if result.exists() and json.loads(result.read_text())['steps'] == steps:
                continue
            command = [sys.executable, str(root/'run.py'), 'train', '--task', task,
                       '--kind', kind, '--seed', str(seed), '--steps', str(steps)]
            print('START', task, kind, seed, flush=True)
            with result.with_suffix('.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            data = json.loads(result.read_text())
            print('DONE', task, kind, seed, data['final'], flush=True)
