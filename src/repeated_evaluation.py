"""Repeat complete evaluator processes, retaining samples and scalar fitness."""
import json
from pathlib import Path
import subprocess

import numpy as np


def validate_num_evols(value):
    if type(value) is not int or value < 1:
        raise ValueError('numEvols must be a positive integer')
    return value


def mean_metric(value):
    values = np.asarray(value, dtype=float)
    if values.size == 0 or values.ndim > 1 or not np.isfinite(values).all():
        raise ValueError('Metric samples must be nonempty and finite')
    return float(np.mean(values))


def repeat_evaluation(command, result_path, metric_names, num_evols, base_seed,
                      aggregation='mean'):
    """Run one complete evaluation per seed; never count missing/failed runs."""
    validate_num_evols(num_evols)
    if aggregation != 'mean':
        raise ValueError('Only mean metric aggregation is supported')
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = result_path.parent / (result_path.stem + '_repeats')
    artifacts.mkdir(exist_ok=True)
    samples = {name: [] for name in metric_names}
    runs = []
    seeds = [base_seed + i for i in range(num_evols)]
    for index, seed in enumerate(seeds):
        result_path.unlink(missing_ok=True)
        result_path.with_suffix('.json').unlink(missing_ok=True)
        # Suppress child completion markers: only the parent may report job done.
        try:
            with (artifacts / f'{index:03d}_seed_{seed}.log').open('w') as log:
                subprocess.run([*command, '--single-evaluation', '--numEvols', '1', '--seed', str(seed), '--trials', '1'],
                               check=True, stdout=log, stderr=subprocess.STDOUT)
            values = [float(value.strip()) for value in result_path.read_text().split(',')]
            if len(values) != len(metric_names):
                raise ValueError('Repeated evaluator returned the wrong number of metrics')
            for name, value in zip(metric_names, values):
                mean_metric(value)  # Reject nonfinite results instead of dropping runs.
                samples[name].append(value)
            detail_path = result_path.with_suffix('.json')
            details = json.loads(detail_path.read_text()) if detail_path.exists() else {}
            run = {'seed': seed, 'metrics': dict(zip(metric_names, values)), 'details': details}
        except Exception:
            # Never leave a partial child result looking like an aggregate.
            result_path.unlink(missing_ok=True)
            result_path.with_suffix('.json').unlink(missing_ok=True)
            raise
        runs.append(run)
        (artifacts / f'{index:03d}_seed_{seed}.json').write_text(json.dumps(run, indent=2) + '\n')
        print(f'Evaluation repeat {index + 1}/{num_evols} completed (seed {seed})', flush=True)
    means = {name: mean_metric(values) for name, values in samples.items()}
    result = {'numEvols': num_evols, 'aggregation': aggregation, 'seeds': seeds,
              'metrics': samples, 'means': means, 'runs': runs}
    result_path.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    result_path.write_text(','.join(str(means[name]) for name in metric_names))
    return result
