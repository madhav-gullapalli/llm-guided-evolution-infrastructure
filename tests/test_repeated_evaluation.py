"""Repeated evaluation preserves measurements while selecting on their means."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.repeated_evaluation import mean_metric, repeat_evaluation, validate_num_evols


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '20'])
def test_invalid_repeat_count(value):
    with pytest.raises(ValueError):
        validate_num_evols(value)


def test_twenty_samples_use_fresh_seeds_and_mean(tmp_path, monkeypatch, capsys):
    result = tmp_path / 'gene_results.txt'
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        seed = int(command[command.index('--seed') + 1])
        assert command[command.index('--numEvols') + 1] == '1'
        assert '--single-evaluation' in command
        assert not result.exists()  # Never reuse an earlier result.
        result.write_text(f'{seed},{seed * 2}')
        result.with_suffix('.json').write_text(json.dumps({'seed': seed}))
        kwargs['stdout'].write('job done\n')
    monkeypatch.setattr(subprocess, 'run', run)
    data = repeat_evaluation(['python', 'eval.py'], result, ['ktau', 'mse'], 20, 242)
    assert len(commands) == 20
    assert data['metrics']['ktau'] == list(range(242, 262))
    assert data['means'] == {'ktau': 251.5, 'mse': 503.0}
    assert data['seeds'] == list(range(242, 262))
    assert len(data['runs']) == 20
    assert json.loads(result.with_suffix('.json').read_text()) == data
    assert result.read_text() == '251.5,503.0'
    assert len(list((tmp_path / 'gene_results_repeats').glob('*.json'))) == 20
    assert 'job done' not in capsys.readouterr().out


def test_real_child_processes_are_independent(tmp_path):
    script = tmp_path / 'child.py'
    result = tmp_path / 'results.txt'
    script.write_text('''import argparse, os
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int)
args, _ = parser.parse_known_args()
Path(__file__).with_name('results.txt').write_text(f'{args.seed},{os.getpid()}')
print('job done')
''')
    data = repeat_evaluation([sys.executable, str(script)], result, ['score', 'pid'], 3, 42)
    assert data['metrics']['score'] == [42, 43, 44]
    assert len(set(data['metrics']['pid'])) == 3
    assert data['means']['score'] == 43


def test_failed_repeat_is_not_silently_dropped(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        repeat_evaluation(['eval'], tmp_path / 'out.txt', ['score'], 20, 242)
    assert not (tmp_path / 'out.txt').exists()


@pytest.mark.parametrize('values', [[], [float('nan')], [float('inf')], [[1, 2]]])
def test_invalid_samples(values):
    with pytest.raises(ValueError):
        mean_metric(values)


def test_selection_uses_all_samples():
    from src.coevolution import PairRecord
    from src.experiment_metrics import configured_objectives, fitness_type

    config = SimpleNamespace(PAIRED_METRICS={'best_individual_found': {}},
                             SURROGATE_METRICS={'mse': {}}, USAGE_METRICS={})
    factory = fitness_type('paired', config.PAIRED_METRICS)
    records = [PairRecord('p1', {'surrogate': 's1'}, (90,), factory, ()),
               PairRecord('p2', {'surrogate': 's2'}, (90,), factory, ())]
    ranks, vectors = configured_objectives(records, {'surrogate': ['s1', 's2']},
        {'surrogate': {'s1': {'mse': [1, 9]}, 's2': {'mse': [3, 3]}}}, config)
    assert vectors['surrogate']['s1'] == (90, 5)
    assert ranks['surrogate'] == {'s1': 1, 's2': 0}


def test_config_defaults_to_twenty_mean_evaluations():
    from src.cfg import constants_surrogate as config
    assert config.numEvols == 20
    assert config.METRIC_AGGREGATION == 'mean'


@pytest.mark.parametrize('payload', ['nan', '1,2', None])
def test_partial_invalid_results_are_removed(tmp_path, monkeypatch, payload):
    result = tmp_path / 'results.txt'
    def run(*args, **kwargs):
        if payload is not None:
            result.write_text(payload)
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises((ValueError, FileNotFoundError)):
        repeat_evaluation(['eval'], result, ['score'], 2, 242)
    assert not result.exists()


@pytest.mark.parametrize('scope,names', [
    ('surrogate_individual', ['ktau', 'mse']),
    ('usage_individual', ['real_evaluations', 'surrogate_usage']),
    ('coevolution', ['best_individual_found', 'ktau']),
])
def test_evaluator_routes_every_scope_through_repeats(tmp_path, scope, names):
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'sota/Surrogate/eval.py'
    tree = ast.parse(path.read_text())
    block = next(node for node in tree.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == 'not custom_args.single_evaluation')
    calls = []
    args = SimpleNamespace(single_evaluation=False, numEvols=20, seed=242,
                           model='model_candidate', metric_options='{}',
                           surrogate_individual=False, usage_individual=False, coevolution=False)
    setattr(args, scope, True)
    namespace = {'custom_args': args, 'Path': Path, 'SURROGATE_RESULTS_DIR': str(tmp_path),
                 'PAIRED_METRICS': {'best_individual_found': {}, 'ktau': {}},
                 'repeat_evaluation': lambda *values: calls.append(values),
                 'METRIC_AGGREGATION': 'mean', 'sys': sys, '__file__': str(path)}
    with pytest.raises(SystemExit) as error:
        exec(compile(ast.Module(body=[block], type_ignores=[]), str(path), 'exec'), namespace)
    assert error.value.code == 0
    assert len(calls) == 1
    assert calls[0][1] == tmp_path / 'candidate_results.txt'
    assert calls[0][2:] == (names, 20, 242, 'mean')
