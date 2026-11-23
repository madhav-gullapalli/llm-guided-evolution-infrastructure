import copy
import pickle
from types import SimpleNamespace

import pytest
from src.experiment_metrics import (fitness_type, validate_config, validate_metrics,
                                    paired_values, configured_objectives, top_tau)
from src.coevolution import PairRecord


def test_metric_fitness_round_trip_and_copy():
    factory = fitness_type('paired', {'best_individual_found': {}, 'time_to_find_top_X_percent': {}})
    value = factory((90, 20))
    for restored in (copy.deepcopy(value), pickle.loads(pickle.dumps(value))):
        assert restored.weights == (1, -1)
        assert restored.values == (90, 20)


@pytest.mark.parametrize('scope,metrics', [
    ('paired', {'bogus': {}}), ('usage', {'surrogate_usage': []}),
    ('surrogate', {'ktau': {'train_size': 1}}),
    ('paired', {'ktau_top_X_percent': {'X': 0}}),
    ('paired', {'ktau': {'train_size': 10}}),
])
def test_invalid_metrics(scope, metrics):
    with pytest.raises(ValueError):
        validate_metrics(scope, metrics)


def test_percentile_time_and_top_tau():
    metrics = {'best_individual_found': {}, 'time_to_find_top_X_percent': {'X': 25},
               'ktau_top_X_percent': {'X': 50}}
    api = {'nb201_data': {str(i): {'cifar100': {'eval_acc1es': [v]}}
                          for i, v in enumerate([10, 20, 30, 40])}}
    optimizer = SimpleNamespace(paired_targets=[1, 2, 3, 4], paired_predictions=[1, 2, 4, 3])
    values = paired_values(metrics, .5, {'best_accuracy': 40, 'trajectory': [10, 30, 40]},
                           optimizer, api, 'cifar100')
    assert values == pytest.approx([40, 3, -1])
    values = paired_values({'time_to_find_top_X_percent': {'X': 1}}, 0,
                           {'trajectory': [10, 20]}, optimizer, api, 'cifar100')
    assert values == [3]  # A finite budget+1 penalty when the target is missed.


def test_paired_objectives_affect_both_populations():
    config = SimpleNamespace(PAIRED_METRICS={'best_individual_found': {}},
        SURROGATE_METRICS={'mse': {}}, USAGE_METRICS={'surrogate_usage': {}})
    factory = fitness_type('paired', config.PAIRED_METRICS)
    records = [PairRecord('p1', {'surrogate': 's1', 'usage': 'u1'}, (90,), factory, ()),
               PairRecord('p2', {'surrogate': 's2', 'usage': 'u2'}, (80,), factory, ())]
    ranks, vectors = configured_objectives(records, {'surrogate': ['s1', 's2'], 'usage': ['u1', 'u2']},
        {'surrogate': {'s1': {'mse': 1}, 's2': {'mse': 2}},
         'usage': {'u1': {'surrogate_usage': 20}, 'u2': {'surrogate_usage': 10}}}, config)
    assert ranks == {'surrogate': {'s1': 0, 's2': 1}, 'usage': {'u1': 0, 'u2': 1}}
    assert vectors['surrogate']['s1'] == (90, 1)


def test_joint_seed_has_both_edit_blocks_and_fixed_imports():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'sota/Surrogate/seed_models/combined/joint_seed.py'
    header, surrogate, usage = path.read_text().split('# --OPTION--')
    assert 'import Bananas' in header
    assert 'import XGBoost' in header
    assert 'def get_surrogate_config' in surrogate
    assert 'class SurrogateUsageOptimizer' in usage
