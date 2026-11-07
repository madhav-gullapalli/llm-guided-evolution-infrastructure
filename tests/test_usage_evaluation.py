"""Oracle predictions and direct performance queries have separate accounting."""
from enum import Enum

import numpy as np
import pytest

from sota.Surrogate.usage_evaluation import EvaluationCounts, PerfectSurrogate, evaluate_usage
from src.coevolution import PairRecord, SearchFitness, INVALID_SEARCH_FITNESS, rank_distinct


class Metric(Enum):
    VAL_ACCURACY = 1
    TEST_ACCURACY = 2
    PARAMETERS = 3


class Architecture:
    def __init__(self, score=90):
        self.score = score

    def query(self, metric, dataset, dataset_api=None):
        assert dataset == 'cifar100'
        return self.score


def test_perfect_surrogate_returns_exact_values_without_training():
    counts = EvaluationCounts()
    predictor = PerfectSurrogate(lambda arch: arch.score, counts)
    assert predictor.fit([], []) == 0
    predictor.set_hyperparams(None)
    assert predictor.query([Architecture(80), Architecture(95)]) == pytest.approx([80, 95])
    assert counts.as_dict() == {'real_evaluations': 0, 'surrogate_predictions': 2,
                                'real_evaluation_fraction': 0}


@pytest.mark.parametrize('budget,expected_real', [(10, 3), (100, 33)])
def test_policy_queries_count_but_oracle_and_setup_do_not(budget, expected_real):
    original = Architecture.query

    class Usage:
        def __init__(self, predictor):
            self.predictors = [predictor() for _ in range(3)]

        def adapt_search_space(self, space, dataset_api=None):
            # Setup/test-set construction is overhead, excluded from fitness.
            space.query(Metric.VAL_ACCURACY, 'cifar100')
            self.space = space

        def before_training(self):
            for predictor in self.predictors:
                predictor.fit([], [])

        def new_epoch(self, epoch):
            for predictor in self.predictors:
                assert predictor.query([self.space, Architecture(95)]) == pytest.approx([90, 95])
            self.space.query(Metric.PARAMETERS, 'cifar100')
            if epoch % 3 == 2:
                self.space.query(Metric.VAL_ACCURACY, 'cifar100')

        def after_training(self):
            pass

    result = evaluate_usage(Usage, Architecture, Metric.VAL_ACCURACY, 'cifar100', {}, budget)
    assert result['real_evaluations'] == expected_real
    assert result['surrogate_predictions'] == 6 * budget
    assert result['real_evaluation_fraction'] == pytest.approx(expected_real / (expected_real + 6 * budget))
    assert result['policy_steps'] == budget
    assert Architecture.query is original


def test_counter_restores_benchmark_on_failure():
    original = Architecture.query

    class Usage:
        def __init__(self, predictor): pass
        def adapt_search_space(self, *args, **kwargs): pass
        def before_training(self): pass
        def new_epoch(self, epoch): raise RuntimeError('bad usage')

    with pytest.raises(RuntimeError, match='bad usage'):
        evaluate_usage(Usage, Architecture, Metric.VAL_ACCURACY, 'cifar100', {}, 1)
    assert Architecture.query is original


def test_usage_minimizes_raw_count_and_requires_successful_oracle_result():
    records = [PairRecord(str(i), {'surrogate': 's', 'usage': f'u{i}'}, (tau, 95),
                          SearchFitness, INVALID_SEARCH_FITNESS)
               for i, tau in enumerate([0.1, 0.9, 1.0])]
    ids = {'surrogate': ['s'], 'usage': ['u0', 'u1', 'u2']}
    surrogate = {'s': {'kendall_tau': 0.8}}
    # Higher real count loses despite a lower fraction and a higher paired tau.
    usage = {'u0': {'real_evaluations': 10, 'surrogate_predictions': 10},
             'u1': {'real_evaluations': 20, 'surrogate_predictions': 10000}}
    ranks = rank_distinct(records, ids, surrogate, usage)
    assert ranks['usage'] == {'u0': 0, 'u1': 1, 'u2': None}
    assert ranks['surrogate'] == {'s': 0}
    assert records[1].fitness.values == (0.9, 95)


def test_pair_accuracy_remains_a_usage_objective():
    records = [PairRecord('a', {'surrogate': 's', 'usage': 'u0'}, (0.1, 90), SearchFitness, INVALID_SEARCH_FITNESS),
               PairRecord('b', {'surrogate': 's', 'usage': 'u1'}, (0.1, 95), SearchFitness, INVALID_SEARCH_FITNESS)]
    ranks = rank_distinct(records, {'surrogate': ['s'], 'usage': ['u0', 'u1']},
                          {'s': {'kendall_tau': 1}},
                          {'u0': {'real_evaluations': 10}, 'u1': {'real_evaluations': 20}})
    # Better accuracy and fewer evaluations trade off; both are nondominated.
    assert ranks['usage'] == {'u0': 0, 'u1': 0}
