"""Independent surrogate measurements and population-specific objectives."""
import copy

import numpy as np
import pytest

from sota.Surrogate.surrogate_evaluation import architecture_split, evaluate_surrogate
from src.coevolution import (PairRecord, SearchFitness, INVALID_SEARCH_FITNESS,
                             rank_distinct)


def test_split_is_uniform_reproducible_disjoint_and_does_not_change_rng():
    import random
    state = random.getstate()
    train, test = architecture_split()
    assert random.getstate() == state
    assert (train, test) == architecture_split()
    assert len(set(train)) == 100
    assert len(set(test)) == 1000
    assert not set(train) & set(test)
    assert (train, test) != architecture_split(seed=243)
    assert all(len(spec) == 6 and all(0 <= op < 5 for op in spec) for spec in train + test)


def test_measurement_fits_only_training_and_queries_only_held_out():
    fits, queries = [], []

    def accuracy(spec):
        return sum(op * 5 ** i for i, op in enumerate(spec)) / 15625

    class Predictor:
        def fit(self, train, labels):
            fits.append(list(train))
            assert labels == pytest.approx([accuracy(spec) for spec in train])

        def query(self, test):
            queries.append(list(test))
            return [accuracy(spec) for spec in test]

    metrics = evaluate_surrogate(Predictor, tuple, accuracy)
    assert metrics['kendall_tau'] == pytest.approx(1)
    assert metrics['mse'] == pytest.approx(0)
    train, test = architecture_split()
    assert fits == [train] * 3
    assert queries == [test] * 3


@pytest.mark.parametrize('prediction', [float('nan'), 1.0])
def test_nonfinite_or_constant_predictions_fail(prediction):
    class Predictor:
        def fit(self, train, labels):
            pass

        def query(self, test):
            return np.full(len(test), prediction)

    with pytest.raises(ValueError):
        evaluate_surrogate(Predictor, tuple, lambda spec: sum(spec))


def pair(match, surrogate, usage, values):
    return PairRecord(match, {'surrogate': surrogate, 'usage': usage}, values,
                      SearchFitness, INVALID_SEARCH_FITNESS)


def test_populations_use_their_own_independent_objectives():
    records = [pair('a', 's0', 'u0', (0.9, 90)), pair('b', 's1', 'u1', (0.1, 90))]
    ids = {'surrogate': ['s0', 's1'], 'usage': ['u0', 'u1']}
    individual = {'s0': {'kendall_tau': 0.1}, 's1': {'kendall_tau': 0.9}}
    ranks = rank_distinct(records, ids, individual,
                          {"u0": {"real_evaluations": 10}, "u1": {"real_evaluations": 20}})
    assert ranks == {'surrogate': {'s0': 1, 's1': 0}, 'usage': {'u0': 0, 'u1': 1}}
    assert [r.fitness.values for r in records] == [(0.9, 90), (0.1, 90)]
    individual['s0']['kendall_tau'] = float('nan')
    assert rank_distinct(records, ids, individual, {})['surrogate']['s0'] is None


def test_missing_individual_score_does_not_invalidate_usage_pair():
    records = [pair('a', 's0', 'u0', (0.9, 90)),
               pair('bad', 's1', 'u1', INVALID_SEARCH_FITNESS),
               pair('old', 'old_s', 'old_u', (1, 100))]
    ranks = rank_distinct(records, {'surrogate': ['s0', 's1'], 'usage': ['u0', 'u1']},
                          {'s1': {'kendall_tau': 1}}, {'u0': {'real_evaluations': 10}})
    assert ranks['surrogate'] == {'s0': None, 's1': None}
    assert ranks['usage'] == {'u0': 0, 'u1': None}


def test_individual_job_cache_and_checkpoint_configuration(tmp_path, monkeypatch):
    import run_coevolution as coev
    monkeypatch.setattr(coev.config, 'MODE', 4)
    worker = coev.PopulationWorker('surrogate', coev.config.POPULATIONS['surrogate'], tmp_path)
    worker.source('s0').write_text('def get_surrogate_config(**kwargs): return kwargs\n')
    worker.population = [coev.legacy.creator.Individual(['s0'])]
    submissions = []

    def evaluate(individuals):
        assert '--surrogate_individual' in coev.legacy.EVAL_RUNLINE
        assert '--coevolution' not in coev.legacy.EVAL_RUNLINE
        submissions.extend(individuals)
        for ind in individuals:
            ind.fitness.values = (0.75, 0.01)

    monkeypatch.setattr(coev.legacy, 'check_and_update_fitness', evaluate)
    monkeypatch.setattr(coev, 'save_intermediate', lambda *args: None)
    workers = {'surrogate': worker}
    coev.evaluate_individuals(workers, 1, 1, tmp_path)
    assert worker.individual_metrics['s0']['kendall_tau'] == 0.75
    saved = worker.snapshot()
    worker.individual_metrics = {}
    worker.restore(saved)
    coev.evaluate_individuals(workers, 2, 1, tmp_path)
    assert len(submissions) == 1
    distinct = copy.deepcopy(coev.configuration_snapshot())
    assert distinct['individual_evaluation']['test_size'] == 1000
    monkeypatch.setattr(coev.config, 'MODE', 3)
    assert 'individual_evaluation' not in coev.configuration_snapshot()
    assert not coev.configuration_compatible(distinct)


def test_pair_scores_are_aggregated_before_candidate_pareto_ranking():
    from src.coevolution import distinct_objectives
    from src.cfg.surrogate_coev import aggregate_pair_scores
    records = [pair('a', 's0', 'u0', (0.1, 100)),
               pair('b', 's0', 'u0', (0.9, 0)),
               pair('c', 's1', 'u1', (0.2, 90)),
               pair('d', 's1', 'u1', (0.8, 90)),
               pair('failed', 's1', 'u1', INVALID_SEARCH_FITNESS)]
    ids = {'surrogate': ['s0', 's1', 'missing'], 'usage': ['u0', 'u1']}
    surrogates = {gene: {'kendall_tau': 0.5} for gene in ids['surrogate']}
    usage = {gene: {'real_evaluations': 10} for gene in ids['usage']}
    assert aggregate_pair_scores([100.0, 0.0]) == 100
    values = distinct_objectives(records, ids, surrogates, usage, aggregate_pair_scores)
    assert values == {'surrogate': {'s0': (0.5, 100), 's1': (0.5, 90)},
                      'usage': {'u0': (100, 10), 'u1': (90, 10)}}
    ranks = rank_distinct(records, ids, surrogates, usage, aggregate_pair_scores)
    assert ranks == {'surrogate': {'s0': 0, 's1': 1, 'missing': None},
                     'usage': {'u0': 0, 'u1': 1}}
    calls = []

    def mean(values):
        calls.append(values[:])
        return sum(values) / len(values)

    ranks = rank_distinct(records, ids, surrogates, usage, mean)
    assert ranks == {'surrogate': {'s0': 1, 's1': 0, 'missing': None},
                     'usage': {'u0': 1, 'u1': 0}}
    assert calls == [[100, 0], [90, 90], [100, 0], [90, 90]]
    with pytest.raises(ValueError, match='finite'):
        rank_distinct(records, ids, surrogates, usage, lambda values: float('nan'))


def test_pair_aggregation_preserves_scores_with_earlier_partners():
    from src.coevolution import distinct_objectives
    records = [pair('old', 's0', 'dropped_usage', (0.1, 95)),
               pair('new', 's0', 'u0', (0.2, 90))]
    values = distinct_objectives(records, {'surrogate': ['s0'], 'usage': ['u0']},
                                 {'s0': {'kendall_tau': 0.5}},
                                 {'u0': {'real_evaluations': 10}})
    assert values == {'surrogate': {'s0': (0.5, 95)}, 'usage': {'u0': (90, 10)}}
