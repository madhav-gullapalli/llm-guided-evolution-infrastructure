"""NPENAS policy behavior without benchmark data or external LLM jobs."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def seed(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / 'sota/Surrogate'))
    spec = importlib.util.spec_from_file_location('npenas_seed_test', REPO / 'sota/Surrogate/seed_models/usage/coev_usage_npenas.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Architecture:
    def __init__(self, label=0):
        self.label = label
    def clone(self):
        return Architecture(self.label)
    def mutate(self, parent, dataset_api=None):
        self.label = parent.label + 1
    def get_hash(self):
        return self.label


def model(label, accuracy):
    result = torch.nn.Module()
    result.arch = Architecture(label)
    result.accuracy = accuracy
    return result


def test_npenas_mutates_top_parents_and_ranks_by_mean(seed):
    policy = seed.SurrogateUsageOptimizer.__new__(seed.SurrogateUsageOptimizer)
    policy.train_data = [model(0, 10), model(20, 90), model(40, 80)]
    policy.num_arches_to_mutate = 2
    policy.max_mutations = 2
    policy.num_candidates = 8
    policy.search_space = SimpleNamespace(clone=Architecture, instantiate_model=False)
    policy.dataset_api = None
    policy.acq_fn_optimization = 'random_sampling'  # NPENAS must still mutate.
    candidates = policy._get_new_candidates([10, 90, 80])
    assert len(candidates) == 4  # NASLib's mutation-round formula.
    assert {c.arch.label for c in candidates} == {22, 24, 42, 44}
    policy.k = 2
    policy.ensemble = SimpleNamespace(query=lambda architectures, info: np.array([
        [arch.label for arch in architectures],
        [arch.label + 2 for arch in architectures],
    ]))
    assert policy._acquisition([])([Architecture(1), Architecture(9)]).tolist() == [2, 10]
    selected = policy._get_best_candidates(candidates, policy._acquisition([]))
    assert [candidate.arch.label for candidate in selected] == [42, 44]


def test_epoch_injects_predictor_and_evaluates_one_candidate(seed, monkeypatch):
    ensembles = []
    class Ensemble:
        def __init__(self, **kwargs):
            ensembles.append(self)
        def fit(self, xs, ys):
            assert xs == ['init'] and ys == [70]
        def query(self, architectures, info=None):
            return np.array([[1, 5], [3, 7]])
    monkeypatch.setattr(seed, 'Ensemble', Ensemble)
    injected = []
    def predictor(**kwargs):
        injected.append(kwargs)
        return object()
    policy = seed.SurrogateUsageOptimizer.__new__(seed.SurrogateUsageOptimizer)
    policy.num_init, policy.k, policy.num_ensemble = 1, 2, 2
    policy.predictor_cls, policy.predictor_kwargs = predictor, {'token': 7}
    policy.ss_type, policy.predictor_type = 'nasbench201', 'xgb'
    policy.zc, policy.zc_only, policy.config = False, False, None
    policy.train_data, policy.next_batch = [], []
    policy._sample_new_model = lambda: SimpleNamespace(arch='init', arch_hash='init')
    removed = []
    policy._remove_from_test_set = removed.append
    def evaluate(candidate):
        candidate.accuracy = 70
        policy.train_data.append(candidate)
    policy._set_scores = evaluate
    policy._get_new_candidates = lambda ys: [SimpleNamespace(arch=label, arch_hash=label) for label in ['low', 'high']]
    for epoch in range(3):
        policy.new_epoch(epoch)
        assert len(policy.train_data) == epoch + 1
    assert removed == ['init', 'high', 'low']
    assert injected == [{'token': 7}, {'token': 7}]
    assert len(ensembles[0].ensemble) == 2


def test_seed_enabled_with_bananas():
    from src.cfg.surrogate_coev import POPULATIONS
    from src.seed_models import discover_seed_models
    paths = discover_seed_models(REPO / POPULATIONS['usage']['seed_dir'])
    assert 'coev_usage.py' in [Path(path).name for path in paths]
    assert 'coev_usage_npenas.py' in [Path(path).name for path in paths]


def test_small_mutation_budget_falls_back_without_evaluating_proposals(seed):
    policy = seed.SurrogateUsageOptimizer.__new__(seed.SurrogateUsageOptimizer)
    policy.train_data = [model(0, 90)]
    policy.num_arches_to_mutate, policy.max_mutations, policy.num_candidates = 1, 4, 1
    policy._sample_new_model = lambda: SimpleNamespace(arch_hash='fallback')
    candidates = policy._get_new_candidates([90])
    assert [candidate.arch_hash for candidate in candidates] == ['fallback']
    assert len(policy.train_data) == 1
