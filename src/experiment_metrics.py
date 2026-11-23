"""Declarative experiment metrics and shared DEAP objective construction."""
from dataclasses import dataclass
import math
import numpy as np
from deap import base
from scipy.stats import kendalltau
from src.repeated_evaluation import mean_metric, validate_num_evols

SCOPES = {
    'surrogate': {'ktau': 1., 'ktau_top_X_percent': 1., 'mse': -1.},
    'usage': {'surrogate_usage': 1.},
    'paired': {'ktau': 1., 'ktau_top_X_percent': 1., 'best_individual_found': 1.,
               'time_to_find_top_X_percent': -1.},
}


def validate_metrics(scope, metrics):
    if not isinstance(metrics, dict):
        raise ValueError(f'{scope} metrics must be a dictionary')
    for name, options in metrics.items():
        if name not in SCOPES[scope] or not isinstance(options, dict):
            raise ValueError(f'Invalid {scope} metric {name!r}: expected a supported name and argument map')
        allowed = {'train_size', 'test_size'} if scope == 'surrogate' else set()
        if 'top_X_percent' in name:
            allowed |= {'X'}
            x = options.get('X', 10 if name.startswith('ktau') else 1)
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 < x <= 100:
                raise ValueError('X must be in (0, 100]')
        if set(options) - allowed:
            raise ValueError(f'Unknown arguments for {name}: {set(options) - allowed}')
        for key in ('train_size', 'test_size'):
            value = options.get(key, 100 if key == 'train_size' else 1000)
            if scope == 'surrogate' and (type(value) is not int or value < 2):
                raise ValueError(f'{key} must be an integer >= 2')
        if scope == 'surrogate' and options.get('train_size', 100) + options.get('test_size', 1000) > 15625:
            raise ValueError('Training and test samples exceed NASBench-201')


def validate_config(config):
    validate_num_evols(getattr(config, "numEvols", 1))
    if getattr(config, "METRIC_AGGREGATION", "mean") != "mean":
        raise ValueError("Only mean metric aggregation is supported")
    for key in ('SEPARATE_POPULATIONS', 'SURROGATE_EVOL', 'USAGE_EVOL'):
        if type(getattr(config, key)) is not bool:
            raise ValueError(f'{key} must be boolean')
    for scope in SCOPES:
        validate_metrics(scope, getattr(config, scope.upper() + '_METRICS'))
    if not config.PAIRED_METRICS:
        raise ValueError('At least one paired metric is required')
    if config.SEPARATE_POPULATIONS and not (config.SURROGATE_EVOL or config.USAGE_EVOL):
        raise ValueError('Enable at least one population evolution step')


class MetricFitness(base.Fitness):
    """Instance weights keep arbitrary objective vectors pickle-compatible."""
    def __deepcopy__(self, memo):
        return MetricFitness(self.values, self.weights)

    def __init__(self, values=(), weights=()):
        self.weights = tuple(weights)
        super().__init__(values)


@dataclass(frozen=True)
class FitnessFactory:
    weights: tuple

    def __call__(self, values=()):
        return MetricFitness(values, self.weights)


def fitness_type(scope, metrics):
    return FitnessFactory(tuple(SCOPES[scope][name] for name in metrics))


def top_tau(targets, predictions, percent):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    indices = np.argsort(targets)[-max(2, math.ceil(len(targets) * percent / 100)):]
    value = float(kendalltau(targets[indices], predictions[indices]).statistic)
    if not math.isfinite(value):
        raise ValueError('Undefined top-percent Kendall tau')
    return value


def paired_values(metrics, tau, search, optimizer, dataset_api, dataset):
    values = []
    for name, options in metrics.items():
        if name == 'ktau':
            value = tau
        elif name == 'ktau_top_X_percent':
            value = top_tau(optimizer.paired_targets, optimizer.paired_predictions, options.get('X', 10))
        elif name == 'best_individual_found':
            value = search['best_accuracy']
        elif name == 'time_to_find_top_X_percent':
            # X is the top fraction: X=1 means the 99th percentile.
            key = 'cifar10-valid' if dataset in ('cifar10', 'cifar10-valid') else dataset
            field = 'eval_acc1es'
            threshold = np.percentile([row[key][field][-1]
                                      for row in dataset_api['nb201_data'].values()],
                                     100 - options.get('X', 1))
            trajectory = search['trajectory']
            value = next((i for i, accuracy in enumerate(trajectory, 1) if accuracy >= threshold),
                         len(trajectory) + 1)
        values.append(float(value))
    return values


def configured_objectives(records, candidate_ids, individual_metrics, config):
    """Reduce paired objectives over partners, then add independent objectives."""
    paired_names = list(config.PAIRED_METRICS)
    paired_weights = [SCOPES['paired'][name] for name in paired_names]
    objectives, ranks = {}, {}
    from src.coevolution import PairRecord, rank_pairs
    pair_ranks = rank_pairs(records, candidate_ids)
    for population, genes in candidate_ids.items():
        metrics = (getattr(config, population.upper() + '_METRICS', {})
                   if population != 'paired' else {})
        weights = tuple(paired_weights + [SCOPES[population][name] for name in metrics])
        fitness = FitnessFactory(weights)
        vectors = {}
        for gene in genes:
            pairs = [record.fitness.values for record in records
                     if record.success and record.source_ids.get(population) == gene]
            if not pairs:
                continue
            standalone = individual_metrics.get(population, {}).get(gene, {})
            if any(name not in standalone for name in metrics):
                continue
            vector = tuple((max if weight > 0 else min)(p[i] for p in pairs)
                           for i, weight in enumerate(paired_weights))
            try:
                vector += tuple(mean_metric(standalone[name]) for name in metrics)
            except ValueError:
                continue
            if all(math.isfinite(v) for v in vector):
                vectors[gene] = vector
        candidates = [PairRecord(gene, {population: gene}, vector, fitness, ())
                      for gene, vector in vectors.items()]
        ranks[population] = (rank_pairs(candidates, {population: genes})[population]
                             if metrics else pair_ranks[population])
        objectives[population] = vectors
    return ranks, objectives
