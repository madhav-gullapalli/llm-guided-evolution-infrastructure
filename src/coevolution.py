"""Generation-local matching and DEAP selection adapters for two populations."""
import random
import math
from collections import Counter
from deap import base, tools


SEARCH_BUDGET = 100
OBJECTIVES = ("kendall_tau", "best_validation_accuracy_at_100")


class SearchFitness(base.Fitness):
    weights = (1.0, 1.0)


INVALID_SEARCH_FITNESS = (-float("inf"), -float("inf"))


def best_search_accuracy(accuracies, budget=SEARCH_BUDGET):
    """Best observed validation accuracy, including random initialization."""
    values = list(accuracies)[:budget]
    if len(values) != budget or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected {budget} finite search validation accuracies")
    return max(values)


def validate_pairing(a, b, matches, kappa):
    """Return the capped match count, requiring minimum attempted coverage."""
    if (not isinstance(kappa, int) or not isinstance(matches, int)
            or not 1 <= kappa <= min(a, b)
            or matches < kappa * max(a, b)):
        raise ValueError(f"Infeasible pairing: a={a}, b={b}, matches={matches}, kappa={kappa}")
    return min(matches, a * b)


def random_uniform_coverage(left, right, matches, kappa, rng=None):
    """Shuffle, cover cyclically, then sample unused cross-product edges."""
    left, right = list(left), list(right)
    a, b = len(left), len(right)
    if len(set(left)) != a or len(set(right)) != b:
        raise ValueError("Candidate IDs must be distinct within each population")
    matches = validate_pairing(a, b, matches, kappa)
    rng = rng or random
    rng.shuffle(left)
    rng.shuffle(right)
    swapped = a < b
    large, small = (right, left) if swapped else (left, right)
    pairs = []
    for offset in range(kappa):
        for j, member in enumerate(large):
            pair = (member, small[(j + offset) % len(small)])
            pairs.append(pair[::-1] if swapped else pair)
    used = set(pairs)
    unused = [(x, y) for x in left for y in right if (x, y) not in used]
    pairs.extend(rng.sample(unused, matches - len(pairs)))
    return pairs


class PairRecord:
    def __init__(self, match_id, source_ids, values, fitness_type, invalid):
        self.match_id = match_id
        self.source_ids = source_ids
        self.fitness = fitness_type()
        self.fitness.values = values
        self.success = tuple(values) != tuple(invalid)
        self.pareto_level = None

    def as_dict(self):
        return {"match_id": self.match_id, "source_ids": self.source_ids,
                "success": self.success, "objectives": self.fitness.values,
                "pareto_level": self.pareto_level}


def rank_pairs(records, candidate_ids):
    """Sort all attempted pairs; project only successful pairs to individuals."""
    ranks = {name: {gene_id: None for gene_id in ids}
             for name, ids in candidate_ids.items()}
    for level, front in enumerate(tools.sortNondominated(records, len(records))):
        for record in front:
            record.pareto_level = level
            if record.success:
                for name, gene_id in record.source_ids.items():
                    current = ranks[name][gene_id]
                    ranks[name][gene_id] = level if current is None else min(level, current)
    return ranks


def barrier_passes(ranks, populations):
    """Advance only after each population exceeds its cumulative success minimum."""
    return all(sum(rank is not None for rank in ranks[name].values()) > cfg["num_elites"]
               for name, cfg in populations.items())


class RankFitness(base.Fitness):
    weights = (-1.0,)


class RankProxy(list):
    def __init__(self, individual, rank):
        super().__init__(individual)
        self.fitness = RankFitness((rank,))


def rank_statistics(ranks, records):
    return {
        "pair_front_counts": dict(Counter(record.pareto_level for record in records)),
        "populations": {
            name: {"passing": sum(rank is not None for rank in values.values()),
                   "rank_0": sum(rank == 0 for rank in values.values()),
                   "rank_counts": dict(Counter(rank for rank in values.values() if rank is not None))}
            for name, values in ranks.items()
        },
    }
