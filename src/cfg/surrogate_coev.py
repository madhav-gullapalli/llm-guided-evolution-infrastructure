"""Existing surrogate configuration plus two-population evolution metadata.

Compatible with the repository's constants.py symlink convention. Jobs, LLM
runtime, objectives, evaluator budgets and era settings are inherited unchanged.
"""
from src.cfg.constants_surrogate import *

# Settings for separate populations; seed_file also supplies frozen partners.
POPULATIONS = {
    "surrogate": {
        "seed_file": "sota/Surrogate/coev_surrogate.py",
        "start_population_size": 40, "population_size": 20, "num_elites": 3,
        "crossover_probability": 0.35, "mutation_probability": 0.8,
        "llm_model": "llama3", "prompt_group": "naslib/general",
    },
    "usage": {
        "seed_file": "sota/Surrogate/coev_usage.py",
        "start_population_size": 40, "population_size": 20, "num_elites": 3,
        "crossover_probability": 0.35, "mutation_probability": 0.8,
        "llm_model": "llama3", "prompt_group": "naslib/ranking",
    },
}
# Matching controls apply only when BOTH separate populations evolve.
matches = 40  # Maximum distinct joint evaluations per generation attempt
kappa = 1  # Minimum attempted partners per candidate
COMPOSITION_EXPORTS = {
    "get_surrogate_config": "surrogate",
    "build_predictor_kwargs": "surrogate",
    "SurrogateUsageOptimizer": "usage",
}


def aggregate_pair_scores(values: list[float]) -> float:
    """Mode 4: reduce successful pairs' best accuracies to one candidate score."""
    return max(values)

# A joint genome uses the same operators/settings as an independent worker.
# Both surrogate and usage OPTION blocks occur in this seed.
PAIRED_POPULATION = {**POPULATIONS["surrogate"],
                     "seed_file": "sota/Surrogate/joint_seed.py"}
