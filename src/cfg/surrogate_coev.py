"""Existing surrogate configuration plus two-population evolution metadata.

Compatible with the repository's constants.py symlink convention. Jobs, LLM
runtime, objectives, evaluator budgets and era settings are inherited unchanged.
"""
from src.cfg.constants_surrogate import *

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
matches = 40  # Maximum distinct joint evaluations per generation attempt
kappa = 1
COMPOSITION_EXPORTS = {
    "get_surrogate_config": "surrogate",
    "build_predictor_kwargs": "surrogate",
    "SurrogateUsageOptimizer": "usage",
}
