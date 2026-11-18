# Experiment configuration

Edit `src/cfg/constants_surrogate.py` for the experiment controls; population
metadata is in `src/cfg/surrogate_coev.py`. The enum and `LLMGE_EVOLUTION_MODE`
are replaced by independent settings:

```python
SEPARATE_POPULATIONS = True  # Separate surrogate and usage populations
SURROGATE_EVOL = True
USAGE_EVOL = True
SURROGATE_METRICS = {"ktau": {"train_size": 100, "test_size": 1000}}
USAGE_METRICS = {"surrogate_usage": {}}
PAIRED_METRICS = {"best_individual_found": {}}
```

Paired evaluation always runs. Paired objectives affect both evolving populations.
A disabled population supplies one fixed seed partner; it generates no LLM
variants, independent metric jobs, or parents. With `SEPARATE_POPULATIONS = False`,
one population evolves a genome containing both components, using
`PAIRED_POPULATION` settings and `joint_seed.py`. The two component evolution
flags and independent metric maps do not participate in joint evolution.

| Map | Metric | Arguments | Direction |
| --- | --- | --- | --- |
| SURROGATE_METRICS | ktau | train_size=100, test_size=1000 | maximize |
| SURROGATE_METRICS | ktau_top_X_percent | train_size=100, test_size=1000, X=10 | maximize |
| SURROGATE_METRICS | mse | train_size=100, test_size=1000 | minimize |
| USAGE_METRICS | surrogate_usage | none | maximize surrogate predictions |
| PAIRED_METRICS | ktau | none | maximize |
| PAIRED_METRICS | ktau_top_X_percent | X=10 | maximize |
| PAIRED_METRICS | best_individual_found | none | maximize validation accuracy |
| PAIRED_METRICS | time_to_find_top_X_percent | X=1 | minimize search steps |

Arguments must be dictionaries, including `{}` for defaults. Empty independent
maps skip their jobs. At least one paired metric is required. Top-percent tau
selects the highest true accuracies (at least two observations); undefined tau
fails evaluation. Time is measured in search steps including initialization;
X=1 means the top 1% of the NASBench-201 validation-accuracy distribution for
the configured dataset. Missing the target scores budget+1.

For paired-only selection, candidates retain their best valid pair-front rank.
When independent objectives are enabled, each paired objective is reduced over
successful partners (maximum for maximizing objectives, minimum for minimizing
objectives), combined with independent objectives, and ranked with DEAP per
population. Independent measurements are cached by source ID; configurations
are recorded in checkpoints and changing metrics requires a new run directory.

The four former modes become:

| Former mode | SURROGATE_EVOL | USAGE_EVOL | SURROGATE_METRICS | USAGE_METRICS | PAIRED_METRICS |
| --- | --- | --- | --- | --- | --- |
| 1 | True | False | {} | {} | ktau, best_individual_found |
| 2 | False | True | {} | {} | ktau, best_individual_found |
| 3 | True | True | {} | {} | ktau, best_individual_found |
| 4 | True | True | ktau | surrogate_usage | best_individual_found |

Run `python run_coevolution.py --checkpoints <directory>` or use the existing
islands controller. With a frozen partner, every evolving candidate is matched to that
partner’s unchanged seed exactly once; `matches` and `kappa` are ignored. Joint evolution evaluates each candidate
once and does not use cross-population matching budgets.
