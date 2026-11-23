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

Initialization seeds are discovered automatically from each population's
`seed_dir` in `src/cfg/surrogate_coev.py`:

- `sota/Surrogate/seed_models/surrogate/`: XGBoost and MLP surrogate seeds.
- `sota/Surrogate/seed_models/usage/`: BANANAS, NPENAS, and random-selection usage seeds.
- `sota/Surrogate/seed_models/combined/`: joint surrogate and usage seeds.

Each generation-0 individual independently samples a seed uniformly with
replacement from its population directory. Regenerated populations use the same
selection. Discovery includes ordinary `.py` files, excludes names beginning
with `_` (including `__init__.py`), and does not recurse. An empty or missing
directory fails explicitly. Add a compatible seed file to the appropriate
directory to make it available without editing a seed list.

`seed_file` remains the fixed partner when evolution is disabled and the EoT
reference. The baseline runner uses `SEED_DIRECTORY`, currently the combined
seed directory. Selected seed paths are recorded in each individual's job
metadata. Explicit `seed_files`/`SEED_NETWORKS` remain supported when a seed
directory is not configured.

The NPENAS wrapper uses NASLib's mutations of top observed architectures and
ranks proposals by ensemble mean with the injected surrogate. It preserves
training/test bookkeeping and one evaluated model per epoch, with random
proposal fallback if NASLib's rounded mutation count produces no candidates.

The legacy `model.py`, `model_test.py`, `usage_bananas.py`, and `usage_random.py`
seeds have been removed. The evaluator's default/fallback seed is now
`combined/joint_seed.py`. Evaluation helpers remain in `sota/Surrogate/`;
generated variants remain in `models/llmge_models/`.

`surrogate/coev_surrogate_mlp.py` wraps NASLib's `MLPPredictor` with three
128-unit hidden layers, batch size 32, learning rate 0.001, 200 epochs, and MSE
loss by default. It uses the same embedding/PCA wrapper and factory interface
as XGBoost. Directory discovery automatically includes both surrogate seeds;
the fixed surrogate reference remains XGBoost. Combined evolution still uses
its own joint seed directory.

Repeated evaluation is controlled in `src/cfg/constants_surrogate.py`:

```python
numEvols = 20
METRIC_AGGREGATION = "mean"
```

Each candidate evaluation runs `numEvols` fresh evaluations with seeds
`242, 243, ..., 242 + numEvols - 1` by default (the evaluator's `--seed`
changes the starting seed). This applies to independent surrogate metrics,
independent usage metrics, and paired metrics. Each paired repeat performs
one complete search at the existing search budget; surrogate-only repeats
refit on a new disjoint training/test split. Seeds are shared across candidates
for comparable measurements. `--numEvols` can override the count for direct
evaluator calls; it must be a positive integer.

The evaluator retains each metric as a list in `<gene>_results.json` under
`metrics`, alongside `seeds`, `numEvols`, `means`, and detailed `runs`.
`<gene>_results_repeats/` contains per-repeat JSON and logs. The scalar results
file remains the metric means for compatibility with DEAP selection. Raw lists
are also archived as `metric_samples` in job metadata, and independent metric
lists remain in worker checkpoints. Selection averages all repeats before
applying the existing partner reduction/ranking rules. Failed/nonfinite repeats
fail the evaluation rather than silently reducing the sample count. Changing
`numEvols` or aggregation requires a new checkpoint directory.
