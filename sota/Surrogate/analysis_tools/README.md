# Analysis tools (adapted from COLE / cole-dev for this LLM-GE layout)

Scripts live here so `tmp/cole_analysis_scripts/` stays a staging reference.

**Team focus (2026-10-31):** paired coevolution fitness is `(τ, best_accuracy)` via
`run_coevolution.py`. Solo Precision fitness is archived (`tmp/PRECISION_ARCHIVE.md`);
these tools remain for diagnostics and for summarizing archived Precision / MSE runs.

**Chart normalization:** `axis_limits.py` sets shared defaults (τ **0–0.65**, top-10% τ
**0–0.45**, CIFAR-100 accuracy **65–74**). Plot scripts use these unless you pass
`--auto-limits` or explicit `--ylim` / `--xlim`.

## Scripts

| Script | Purpose |
|---|---|
| `plot_search_trajectory.py` | Best-so-far NAS accuracy vs #architectures evaluated (`errors.json`) |
| `plot_surrogate_metrics.py` | Surrogate Kendall τ / MSE over BANANAS retrain steps |
| `plot_tsne_clusters.py` | Embedding map + DBSCAN clusters colored by residuals (default `--method pca`; optional `--method tsne`) |
| `summarize_run.py` | One CSV row per gene for a whole islands run: island, generation, parent, τ/MSE, top-10% τ, Precision@top-10%, parameter-count baseline, best accuracy found by the surrogate-guided picks, evaluations to reach it, evaluations to get within 1% of the best architecture in the search space (COLE's metric) |
| `summarize_coevolution.py` | Independent-score table from coev `history/*.json` (per-population passing / rank-0) |
| `plot_tau_vs_accuracy.py` | τ vs best / mean accuracy of the surrogate-guided picks, one point per gene, with Kendall/Spearman correlation (reads `summary.csv`) |
| `plot_tau_trend.py` | Best / median τ and top-10% τ per generation, with the parameter-count baseline as dashed lines (reads `summary.csv`) |
| `plot_search_outcomes.py` | Reach / evals-to-opt style search outcome plots |
| `seed_rerun.py` | Re-evaluate a run's top genes with extra BANANAS search seeds (GPU jobs) and report mean ± std per gene |

## Summarize a run (from repo root)

```bash
UV_CACHE_DIR=$HOME/.cache/uv uv run python sota/Surrogate/analysis_tools/summarize_run.py --run surrogate_islands_run
UV_CACHE_DIR=$HOME/.cache/uv uv run python sota/Surrogate/analysis_tools/plot_tau_vs_accuracy.py --run surrogate_islands_run
UV_CACHE_DIR=$HOME/.cache/uv uv run python sota/Surrogate/analysis_tools/plot_tau_trend.py --run surrogate_islands_run

# Shared axis limits when comparing two runs on slides:
UV_CACHE_DIR=$HOME/.cache/uv uv run python sota/Surrogate/analysis_tools/plot_tau_trend.py \
  --run surrogate_islands_run --match-run surrogate_precision_run
UV_CACHE_DIR=$HOME/.cache/uv uv run python sota/Surrogate/analysis_tools/plot_tau_trend.py \
  --run surrogate_precision_run --match-run surrogate_islands_run
UV_CACHE_DIR=$HOME/.cache/uv uv run python sota/Surrogate/analysis_tools/plot_tau_vs_accuracy.py \
  --run surrogate_islands_run --match-run surrogate_precision_run
UV_CACHE_DIR=$HOME/.cache/uv uv run python sota/Surrogate/analysis_tools/plot_tau_vs_accuracy.py \
  --run surrogate_precision_run --match-run surrogate_islands_run
```

Writes `<run>/analysis/summary.csv`, `<run>/analysis/tau_vs_accuracy.png` (`--color-by island|mode|generation|mutation`), and `<run>/analysis/tau_trend.png`. Column meanings are in the script docstring. With a fixed search seed the
random initial architectures are identical for every gene, so compare genes on the `guided_*` columns.
`--match-run` forces identical x/y limits across two runs so slide comparisons are not distorted by autoscale.

`top10_tau` is Kendall τ on only the truly best 10% of the held-out test architectures, which is what BANANAS
needs to get right. `precision_top10` is Precision@top-10% (|true top-10% ∩ predicted top-10%| / K), as in
Ning et al. / GATES. `param_count_*` score a no-learning baseline that ranks architectures by parameter
count; a gene below them learned nothing beyond "bigger is better".

## Check scores across search seeds (GPU jobs)

```bash
# --num_init 20 matches COLE (default). Use 10 for NASLib/BANANAS, or 50 to match older evolution runs.
uv run python sota/Surrogate/analysis_tools/seed_rerun.py submit --run surrogate_islands_run --top 5 --seeds 1 2 --num_init 20
# after the seedRerun jobs finish:
uv run python sota/Surrogate/analysis_tools/seed_rerun.py collect --run surrogate_islands_run
```

Evaluates copies named `<gene>__seed<N>` (in `models/seed_reruns/`), so the evolution's own results are never
overwritten. Each job uses one GPU for up to 30 minutes; don't submit while an evolution run is active.
`collect` writes `<run>/analysis/seed_reruns.csv`. New evolution jobs pick up `NUM_INIT` from
`src/cfg/constants_surrogate.py` (default 20); `eval.py --num_init` overrides it for one-off runs.

## Typical run (from repo root)

```bash
UV_CACHE_DIR=$HOME/.cache/uv uv run --with numpy --with scikit-learn --with matplotlib --with scipy --with pandas --with seaborn \
  python sota/Surrogate/analysis_tools/plot_search_trajectory.py \
  --genes seed:seed_mlp \
          xXxG6eiSkPigHllPGknB63fm57V:best \
          xXxKQwdEO2RqgA5YJT4nfGbLzuo:bad_pca \
  --seed 242

UV_CACHE_DIR=$HOME/.cache/uv uv run --with numpy --with scikit-learn --with matplotlib --with scipy --with pandas --with seaborn \
  python sota/Surrogate/analysis_tools/plot_surrogate_metrics.py \
  --genes seed:seed_mlp \
          xXxG6eiSkPigHllPGknB63fm57V:best \
          xXxKQwdEO2RqgA5YJT4nfGbLzuo:bad_pca \
  --seed 242

UV_CACHE_DIR=$HOME/.cache/uv uv run --with numpy --with scikit-learn --with matplotlib --with scipy --with pandas --with seaborn \
  python sota/Surrogate/analysis_tools/plot_tsne_clusters.py \
  --gene xXxG6eiSkPigHllPGknB63fm57V
```

Outputs default to `sota/Surrogate/results/analysis_tools/`.
