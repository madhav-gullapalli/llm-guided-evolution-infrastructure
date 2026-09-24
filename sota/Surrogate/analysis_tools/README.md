# Analysis tools (adapted from COLE / cole-dev for this LLM-GE layout)

Scripts live here so `tmp/cole_analysis_scripts/` stays a staging reference.

## Scripts

| Script | Purpose |
|---|---|
| `plot_search_trajectory.py` | Best-so-far NAS accuracy vs #architectures evaluated (`errors.json`) |
| `plot_surrogate_metrics.py` | Surrogate Kendall τ / MSE over BANANAS retrain steps |
| `plot_tsne_clusters.py` | Embedding map + DBSCAN clusters colored by residuals (default `--method pca`; optional `--method tsne`) |

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
