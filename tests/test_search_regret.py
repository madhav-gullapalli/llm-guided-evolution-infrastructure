"""Anytime regret helpers: within-1%-of-opt ≠ top-1% of the space."""
import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "sota/Surrogate/analysis_tools/search_regret.py"
    spec = importlib.util.spec_from_file_location("search_regret_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regret_and_near_opt_hit():
    mod = load_module()
    acc = [70.0, 71.0, 72.6, 72.6]
    optimum = 73.26
    regrets = mod.regret_curve(acc, optimum)
    assert list(regrets) == [
        optimum - 70.0,
        optimum - 71.0,
        optimum - 72.6,
        optimum - 72.6,
    ]
    near = mod.near_opt_threshold(optimum)
    assert near == 0.99 * optimum
    assert mod.evals_to_threshold(acc, near) == 3
    row = mod.summarize_trajectory(acc, optimum, num_init=2, top_threshold=71.18)
    assert row["within_1pct_opt"] is True
    assert row["evals_to_within_1pct_opt"] == 3
    assert row["evals_to_top_percent"] == 3  # 71.0 < 71.18; 72.6 clears it
    assert row["regret_after_init"] == optimum - 71.0


def test_aggregate_median():
    mod = load_module()
    rows = [
        mod.summarize_trajectory([70.0, 72.0], 73.0, num_init=1),
        mod.summarize_trajectory([71.0, 73.0], 73.0, num_init=1),
    ]
    for i, row in enumerate(rows):
        row["match_id"] = f"m{i}"
    agg = mod.aggregate_search_rows(rows)
    assert agg["n_trajectories"] == 2
    assert agg["best_final_regret"] == 0.0
    assert agg["frac_within_1pct_opt"] == 0.5
