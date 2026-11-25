"""Hamming locality helpers for BANANAS-style query diversity."""
import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "sota/Surrogate/analysis_tools/query_diversity.py"
    spec = importlib.util.spec_from_file_location("query_diversity_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hamming():
    mod = load_module()
    assert mod.hamming([0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 4, 0]) == 0
    assert mod.hamming([0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 4, 1]) == 1
    assert mod.hamming([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]) == 6


def test_summarize_ops_local_steps():
    mod = load_module()
    ops = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],  # edit-1
        [0, 0, 0, 0, 1, 1],  # edit-1
        [1, 1, 1, 1, 1, 1],  # Hamming 4 vs previous (last two already 1)
    ]
    summary = mod.summarize_ops(ops, num_init=1)
    assert summary["n_queries"] == 4
    assert summary["n_unique"] == 4
    assert summary["step_hamming"][1:] == [1, 1, 4]
    assert abs(summary["frac_step_le_1"] - (2 / 3)) < 1e-9
    assert abs(summary["mean_guided_step_hamming"] - (1 + 4) / 2) < 1e-9
    # Pairwise includes the distant [0]*6 vs [1]*6 pair (Hamming 6), so mean > step mean.
    assert summary["mean_pairwise_hamming"] is not None
    assert summary["mean_pairwise_hamming"] > summary["mean_step_hamming"]


def test_summarize_with_accuracy_incumbent():
    mod = load_module()
    rows = [
        {"op_indices": [0, 0, 0, 0, 0, 0], "true_accuracy": 70.0, "selected": True},
        {"op_indices": [0, 0, 0, 0, 0, 1], "true_accuracy": 71.0, "selected": True},
        {"op_indices": [1, 1, 1, 1, 1, 1], "true_accuracy": 69.0, "selected": True},
        {"op_indices": [0, 0, 0, 0, 0, 2], "true_accuracy": 72.0, "selected": True},
    ]
    summary = mod.summarize_with_accuracy(rows, selected_only=True)
    # i=1: dist to ops[0]=1, then incumbent ← i=1
    # i=2: dist to ops[1]=5
    # i=3: dist to ops[1]=1, then incumbent ← i=3
    assert summary["dist_to_incumbent"] == [0, 1, 5, 1]
    assert summary["best_accuracy"] == 72.0
