"""Joint coevolution history summarizes pair objectives, not empty population columns."""
import importlib.util
from pathlib import Path


def load_summary():
    path = Path(__file__).resolve().parents[1] / "sota/Surrogate/analysis_tools/summarize_coevolution.py"
    spec = importlib.util.spec_from_file_location("summarize_coevolution_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_joint_history_reports_objectives_rescore_and_best_so_far(tmp_path):
    summary = load_summary()
    report = {
        "generation": 1,
        "attempt": 1,
        "attempted_matches": 2,
        "configuration": {"objectives": ["best_individual_found", "ktau"],
                          "paired_population": {"seed_file": "missing.py"}},
        "statistics": {"populations": {"paired": {
            "passing": 2, "rank_0": 1, "rank_counts": {"0": 1, "1": 1}}}},
        "matches": [
            {"success": True, "source_ids": {"paired": "kept"},
             "objectives": [70.0, 0.40], "pareto_level": 1},
            {"success": True, "source_ids": {"paired": "best"},
             "objectives": [73.0, 0.50], "pareto_level": 0},
            {"success": False, "source_ids": {"paired": "bad"},
             "objectives": [], "pareto_level": None},
        ],
    }
    later = {
        "generation": 2,
        "attempt": 1,
        "attempted_matches": 1,
        "configuration": report["configuration"],
        "statistics": {"populations": {"paired": {
            "passing": 1, "rank_0": 1, "rank_counts": {"0": 1}}}},
        "matches": [
            {"success": True, "source_ids": {"paired": "kept"},
             "objectives": [71.5, 0.40], "pareto_level": 0},
        ],
    }
    text = summary.render(tmp_path, [report, later])
    assert "Meeting scoreboard" in text
    assert "acc b/m/w" in text
    assert "tau b/m/w" in text
    assert "surr_pass" not in text
    assert "Best-so-far accuracy 73.00 first reached at generation 1" in text
    assert "1 genes evaluated in more than one generation, 1 changed." in text
    assert "acc 70.00->71.50" in text
    assert "gen 1: acc 73.00–73.00" in text
    assert "within-1%-of-opt" in text
    assert "Search from pair objectives" in text
