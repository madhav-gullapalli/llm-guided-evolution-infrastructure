"""Summary output accepts paired rows without a parent-score comparison."""
import importlib.util
import sys
from pathlib import Path


def test_coevolution_summary_without_parent_score(tmp_path, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1] / 'sota/Surrogate/analysis_tools/summarize_run.py'
    spec = importlib.util.spec_from_file_location('summary_under_test', path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    (tmp_path / 'global_data').mkdir()
    monkeypatch.setattr(summary, 'coevolution_rows', lambda *_: [{
        'gene': 'pair_1', 'generation': 1, 'mode': 'COEVOLUTION',
        'status': 'ok', 'kendall_tau': 0.5,
    }])
    monkeypatch.setattr(sys, 'argv', ['summarize_run.py', '--run', str(tmp_path)])
    summary.main()
    output = capsys.readouterr().out
    assert '1 genes (1 ok, 0 failed)' in output
    assert 'same score as parent' not in output
    assert (tmp_path / 'analysis/summary.csv').is_file()
