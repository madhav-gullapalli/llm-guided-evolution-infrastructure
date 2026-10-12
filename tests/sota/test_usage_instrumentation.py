"""Verify instrumentation respects the evolved optimizer."""
import importlib.util
from pathlib import Path
import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "sota" / "Surrogate" / "usage_instrumentation.py"
spec = importlib.util.spec_from_file_location("usage_instrumentation", SOURCE)
instrumentation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(instrumentation)

class FakeEnsemble:
    def fit(self, *args, **kwargs):
        return "fit complete"

    def query(self, xs):
        return np.array([[3.0, 1.0, 2.0], [3.0, 1.0, 2.0]])

class EvolvedUsage:
    def __init__(self):
        self.calls = []
        self.test_data = ["a", "b", "c"]
        self.test_accuracies = [3.0, 1.0, 2.0]
        self.surrogate_test_metrics = []
        self.surrogate_test_metrics_mse = []

    def _get_ensemble(self):
        return FakeEnsemble()

    def new_epoch(self, epoch):
        self.calls.append(epoch)
        assert self._get_ensemble().fit([], []) == "fit complete"
        return "evolved"

def test_evolved_dispatch_and_metrics():
    opt = EvolvedUsage()
    fn = opt.new_epoch.__func__
    instrumentation.instrument_surrogate_usage(opt)
    instrumentation.instrument_surrogate_usage(opt)
    assert opt.new_epoch.__func__ is fn
    assert opt.new_epoch(7) == "evolved"
    assert opt.calls == [7]
    assert opt.surrogate_test_metrics == [1.0]
    assert opt.surrogate_test_metrics_mse == [0.0]

def test_no_test_data_keeps_evolution_working():
    opt = EvolvedUsage()
    opt.test_data = []
    instrumentation.instrument_surrogate_usage(opt)
    assert opt.new_epoch(1) == "evolved"
    assert opt.surrogate_test_metrics == []
