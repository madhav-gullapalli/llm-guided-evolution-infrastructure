"""Feature-specific profile, component isolation, and evaluator-policy checks."""
import ast
import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.cfg import surrogate_coev as profile

REPO = Path(__file__).resolve().parents[1]


def test_metadata_inherits_existing_constants():
    from src.cfg import constants_surrogate as existing
    for name, value in vars(existing).items():
        if not name.startswith("_"):
            assert getattr(profile, name) is value
    assert profile.SEED_NETWORK == existing.SEED_NETWORK
    assert not hasattr(profile, "runtime_values")
    assert not hasattr(profile, "slurm_config")


def test_existing_controller_dispatches_population_metadata(monkeypatch):
    tree = ast.parse((REPO / "islands_wrapper.py").read_text())
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "globals().get('MODE') == 3")
    calls = []
    namespace = {"MODE": 3, "args": SimpleNamespace(checkpoints="existing_run"),
                 "spawn_coevolution_islands": calls.append}
    with pytest.raises(SystemExit) as result:
        exec(compile(ast.Module(body=[guard], type_ignores=[]), "islands_wrapper.py", "exec"), namespace)
    assert result.value.code == 0
    assert calls == ["existing_run"]
    calls.clear()
    namespace["MODE"] = 2
    exec(compile(ast.Module(body=[guard], type_ignores=[]), "islands_wrapper.py", "exec"), namespace)
    assert not calls


@pytest.mark.parametrize("generations,era_length,expected", [(5, 1, 5), (6, 2, 3), (5, 0, 1)])
def test_controller_retains_existing_era_budget(monkeypatch, tmp_path, generations, era_length, expected):
    import islands_wrapper as controller
    events, scripts = [], []
    monkeypatch.setattr(controller, "num_generations", generations)
    monkeypatch.setattr(controller, "migration_gen", era_length)

    def submit(path, text):
        job_id = str(len(scripts) + 1)
        events.append(("submit", job_id))
        scripts.append(text)
        assert path == "src/island_temp_script.sh"
        return job_id

    def wait(job_id):
        events.append(("wait", job_id))
        return True

    monkeypatch.setattr(controller, "submit_run", submit)
    monkeypatch.setattr(controller, "check4job_completion", wait)
    controller.spawn_coevolution_islands(str(tmp_path / "run with spaces"))
    assert len(scripts) == expected
    assert events == [(action, str(i + 1)) for i in range(expected) for action in ("submit", "wait")]
    for script in scripts:
        assert "run_coevolution.py --checkpoints" in script
        assert "run_improved.py" not in script
        before, after = controller.ISLANDS_BASH_SCRIPT_TEMPLATE.split("uv run python run_improved.py", 1)
        assert script.startswith(before)
        assert script.count("#SBATCH") == controller.ISLANDS_BASH_SCRIPT_TEMPLATE.count("#SBATCH")
        assert "--global_path" not in script
        result = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("job_id", [None, "failed_job"])
def test_failed_island_stops_controller(monkeypatch, tmp_path, job_id):
    import islands_wrapper as controller
    submissions = []
    monkeypatch.setattr(controller, "num_generations", 5)
    monkeypatch.setattr(controller, "migration_gen", 1)
    monkeypatch.setattr(controller, "submit_run", lambda *args: submissions.append(args) or job_id)
    monkeypatch.setattr(controller, "check4job_completion", lambda job: False)
    with pytest.raises(RuntimeError, match="did not complete"):
        controller.spawn_coevolution_islands(str(tmp_path))
    assert len(submissions) == 1


def test_coevolution_seeds_have_only_their_component():
    surrogate = (REPO / profile.POPULATIONS["surrogate"]["seed_file"]).read_text()
    usage = (REPO / profile.POPULATIONS["usage"]["seed_file"]).read_text()
    surrogate_names = {getattr(node, "name", "") for node in ast.parse(surrogate).body}
    usage_names = {getattr(node, "name", "") for node in ast.parse(usage).body}
    assert "SurrogateUsageOptimizer" not in surrogate_names
    assert {"get_surrogate_config", "build_predictor_kwargs", "CustomXGBoost"} <= surrogate_names
    assert "SurrogateUsageOptimizer" in usage_names
    assert not {"CustomXGBoost", "get_surrogate_config", "build_predictor_kwargs"} & usage_names
    # All feature implementations belong to the editable block, not frozen headers.
    assert "def get_surrogate_config" in surrogate.split("# --OPTION--")[1]
    assert "class SurrogateUsageOptimizer" in usage.split("# --OPTION--")[1]


def evaluator_policy(coevolution):
    """Execute the actual policy boundary from eval.py with cheap NASLib stubs."""
    tree = ast.parse((REPO / "sota/Surrogate/eval.py").read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == "run_experiment")
    branch = next(node for node in method.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "optimizer_name == 'rea'")
    calls = []

    class Usage:
        def __init__(self, *args, **kwargs):
            pass

        def new_epoch(self, epoch):
            calls.append(epoch)

    original_method = Usage.new_epoch
    banana = SimpleNamespace(new_epoch="original", _get_best_candidates="original")
    namespace = {
        "_coevolution": profile if coevolution else None,
        "optimizer_name": "bananas", "_model_module": SimpleNamespace(SurrogateUsageOptimizer=Usage),
        "config": object(), "predictor_cls": None, "predictor_kwargs": {},
        "debug_new_epoch": lambda self, epoch: calls.append("debug"),
        "types": __import__("types"), "bananas_opt": SimpleNamespace(Bananas=banana),
        "acq_funcs": SimpleNamespace(), "batch_acquisition_function": object(),
        "_get_best_candidates_batch": object(),
    }
    # Factory hook is unnecessary in this boundary-only test.
    observer_module = SimpleNamespace(observe_usage_optimizer=lambda opt: setattr(opt, "observed", True))
    previous = sys.modules.get("coev_evaluation")
    sys.modules["coev_evaluation"] = observer_module
    try:
        guards = [node for node in tree.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "not _coevolution"]
        for node in guards:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "eval.py", "exec"), namespace)
        exec(compile(ast.Module(body=[branch], type_ignores=[]), "eval.py", "exec"), namespace)
    finally:
        if previous is None:
            sys.modules.pop("coev_evaluation", None)
        else:
            sys.modules["coev_evaluation"] = previous
    optimizer = namespace["optimizer"]
    optimizer.new_epoch(7)
    if coevolution:
        assert optimizer.new_epoch.__func__ is original_method
        assert optimizer.observed
        assert calls == [7]
        assert banana.new_epoch == banana._get_best_candidates == "original"
    else:
        assert calls == ["debug"]
        assert banana.new_epoch is namespace["debug_new_epoch"]
        assert banana._get_best_candidates is namespace["_get_best_candidates_batch"]


@pytest.mark.parametrize("coevolution", [False, True])
def test_only_coevolution_preserves_usage_policy(coevolution):
    evaluator_policy(coevolution)


def test_fit_observer_keeps_usage_and_raw_metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "sota/Surrogate"))
    from coev_evaluation import observe_usage_optimizer

    class Ensemble:
        def fit(self, *args):
            return "fit result"

        def query(self, test):
            return np.array([[1., 2., 3.], [1., 2., 3.]])

    class Usage:
        def __init__(self):
            self.test_data = [1, 2, 3]
            self.test_accuracies = [1., 2., 3.]
            self.surrogate_test_metrics = []
            self.surrogate_test_metrics_mse = []
            self.calls = []

        def _get_ensemble(self):
            return Ensemble()

        def new_epoch(self, epoch):
            self.calls.append(epoch)
            self.ensemble = self._get_ensemble()
            assert self.ensemble.fit([], []) == "fit result"

    optimizer = Usage()
    before = optimizer.new_epoch.__func__
    observe_usage_optimizer(optimizer)
    optimizer.new_epoch(50)
    assert optimizer.new_epoch.__func__ is before
    assert optimizer.calls == [50]
    assert optimizer.surrogate_test_metrics == [1.0]
    assert optimizer.surrogate_test_metrics_mse == [0.0]


def test_constants_symlink_and_solo_modes_import_without_missing_names():
    script = """
import importlib, os
from src.cfg import constants, constants_surrogate
import run_improved
for name in ['SEED_NETWORK', 'DNA_TXT', 'LLM_LLAMA3', 'FITNESS_WEIGHTS', 'ISLANDS_BASH_SCRIPT_TEMPLATE']:
    assert hasattr(constants, name), name
for mode, seed in [('SURROGATE', 'model.py'), ('USE', 'usage_bananas.py')]:
    os.environ['LLMGE_EVOLUTION_MODE'] = mode
    importlib.reload(constants_surrogate)
    importlib.reload(constants)
    assert constants.SEED_NETWORK.endswith('/' + seed)
    assert constants.SEED_NETWORK == constants_surrogate.SEED_NETWORK
    assert constants.VARIANT_DIR.endswith('/models/llmge_models')
    assert constants.HOSTNAME_DIR == constants_surrogate.HOSTNAME_DIR
"""
    env = dict(os.environ)
    env.pop("LLMGE_USAGE_SEED", None)
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_numeric_mode_selects_coevolution_only_for_both(monkeypatch, mode):
    monkeypatch.delenv("LLMGE_EVOLUTION_MODE", raising=False)
    path = REPO / "src/cfg/constants_surrogate.py"
    source = path.read_text().replace("MODE = 3", f"MODE = {mode}", 1)
    namespace = {"__file__": str(path), "__name__": "mode_test"}
    exec(compile(source, str(path), "exec"), namespace)
    assert namespace["MODE"] == mode
    assert int(namespace["evolution_mode"]) == mode
    if mode == 1:
        assert namespace["SEED_NETWORK"].endswith("/model.py")
    elif mode == 2:
        assert namespace["SEED_NETWORK"].endswith("/usage_bananas.py")
    calls = []
    namespace.update(args=SimpleNamespace(checkpoints="mode_run"), spawn_coevolution_islands=calls.append)
    tree = ast.parse((REPO / "islands_wrapper.py").read_text())
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "globals().get('MODE') == 3")
    code = compile(ast.Module(body=[guard], type_ignores=[]), "islands_wrapper.py", "exec")
    if mode == 3:
        with pytest.raises(SystemExit):
            exec(code, namespace)
        assert calls == ["mode_run"]
    else:
        exec(code, namespace)
        assert not calls
