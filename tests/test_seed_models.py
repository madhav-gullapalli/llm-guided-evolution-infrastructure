"""Initialization samples each seed independently without external jobs."""
import copy
from pathlib import Path

import pytest


@pytest.mark.parametrize("from_directory", [False, True])
def test_each_individual_uses_selected_seed(monkeypatch, tmp_path, from_directory):
    import run_improved as runner

    seeds = [str(tmp_path / f"seed_{i}.py") for i in range(3)]
    monkeypatch.setattr(runner, "SEED_NETWORKS", seeds)
    for seed in seeds:
        Path(seed).write_text("# seed\n")
    monkeypatch.setattr(runner, "SEED_DIRECTORY", str(tmp_path) if from_directory else None)
    monkeypatch.setattr(runner, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(runner, "GLOBAL_DATA", {})
    monkeypatch.setattr(runner, "GLOBAL_DATA_ANCESTRY", {})
    monkeypatch.setattr(runner, "DELAYED_CHECK", True)
    monkeypatch.setattr(runner, "load_yaml", lambda: {})
    selected = iter([seeds[2], seeds[0], seeds[2], seeds[1]])
    choices = []

    def choose(options):
        assert options == seeds
        return next(selected)

    # Keep ID generation separate from the random seed choice.
    ids = iter(["a", "b", "c", "d"])
    monkeypatch.setattr(runner, "generate_random_string", lambda length: next(ids))
    monkeypatch.setattr(runner.random, "choice", choose)
    def submit(path, **kwargs):
        choices.append(kwargs)
        return True, None, None
    monkeypatch.setattr(runner, "submit_bash", submit)
    for _ in range(4):
        runner.create_individual(list, "llama3")
    expected = [seeds[2], seeds[0], seeds[2], seeds[1]]
    assert [call["input_filename_x"] for call in choices] == expected
    assert [call["seed_network"] for call in choices] == expected
    assert [data["seed_network"] for data in runner.GLOBAL_DATA.values()] == expected


def test_worker_scopes_seed_pool(monkeypatch, tmp_path):
    import run_coevolution as coev

    settings = copy.deepcopy(coev.config.POPULATIONS["usage"])
    seeds = ["sota/Surrogate/seed_models/usage/coev_usage.py", "sota/Surrogate/seed_models/usage/coev_usage_random.py"]
    settings.pop("seed_dir", None)
    settings["seed_files"] = seeds
    worker = coev.PopulationWorker("usage", settings, tmp_path)
    previous = coev.legacy.SEED_NETWORKS
    with worker.activate(0):
        assert coev.legacy.SEED_NETWORKS == [
            str(Path(coev.legacy.ROOT_DIR, seed).resolve()) for seed in seeds]
    assert coev.legacy.SEED_NETWORKS is previous
    settings["seed_files"] = []
    with pytest.raises(ValueError, match="non-empty"):
        coev.PopulationWorker("usage", settings, tmp_path)
    settings.pop("seed_files")
    worker = coev.PopulationWorker("usage", settings, tmp_path)
    assert worker.state["SEED_NETWORKS"] == [worker.state["SEED_NETWORK"]]


def test_relocated_default_seed_imports(monkeypatch):
    import importlib
    from src.cfg import constants_surrogate

    path = Path(constants_surrogate.SEED_NETWORK)
    assert path.parent.name == "combined"
    assert path.parent.parent.name == "seed_models"
    assert path.is_file()
    monkeypatch.syspath_prepend(str(path.parents[2]))
    module = importlib.import_module("sota.Surrogate.seed_models.combined.joint_seed")
    assert Path(module.__file__).resolve() == path.resolve()
    assert callable(module.get_surrogate_config)
    assert callable(module.SurrogateUsageOptimizer)


def test_slurm_helper_finds_relocated_seed(monkeypatch):
    import slurm

    root = Path(__file__).resolve().parents[1] / "sota/Surrogate"
    monkeypatch.setenv("LLMGE_SOTA_ROOT", str(root))
    monkeypatch.delenv("LLMGE_SEED_NETWORK", raising=False)
    assert Path(slurm.RuntimeConfig().SEED_NETWORK) == root / "seed_models/combined/joint_seed.py"


def test_directory_discovery_and_new_seed_files(tmp_path):
    from src.seed_models import discover_seed_models

    (tmp_path / '__init__.py').write_text('')
    (tmp_path / '_helper.py').write_text('')
    (tmp_path / 'notes.txt').write_text('')
    with pytest.raises(ValueError, match='no seed models'):
        discover_seed_models(tmp_path)
    for name in ['b.py', 'a.py']:
        (tmp_path / name).write_text('')
    assert [Path(path).name for path in discover_seed_models(tmp_path)] == ['a.py', 'b.py']
    (tmp_path / 'c.py').write_text('')
    assert len(discover_seed_models(tmp_path)) == 3
    with pytest.raises(ValueError, match='does not exist'):
        discover_seed_models(tmp_path / 'missing')


def test_worker_discovers_population_directory(tmp_path):
    import run_coevolution as coev
    from src.seed_models import discover_seed_models

    for name, settings in {**coev.config.POPULATIONS, 'paired': coev.config.PAIRED_POPULATION}.items():
        worker = coev.PopulationWorker(name, settings, tmp_path)
        assert worker.state['SEED_NETWORKS'] == discover_seed_models(worker.state['SEED_DIRECTORY'])
        assert all(Path(path).parent == Path(worker.state['SEED_DIRECTORY'])
                   for path in worker.state['SEED_NETWORKS'])
