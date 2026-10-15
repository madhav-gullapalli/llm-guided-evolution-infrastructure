"""Focused MVP acceptance tests; external LLM/Slurm/NAS work is stubbed."""
import copy
import importlib.util
import random
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from deap import base

from src.coevolution import (PairRecord, barrier_passes, random_uniform_coverage,
                             rank_pairs, rank_statistics)
from src.source_composition import compose_sources

REPO = Path(__file__).resolve().parents[1]


class ObjectiveFitness(base.Fitness):
    weights = (1.0, -1.0)


INVALID = (-float("inf"), float("inf"))


@pytest.mark.parametrize("a,b,kappa,matches", [(12, 12, 2, 36), (4, 7, 3, 22),
                                              (7, 4, 3, 28), (1, 3, 1, 3),
                                              (7, 5, 2, 40), (5, 5, 2, 40), (3, 4, 1, 13)])
def test_pairing(a, b, kappa, matches):
    pairs = random_uniform_coverage(range(a), range(b), matches, kappa, random.Random(242))
    assert len(pairs) == len(set(pairs)) == min(matches, a * b)
    assert min(Counter(x for x, _ in pairs).values()) >= kappa
    assert min(Counter(y for _, y in pairs).values()) >= kappa
    assert set(x for x, _ in pairs) == set(range(a))
    assert set(y for _, y in pairs) == set(range(b))


@pytest.mark.parametrize("a,b,kappa,matches", [(0, 3, 1, 3), (3, 4, 0, 4),
                                              (3, 4, 4, 12), (3, 4, 2, 7)])
def test_impossible_pairing(a, b, kappa, matches):
    with pytest.raises(ValueError, match="Infeasible"):
        random_uniform_coverage(range(a), range(b), matches, kappa)


def test_deap_fronts_and_best_valid_projection():
    records = [PairRecord(str(i), {"surrogate": s, "usage": u}, values,
                          ObjectiveFitness, INVALID)
               for i, (s, u, values) in enumerate([
                   ("s0", "u0", (0.9, 2.0)), ("s1", "u1", (0.8, 1.0)),
                   ("s0", "u2", (0.7, 3.0)), ("s2", "u0", INVALID),
               ])]
    ranks = rank_pairs(records, {"surrogate": ["s0", "s1", "s2"],
                                 "usage": ["u0", "u1", "u2"]})
    assert [r.pareto_level for r in records] == [0, 0, 1, 2]
    assert ranks == {"surrogate": {"s0": 0, "s1": 0, "s2": None},
                     "usage": {"u0": 0, "u1": 0, "u2": 1}}
    assert records[0].fitness.values == (0.9, 2.0)
    assert rank_statistics(ranks, records)["populations"]["surrogate"]["rank_0"] == 2
    for record in records:
        record.fitness.values = INVALID
        record.success = False
    assert all(rank is None for pool in rank_pairs(records, {
        "surrogate": ["s0", "s1", "s2"], "usage": ["u0", "u1", "u2"]}).values()
               for rank in pool.values())


@pytest.mark.parametrize("surrogates,usage,expected", [(4, 10, False), (5, 5, False),
                                                     (6, 6, True), (10, 5, False)])
def test_independent_minimum(surrogates, usage, expected):
    ranks = {"surrogate": dict.fromkeys(range(surrogates), 0),
             "usage": dict.fromkeys(range(usage), 0)}
    assert barrier_passes(ranks, {name: {"num_elites": 5} for name in ranks}) is expected


def import_file(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_generic_composition_namespaces(tmp_path):
    sources = []
    for index in range(3):
        path = tmp_path / f"source_{index}.py"
        path.write_text(f"VALUE = {index}\ndef value():\n    return VALUE\n")
        sources.append(path)
    output = compose_sources(sources, {"first": 0, "second": 1, "third": 2}, tmp_path / "bad.py")
    # Exports are symbol names, not aliases: each source must provide the named symbol.
    with pytest.raises(AttributeError):
        import_file(output)
    for index, path in enumerate(sources):
        path.write_text(path.read_text() + f"{['first', 'second', 'third'][index]} = value\n")
    module = import_file(compose_sources(sources, {"first": 0, "second": 1, "third": 2},
                                         tmp_path / "model_generic.py"))
    assert [module.first(), module.second(), module.third()] == [0, 1, 2]


def test_existing_seed_composition(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "sota/Surrogate"))
    module = import_file(compose_sources(
        [REPO / "sota/Surrogate/model.py", REPO / "sota/Surrogate/usage_bananas.py"],
        {"get_surrogate_config": 0, "build_predictor_kwargs": 0, "SurrogateUsageOptimizer": 1},
        tmp_path / "model_seed_pair.py"))
    cfg = module.get_surrogate_config(corpus_path="test.csv")
    kwargs = module.build_predictor_kwargs(cfg)
    assert kwargs["corpus_path"] == "test.csv"
    assert kwargs["base_predictor_cls"].__module__.endswith("_source_0")
    assert module.SurrogateUsageOptimizer.__module__.endswith("_source_1")


@pytest.fixture
def local_jobs(monkeypatch, tmp_path):
    import run_coevolution as coev
    legacy = coev.legacy
    # Keep acceptance tests independent of edits to the live experiment metadata.
    settings = copy.deepcopy(coev.config.POPULATIONS)
    for population in settings.values():
        population.update(start_population_size=12, population_size=12, num_elites=5)
    monkeypatch.setattr(coev.config, "POPULATIONS", settings)
    monkeypatch.setattr(coev.config, "matches", 36)
    monkeypatch.setattr(coev.config, "kappa", 2)
    monkeypatch.syspath_prepend(str(REPO / "sota/Surrogate"))
    monkeypatch.setattr(legacy, "LOCAL", True)
    monkeypatch.setattr(legacy, "DELAYED_CHECK", False)
    monkeypatch.setattr(legacy, "PROB_EOT", 0)
    monkeypatch.setattr(legacy, "SOTA_ROOT", str(tmp_path / "sota"))
    results = Path(legacy.SOTA_ROOT, "results")
    results.mkdir(parents=True)
    llm_calls, evaluation_calls = [], []

    def fake_llm(file_path, **kwargs):
        # Exercise the existing operators/script construction; stub only remote work.
        legacy.create_bash_file(file_path, **kwargs)
        output = Path(kwargs["output_filename"])
        output.write_text(Path(kwargs["input_filename_x"]).read_text()
                          + f"\n# stub variant {output.stem}\n")
        llm_calls.append(kwargs)
        return True, None, "Job done"

    def fake_job(command, **kwargs):
        script = Path(command[1]).read_text()
        line = next(line for line in script.splitlines() if "--variant_dir" in line)
        args = shlex.split(line)
        module_name = args[args.index("--model") + 1]
        variants = Path(args[args.index("--variant_dir") + 1])
        assert args[args.index("python") + 1] == legacy.TRAIN_FILE
        assert module_name.startswith("model_coev_")
        module = import_file(variants / f"{module_name}.py")
        assert module.get_surrogate_config.__module__.endswith("_source_0")
        assert module.SurrogateUsageOptimizer.__module__.endswith("_source_1")
        evaluation_calls.append(module_name)
        assert "--coevolution" in args
        (results / f"{module_name.removeprefix('model_')}_results.txt").write_text("0.9,1.0")
        return SimpleNamespace(returncode=0, stdout="Job done", stderr="")

    monkeypatch.setattr(legacy, "submit_bash", fake_llm)
    monkeypatch.setattr(legacy.subprocess, "run", fake_job)
    return coev, llm_calls, evaluation_calls


def test_joint_smoke_selection_checkpoint_resume(local_jobs, tmp_path, monkeypatch):
    coev, llm_calls, evaluation_calls = local_jobs
    settings = copy.deepcopy(coev.config.POPULATIONS)
    for cfg in settings.values():
        cfg.update(start_population_size=3, population_size=3, num_elites=2,
                   crossover_probability=1.0, mutation_probability=1.0)
    monkeypatch.setattr(coev.config, "POPULATIONS", settings)
    monkeypatch.setattr(coev.config, "matches", 9)
    monkeypatch.setattr(coev.config, "kappa", 1)
    monkeypatch.setattr(coev.legacy, "migration_gen", 1)
    original_globals = (coev.legacy.VARIANT_DIR, coev.legacy.GLOBAL_DATA)
    root = tmp_path / "run"
    coev.run(root)
    checkpoint, start, _ = coev.legacy.load_checkpoint(str(root), global_path=str(root))
    assert start == 2
    assert len(evaluation_calls) == 9
    assert len(llm_calls) == 6
    assert all(len(w["elites"]) == 2 for w in checkpoint["workers"].values())
    assert all(not ind.fitness.valid for w in checkpoint["workers"].values()
               for ind in w["population"] + w["parents"] + w["elites"])
    # Resume one shared generation and exercise real crossover/mutation primitives.
    coev.run(root)
    assert (root / "checkpoint_gen_2.pkl").exists()
    assert len(evaluation_calls) == len(set(evaluation_calls)) == 18
    assert sum(call["python_file"] == "src/llm_crossover.py" for call in llm_calls) == 4
    assert sum(call["python_file"] == "src/llm_mutation.py" for call in llm_calls) == 12
    assert coev.legacy.VARIANT_DIR == original_globals[0]
    assert coev.legacy.GLOBAL_DATA is original_globals[1]
    for call in llm_calls:
        source = Path(call["input_filename_x"])
        destination = Path(call["output_filename"])
        if source.name not in ("coev_surrogate.py", "coev_usage.py"):
            assert source.parent == destination.parent
        if call.get("input_filename_y"):
            assert Path(call["input_filename_y"]).parent == destination.parent


def test_cumulative_successes_survive_retry_and_restart(local_jobs, tmp_path, monkeypatch):
    coev, _, evaluation_calls = local_jobs
    settings = copy.deepcopy(coev.config.POPULATIONS)
    for cfg in settings.values():
        cfg.update(start_population_size=10, population_size=10, num_elites=5)
    monkeypatch.setattr(coev.config, "POPULATIONS", settings)
    monkeypatch.setattr(coev.config, "matches", 100)
    monkeypatch.setattr(coev.config, "kappa", 1)
    monkeypatch.setattr(coev.legacy, "migration_gen", 1)
    observed = []
    original_candidates = coev.PopulationWorker.candidates
    retained = {}
    interrupted = False

    def candidates(worker, generation, regenerate=False):
        nonlocal interrupted
        if regenerate and not interrupted:
            interrupted = True
            raise RuntimeError("simulated worker restart")
        pool = original_candidates(worker, generation, regenerate)
        if regenerate:
            assert retained[worker.name] <= {ind[0] for ind in pool}
            assert len(pool) == 10
        return pool

    def evaluate(workers, pairs, generation, attempt, root):
        observed.append((generation, attempt, [len(w.population) for w in workers.values()]))
        if attempt == 1:
            # Exactly five successes is insufficient; preserve them on disk.
            passing = {ind[0] for ind in workers["surrogate"].population[:5]}
            retained.update(surrogate=passing,
                            usage={ind[0] for ind in workers["usage"].population})
            return [PairRecord(str(i), {"surrogate": s, "usage": u},
                               (0.9, 1.0) if s in passing else INVALID,
                               ObjectiveFitness, INVALID)
                    for i, (s, u) in enumerate(pairs)]
        assert not (Path(root) / "checkpoint_gen_1.pkl").exists()
        # Only one NEW surrogate succeeds in this attempt. Earlier successes
        # fail now, but still count cumulatively toward the strict >5 barrier.
        new_gene = next(ind[0] for ind in workers["surrogate"].population
                        if ind[0] not in retained["surrogate"])
        return [PairRecord(f"retry_{i}", {"surrogate": s, "usage": u},
                           (0.8, 1.0) if s == new_gene else INVALID,
                           ObjectiveFitness, INVALID)
                for i, (s, u) in enumerate(pairs)]

    monkeypatch.setattr(coev, "evaluate_matches", evaluate)
    monkeypatch.setattr(coev.PopulationWorker, "candidates", candidates)
    root = tmp_path / "restart"
    with pytest.raises(RuntimeError, match="simulated worker restart"):
        coev.run(root)
    assert (root / "pending_attempt.pkl").exists()
    assert not (root / "checkpoint_gen_1.pkl").exists()
    coev.run(root)
    assert observed == [(1, 1, [10, 10]), (1, 2, [10, 10])]
    checkpoint, _, _ = coev.legacy.load_checkpoint(str(root), global_path=str(root))
    assert len(checkpoint["workers"]["surrogate"]["successful"]) == 6
    assert len(checkpoint["workers"]["usage"]["successful"]) == 10
    assert not (root / "pending_attempt.pkl").exists()
    assert not evaluation_calls


def test_impossible_actual_pool_submits_no_evaluations(local_jobs, tmp_path, monkeypatch):
    coev, _, evaluation_calls = local_jobs
    settings = copy.deepcopy(coev.config.POPULATIONS)
    for cfg in settings.values():
        cfg.update(start_population_size=2, num_elites=2)
    monkeypatch.setattr(coev.config, "POPULATIONS", settings)
    monkeypatch.setattr(coev.config, "matches", 3)
    # A cap of 3 cannot cover both 2-member populations twice.
    with pytest.raises(ValueError, match="Infeasible"):
        coev.run(tmp_path / "infeasible")
    assert not evaluation_calls


def test_legacy_startup_keeps_original_cli():
    # Separate interpreter verifies the original entrypoint never loads new config.
    script = """
import runpy, sys
sys.argv = ['run_improved.py', '--help']
try:
    runpy.run_path('run_improved.py', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
assert 'src.cfg.coevolution' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--llm_model" in result.stdout and "--prompt_group" in result.stdout


@pytest.mark.parametrize("extra_population", [True, False])
def test_unsupported_workers_rejected(tmp_path, monkeypatch, extra_population):
    import run_coevolution as coev
    settings = copy.deepcopy(coev.config.POPULATIONS)
    if extra_population:
        settings["third"] = copy.deepcopy(settings["usage"])
    else:
        settings["usage"]["num_islands"] = 2
    monkeypatch.setattr(coev.config, "POPULATIONS", settings)
    with pytest.raises(ValueError, match="MVP"):
        coev.run(tmp_path / "unsupported")
    assert not (tmp_path / "unsupported").exists()


@pytest.mark.parametrize("usage_count", [4, 5])
def test_deduplication_regenerates_before_pairing(local_jobs, tmp_path, monkeypatch, usage_count):
    coev, _, evaluations = local_jobs
    settings = copy.deepcopy(coev.config.POPULATIONS)
    monkeypatch.setattr(coev.config, "POPULATIONS", settings)
    original = coev.PopulationWorker.candidates
    calls = []

    def candidates(worker, generation, regenerate=False):
        calls.append((worker.name, generation, regenerate))
        pool = original(worker, generation, regenerate)
        if not regenerate:
            # Small pools can contribute successes before the retry fills slots.
            worker.population = pool[:5 if worker.name == "surrogate" else usage_count]
        return worker.population

    monkeypatch.setattr(coev.PopulationWorker, "candidates", candidates)
    coev.run(tmp_path / "deduplicated")
    expected = [("surrogate", 1, False), ("usage", 1, False)]
    expected += [("surrogate", 1, True), ("usage", 1, True)]
    assert calls == expected
    assert len(evaluations) == 5 * usage_count + 36
    assert (tmp_path / "deduplicated/checkpoint_gen_1.pkl").exists()
    import json
    reports = [json.loads(path.read_text()) for path in
               (tmp_path / "deduplicated/history").glob("*.json")]
    assert all(report["matches_cap"] == 36 for report in reports)
    assert sum(report["attempted_matches"] for report in reports) == len(evaluations)


def test_regeneration_keeps_one_island_and_generation(local_jobs, tmp_path, monkeypatch, capsys):
    coev, _, evaluations = local_jobs
    import islands_wrapper as controller
    monkeypatch.setattr(controller, "num_generations", 1)
    monkeypatch.setattr(controller, "migration_gen", 1)
    original_candidates = coev.PopulationWorker.candidates
    original_evaluate = coev.evaluate_matches
    candidate_calls, submissions, waits, attempts = [], [], [], []

    def candidates(worker, generation, regenerate=False):
        candidate_calls.append((worker.name, generation, regenerate))
        original_candidates(worker, generation, regenerate)
        if not regenerate:
            worker.population = worker.population[:5 if worker.name == "surrogate" else 4]
        return worker.population

    def evaluate(workers, pairs, generation, attempt, root):
        attempts.append((generation, attempt))
        if attempt == 2:
            # A whole failed evaluation attempt must regenerate inside this worker too.
            return [PairRecord(f"failed_{i}", {"surrogate": s, "usage": u},
                               INVALID, ObjectiveFitness, INVALID)
                    for i, (s, u) in enumerate(pairs)]
        return original_evaluate(workers, pairs, generation, attempt, root)

    def submit(path, script):
        submissions.append((path, script))
        # Simulate the child job without contacting Slurm.
        coev.run(tmp_path / "one_island")
        return "same_island_job"

    def wait(job_id):
        waits.append(job_id)
        return True

    monkeypatch.setattr(coev.PopulationWorker, "candidates", candidates)
    monkeypatch.setattr(coev, "evaluate_matches", evaluate)
    monkeypatch.setattr(controller, "submit_run", submit)
    monkeypatch.setattr(controller, "check4job_completion", wait)
    controller.spawn_coevolution_islands(str(tmp_path / "one_island"))
    assert len(submissions) == 1
    assert waits == ["same_island_job"]
    assert attempts == [(1, 1), (1, 2), (1, 3)]
    assert candidate_calls == [("surrogate", 1, False), ("usage", 1, False),
                               ("surrogate", 1, True), ("usage", 1, True),
                               ("surrogate", 1, True), ("usage", 1, True)]
    assert len(evaluations) == 20 + 36
    assert (tmp_path / "one_island/checkpoint_gen_1.pkl").exists()
    assert capsys.readouterr().out.count("-- End of Era --") == 1


def test_source_deduplication_persists_and_preserves_carried_ids(local_jobs, tmp_path):
    coev, _, _ = local_jobs
    settings = coev.config.POPULATIONS['surrogate']
    worker = coev.PopulationWorker('surrogate', settings, tmp_path)

    def individual(owner, gene, code):
        owner.source(gene).write_text(code)
        return coev.legacy.creator.Individual([gene])

    original = individual(worker, 'original', 'VALUE = 1\n')
    failed = individual(worker, 'failed', 'VALUE = 2\n')
    assert worker.remove_historical_duplicates([original, failed], set()) == [original, failed]
    snapshot = worker.snapshot()
    # A later attempt rejects both previously successful and failed code under
    # fresh IDs, while keeping the actual selected source.
    copy = individual(worker, 'copy', 'VALUE = 1\n')
    failed_copy = individual(worker, 'failed_copy', 'VALUE = 2\n')
    fresh = individual(worker, 'fresh', 'VALUE = 3\n')
    assert worker.remove_historical_duplicates([copy, original, failed_copy, fresh], {'original'}) == [original, fresh]

    # Restore an older checkpoint without forgetting newer attempts.
    resumed = coev.PopulationWorker('surrogate', settings, tmp_path)
    resumed.restore(snapshot)
    fresh_copy = individual(resumed, 'fresh_copy', 'VALUE = 3\n')
    assert resumed.remove_historical_duplicates([fresh_copy, original], {'original'}) == [original]

    # Registries belong to populations, not the shared match directory.
    usage = coev.PopulationWorker('usage', coev.config.POPULATIONS['usage'], tmp_path)
    other = individual(usage, 'other', 'VALUE = 1\n')
    assert usage.remove_historical_duplicates([other], set()) == [other]


def test_source_deduplication_bootstraps_old_runs(local_jobs, tmp_path):
    coev, _, _ = local_jobs
    variants = tmp_path / 'surrogate' / 'variants'
    variants.mkdir(parents=True)
    (variants / f'{coev.legacy.MODEL}_archived.py').write_text('VALUE = 1\n')
    worker = coev.PopulationWorker('surrogate', coev.config.POPULATIONS['surrogate'], tmp_path)
    # Old checkpoints have no hash field.
    worker.restore({'successful': []})
    worker.source('regenerated').write_text('VALUE = 1\n')
    regenerated = coev.legacy.creator.Individual(['regenerated'])
    assert worker.remove_historical_duplicates([regenerated], set()) == []


def test_resume_allows_only_lowering_elite_minimum(local_jobs, monkeypatch):
    coev, _, _ = local_jobs
    saved = coev.configuration_snapshot()
    settings = copy.deepcopy(coev.config.POPULATIONS)
    for cfg in settings.values():
        cfg['num_elites'] = 1
    monkeypatch.setattr(coev.config, 'POPULATIONS', settings)
    assert coev.configuration_compatible(saved)
    assert barrier_passes({'surrogate': {'s0': 0, 's1': 1},
                          'usage': {'u0': 0, 'u1': 1}}, settings)
    assert not barrier_passes({'surrogate': {'s0': 0},
                              'usage': {'u0': 0, 'u1': 1}}, settings)
    incompatible = copy.deepcopy(saved)
    incompatible['populations']['surrogate']['seed_file'] = 'different.py'
    assert not coev.configuration_compatible(incompatible)
    settings['usage']['num_elites'] = 6
    assert not coev.configuration_compatible(saved)


@pytest.mark.parametrize('component,group,required', [
    ('surrogate', 'general', ['early_stopping_rounds', 'GridSearchCV', 'DEFAULT_SURROGATE_CONFIG', 'corpus_path']),
    ('usage', 'ranking', ['axis=0', '(E,N)', 'scalar integer indices', 'import random', 'predictor_cls']),
])
def test_component_prompts_include_contract_and_fixed_header(local_jobs, monkeypatch, component, group, required):
    coev, _, _ = local_jobs
    legacy = coev.legacy
    seed = REPO / f'sota/Surrogate/coev_{component}.py'
    monkeypatch.setattr(legacy, 'SEED_NETWORK', str(seed))
    monkeypatch.setattr(legacy, 'PROMPT_GLOB', f'templates/naslib/{group}/*.txt')
    editable = seed.read_text().split('# --OPTION--', 1)[1]
    # Check all three normal prompts, including literal dictionary braces in
    # the fixed surrogate header after the mutation runner formats the prompt.
    for path in sorted((REPO / f'templates/naslib/{group}').glob('*.txt')):
        monkeypatch.setattr(legacy.np.random, 'choice', lambda paths, chosen=str(path): chosen)
        template, _ = legacy.generate_template(0, -1, None, legacy.SOTA_ROOT, str(seed), str(REPO), 'llama3')
        rendered = template.format(editable)
        assert editable in rendered
        assert seed.read_text().split('# --OPTION--', 1)[0] in rendered
        assert all(term in rendered for term in required)
        assert 'INSIDE the editable block' in rendered
