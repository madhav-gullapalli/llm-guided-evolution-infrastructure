"""Two isolated population contexts, synchronized at one joint evaluation barrier.

Run from the repository root. This entrypoint runs one existing migration era,
without migration; rerunning the same command resumes at the next shared era.
"""
import argparse
import copy
import hashlib
import json
import pickle
import random
import shlex
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from deap import tools
import run_improved as legacy
from src.cfg import surrogate_coev as config
from src.coevolution import (PairRecord, RankProxy, SEARCH_BUDGET, barrier_passes,
                             random_uniform_coverage, rank_pairs, rank_statistics, validate_pairing)
from src.source_composition import compose_sources
from src.seed_models import discover_seed_models
from src.experiment_metrics import validate_config, fitness_type, configured_objectives
from sota.Surrogate.surrogate_evaluation import SAMPLE_SEED


@contextmanager
def evolution_context(state):
    """Scope legacy globals to one worker. Workers execute sequentially.

    Actual LLM/evaluator jobs still use legacy batch submission. No global state
    from a population is visible to the other population or to legacy callers.
    """
    previous = {key: getattr(legacy, key) for key in state}
    random_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        for key, value in state.items():
            setattr(legacy, key, value)
        yield
    finally:
        for key in state:
            state[key] = getattr(legacy, key)
            setattr(legacy, key, previous[key])
        random.setstate(random_state)
        np.random.set_state(numpy_state)


class PopulationWorker:
    def __init__(self, name, settings, root):
        self.name, self.settings = name, settings
        self.root = Path(root).resolve() / name
        variant_dir = self.root / "variants"
        variant_dir.mkdir(parents=True, exist_ok=True)
        seed_directory = (str(Path(legacy.ROOT_DIR, settings["seed_dir"]).resolve())
                          if "seed_dir" in settings else None)
        seeds = (discover_seed_models(seed_directory) if seed_directory is not None else
                 settings.get("seed_files", [settings["seed_file"]]))
        if not isinstance(seeds, (list, tuple)) or not seeds:
            raise ValueError("seed_files must be a non-empty list of seed paths")
        seed_paths = [str(Path(legacy.ROOT_DIR, seed).resolve()) for seed in seeds]
        for seed in seed_paths:
            if not Path(seed).is_file():
                raise FileNotFoundError(f"Seed model does not exist: {seed}")
        self.state = {
            "SEED_NETWORKS": seed_paths,
            "SEED_DIRECTORY": seed_directory,
            "SEED_NETWORK": str(Path(legacy.ROOT_DIR, settings["seed_file"]).resolve()),
            "VARIANT_DIR": str(variant_dir), "OUTPUT_DIR": str(self.root / "jobs"),
            "PROMPT_GLOB": legacy.resolve_prompt_glob(settings["prompt_group"]),
            "GEN_COUNT": -1, "GENERATION": 0, "TOP_N_GENES": None,
            "LINKED_GENES": {}, "GLOBAL_DATA": {}, "GLOBAL_DATA_HIST": {},
            "GLOBAL_DATA_ANCESTRY": {legacy.MODEL: {"GENES": [legacy.MODEL],
                                                   "MUTATE_TYPE": ["CREATED"]}},
        }
        self.population, self.parents, self.elites = [], [], []
        self.successful = []
        self.individual_metrics = {}
        # Include archived variants when upgrading an existing run. Keep the
        # registry independent of generation checkpoints: even an infeasible
        # attempt must remember the code it generated.
        self.hash_registry = self.root / "seen_source_hashes.json"
        self.seen_source_hashes = set()
        if self.hash_registry.exists():
            self.seen_source_hashes.update(json.loads(self.hash_registry.read_text()))
        for path in variant_dir.glob(f"{legacy.MODEL}_*.py"):
            self.seen_source_hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
        # Preserve existing RNG conventions; keep each worker's state independent.
        self.random_state = random.getstate()
        self.numpy_state = np.random.get_state()

    @contextmanager
    def activate(self, generation):
        self.state["GENERATION"] = generation
        with evolution_context(self.state):
            random.setstate(self.random_state)
            np.random.set_state(self.numpy_state)
            try:
                yield
            finally:
                self.random_state = random.getstate()
                self.numpy_state = np.random.get_state()

    def source(self, gene_id):
        return Path(self.state["VARIANT_DIR"], f"{legacy.MODEL}_{gene_id}.py")

    def candidates(self, generation, regenerate=False):
        if config.SEPARATE_POPULATIONS and not getattr(config, self.name.upper() + "_EVOL"):
            if not self.population:
                gene = "fixed_" + self.name
                self.source(gene).write_bytes(Path(self.state["SEED_NETWORK"]).read_bytes())
                self.population = [legacy.creator.Individual([gene])]
            return self.population
        carried_ids = {ind[0] for ind in self.successful + self.parents + self.elites}
        with self.activate(generation):
            llm = self.settings["llm_model"]
            if regenerate or not self.parents:
                # Same seed-regeneration primitives as the baseline restart.
                legacy.GEN_COUNT, legacy.TOP_N_GENES = -1, None
                legacy.LINKED_GENES = {}
                legacy.GLOBAL_DATA_HIST.update(legacy.GLOBAL_DATA)
                legacy.GLOBAL_DATA = {}
                population = legacy.toolbox.population(
                    n=max(0, self.settings["start_population_size"] - len(self.successful)),
                    llm_model=llm, temp_min=0.7, temp_max=0.7, top_p=0.9)
                legacy.delayed_creation_check(population)
                for individual in self.successful:
                    gene_id = individual[0]
                    if gene_id in legacy.GLOBAL_DATA_HIST:
                        legacy.GLOBAL_DATA[gene_id] = legacy.GLOBAL_DATA_HIST[gene_id]
                population = copy.deepcopy(self.successful) + population
            else:
                legacy.GEN_COUNT = generation
                legacy.GLOBAL_DATA_HIST.update(copy.deepcopy(legacy.GLOBAL_DATA))
                population = list(map(legacy.toolbox.clone, self.parents))
                for i in range(0, len(population) - 1, 2):
                    if random.random() < self.settings["crossover_probability"]:
                        population[i], population[i + 1] = legacy.toolbox.mate(
                            population[i], population[i + 1], llm_model=llm)
                population = legacy.delayed_mate_check(population)
                for i, individual in enumerate(population):
                    if random.random() < self.settings["mutation_probability"]:
                        population[i] = legacy.toolbox.mutate(individual, llm_model=llm)
                population = legacy.delayed_mutate_check(population)
                population.extend(copy.deepcopy(self.elites))
                for elite in self.elites:
                    if elite[0] in legacy.GLOBAL_DATA_HIST:
                        legacy.GLOBAL_DATA[elite[0]] = legacy.GLOBAL_DATA_HIST[elite[0]]
            population = self.remove_historical_duplicates(population, carried_ids)
            self.population = legacy.remove_duplicates(population)
            # These individuals never carry raw pair fitness or durable rank fitness.
            for individual in self.population:
                del individual.fitness.values
            legacy.LINKED_GENES = {}
        return self.population

    def remove_historical_duplicates(self, population, carried_ids):
        """Reject recreated code, allowing explicitly carried source IDs."""
        unique = []
        removed = 0
        for individual in population:
            try:
                digest = hashlib.sha256(self.source(individual[0]).read_bytes()).hexdigest()
            except OSError:
                # Preserve the legacy handling of missing/failed LLM output.
                unique.append(individual)
                continue
            if digest in self.seen_source_hashes and individual[0] not in carried_ids:
                removed += 1
                continue
            self.seen_source_hashes.add(digest)
            unique.append(individual)
        temporary = self.hash_registry.with_suffix(".tmp")
        temporary.write_text(json.dumps(sorted(self.seen_source_hashes)) + "\n")
        temporary.replace(self.hash_registry)
        if removed:
            print(f"{self.name}: rejected {removed} previously generated source copies", flush=True)
        return unique

    def retain_successes(self, ranks):
        self.successful = [copy.deepcopy(ind) for ind in self.population
                           if ranks[ind[0]] is not None]

    def select(self, generation, ranks):
        with self.activate(generation):
            valid = [RankProxy(ind, ranks[ind[0]]) for ind in self.population
                     if ranks[ind[0]] is not None]
            elites = tools.selSPEA2(valid, self.settings["num_elites"])
            count = max(2, min(self.settings["population_size"], len(valid)))
            parents = legacy.toolbox.select(valid, count)
            top = tools.selSPEA2(valid, min(legacy.NUM_EOT_ELITES, len(valid)))
            by_id = {ind[0]: ind for ind in self.population}
            self.elites = [copy.deepcopy(by_id[ind[0]]) for ind in elites]
            self.parents = [copy.deepcopy(by_id[ind[0]]) for ind in parents]
            legacy.TOP_N_GENES = [copy.deepcopy(by_id[ind[0]]) for ind in top]
            legacy.GLOBAL_DATA_HIST.update(copy.deepcopy(legacy.GLOBAL_DATA))
            # Discard proxies now; only selected source IDs cross generations.

    def snapshot(self):
        return {"state": self.state, "population": self.population,
                "parents": self.parents, "elites": self.elites,
                "successful": self.successful,
                "individual_metrics": self.individual_metrics,
                "seen_source_hashes": self.seen_source_hashes.copy(),
                "random_state": self.random_state, "numpy_state": self.numpy_state}

    def restore(self, snapshot):
        for key, value in snapshot.items():
            if key == "seen_source_hashes":
                self.seen_source_hashes.update(value)
            else:
                setattr(self, key, value)


def save_pickle(path, data):
    """Replace a snapshot atomically so interrupted writes leave the old file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        pickle.dump(data, file)
    temporary.replace(path)


def save_intermediate(root, workers, generation, attempt, phase, match_state=None):
    """Archive phase state and export cumulative legacy-shaped analysis data.

    Phase archives are diagnostic, not resume checkpoints. Only completed
    barriers and pending_attempt.pkl determine where the coordinator resumes.
    """
    root = Path(root)
    snapshot = {"mode": "coevolution", "generation": generation,
                "attempt": attempt, "phase": phase,
                "configuration": configuration_snapshot(),
                "workers": {name: worker.snapshot() for name, worker in workers.items()},
                "match_state": match_state,
                "random_state": random.getstate(), "numpy_state": np.random.get_state()}
    save_pickle(root / "intermediate" / f"generation_{generation}" /
                f"attempt_{attempt}_{phase}_{uuid.uuid4().hex}.pkl", snapshot)
    path = root / "global_data" / f"global_gen_{generation}.pkl"
    keys = ("GLOBAL_DATA", "GLOBAL_DATA_HIST", "GLOBAL_DATA_ANCESTRY")
    if path.exists():
        with path.open("rb") as file:
            data = pickle.load(file)
    else:
        data = {key: {} for key in keys}
    for worker in workers.values():
        for key in keys:
            # Each worker has its own seed ancestry under the same MODEL ID;
            # preserve those separately in workers rather than flattening it.
            data[key].update({gene: info for gene, info in worker.state[key].items()
                              if gene != legacy.MODEL})
    if match_state is not None:
        for key in keys:
            data[key].update(match_state.get(key, {}))
    data.update(mode="coevolution", generation=generation, attempt=attempt,
                phase=phase, workers=snapshot["workers"])
    save_pickle(path, data)


def evaluate_matches(workers, pairings, generation, attempt, root):
    """Compose and submit through the unchanged legacy evaluation path."""
    pair_fitness = fitness_type("paired", config.PAIRED_METRICS)
    invalid = tuple(-float("inf") if w > 0 else float("inf") for w in pair_fitness.weights)
    placeholder = tuple(-9999999999 for _ in pair_fitness.weights)
    match_root = Path(root).resolve() / "matches"
    variants = match_root / "variants"
    variants.mkdir(parents=True, exist_ok=True)
    state = {"VARIANT_DIR": str(variants), "OUTPUT_DIR": str(match_root / "jobs"),
             "GENERATION": generation, "GLOBAL_DATA": {},
             "EVAL_RUNLINE": legacy.EVAL_RUNLINE + f" --coevolution --numEvols {config.numEvols}",
             "FITNESS_WEIGHTS": pair_fitness.weights,
             "INVALID_FITNESS_MAX": invalid,
             "PLACEHOLDER_FITNESS": placeholder}
    Path(state["OUTPUT_DIR"], str(generation)).mkdir(parents=True, exist_ok=True)
    batch_id = uuid.uuid4().hex
    individuals, source_ids = [], []
    for index, (surrogate, usage) in enumerate(pairings):
        match_id = f"coev_{batch_id}_g{generation}_a{attempt}_m{index}"
        ids = ({"surrogate": surrogate, "usage": usage} if config.SEPARATE_POPULATIONS
               else {"paired": surrogate})
        paths = [workers[name].source(ids[name]) for name in workers]
        owners = {name: index for index, name in enumerate(workers)}
        exports = {symbol: owners[owner] if config.SEPARATE_POPULATIONS else 0
                   for symbol, owner in config.COMPOSITION_EXPORTS.items()}
        compose_sources(paths, exports, variants / f"{legacy.MODEL}_{match_id}.py")
        individual = legacy.creator.Individual([match_id])
        individual.fitness = pair_fitness(state["PLACEHOLDER_FITNESS"])
        individuals.append(individual)
        source_ids.append(ids)
        # Missing/generated-invalid sources are evaluated by the existing path;
        # preserve the evaluator's existing fallback-to-seed behavior.
        state["GLOBAL_DATA"][match_id] = {
            "sub_flag": True, "status": "subbed file", "fitness": None,
            "job_id": None, "start_time": time.time(),
            "source_ids": ids,
        }
    save_intermediate(root, workers, generation, attempt, "matches_composed", state)
    try:
        with evolution_context(state):
            legacy.check_and_update_fitness(individuals)
    finally:
        # Preserve job IDs/statuses even if evaluation is interrupted or fails.
        save_intermediate(root, workers, generation, attempt, "evaluation", state)
    records = []
    for individual, ids in zip(individuals, source_ids):
        info = state["GLOBAL_DATA"].get(individual[0], {})
        values = individual.fitness.values
        # Only completed jobs with finite objectives can count at the barrier.
        if (info.get("status") != "completed"
                or len(values) != len(pair_fitness.weights)
                or not np.isfinite(values).all()):
            values = invalid
        records.append(PairRecord(individual[0], ids, values, pair_fitness, invalid))
    return records


def distinct_mode():
    return bool(config.SURROGATE_METRICS or config.USAGE_METRICS)


def evaluate_individuals(workers, generation, attempt, root, population="surrogate", metric=None, options=None):
    """Cache independent surrogate/oracle-usage metrics across partners and eras."""
    if metric is None:
        if not getattr(config, population.upper() + "_EVOL"):
            return workers[population].individual_metrics
        for name, arguments in getattr(config, population.upper() + "_METRICS").items():
            evaluate_individuals(workers, generation, attempt, root, population, name, arguments)
        return workers[population].individual_metrics
    options = dict(options or {})
    if population == "surrogate":
        options["require_tau"] = metric != "mse"
    if "X" in options:
        options["top_percent"] = options.pop("X")
    if metric == "ktau_top_X_percent":
        options.setdefault("top_percent", 10)
    worker = workers[population]
    match_root = Path(root).resolve() / ("individuals" if population == "surrogate" else "usage_individuals")
    variants = match_root / "variants"
    variants.mkdir(parents=True, exist_ok=True)
    state = {"VARIANT_DIR": str(variants), "OUTPUT_DIR": str(match_root / "jobs"),
             "GENERATION": generation, "GLOBAL_DATA": {},
             "EVAL_RUNLINE": legacy.EVAL_RUNLINE + f" --{population}_individual --numEvols {config.numEvols} --metric-options " + shlex.quote(json.dumps(options).replace("{", "{{").replace("}", "}}"))}
    Path(state["OUTPUT_DIR"], str(generation)).mkdir(parents=True, exist_ok=True)
    individuals, genes = [], []
    for candidate in worker.population:
        gene = candidate[0]
        if metric in worker.individual_metrics.get(gene, {}):
            continue
        evaluation_id = f"{population}_individual_{uuid.uuid4().hex}"
        (variants / f"{legacy.MODEL}_{evaluation_id}.py").write_bytes(worker.source(gene).read_bytes())
        individual = legacy.creator.Individual([evaluation_id])
        individual.fitness.values = legacy.PLACEHOLDER_FITNESS
        individuals.append(individual)
        genes.append(gene)
        state["GLOBAL_DATA"][evaluation_id] = {
            "sub_flag": True, "status": "subbed file", "fitness": None,
            "job_id": None, "start_time": time.time(), "source_ids": {population: gene},
        }
    save_intermediate(root, workers, generation, attempt, f"{population}_individuals_composed", state)
    try:
        if individuals:
            with evolution_context(state):
                legacy.check_and_update_fitness(individuals)
        for individual, gene in zip(individuals, genes):
            values = individual.fitness.values
            if (tuple(values) != tuple(legacy.INVALID_FITNESS_MAX)
                    and np.isfinite(values).all()):
                if population == "usage" and any(value < 0 for value in values):
                    continue
                value = (values[1] if population == "usage" or metric == "mse" else values[0])
                info = state["GLOBAL_DATA"].get(individual[0], {})
                samples = info.get("metric_samples", {}).get(metric, value)
                worker.individual_metrics.setdefault(gene, {}).update({
                    metric: samples, "evaluation_id": individual[0],
                    "seeds": info.get("evaluation_seeds", [SAMPLE_SEED]),
                })
    finally:
        save_intermediate(root, workers, generation, attempt, f"{population}_individual_evaluation", state)
    return worker.individual_metrics


def run(checkpoints):
    validate_config(config)
    populations = (config.POPULATIONS if config.SEPARATE_POPULATIONS
                   else {"paired": config.PAIRED_POPULATION})
    if set(config.POPULATIONS) != {"surrogate", "usage"}:
        raise ValueError("MVP requires exactly one surrogate and one usage worker")
    fields = {"seed_file", "start_population_size", "population_size", "num_elites",
              "crossover_probability", "mutation_probability", "llm_model", "prompt_group"}
    if any(set(settings) - {"seed_files", "seed_dir"} != fields
           for settings in [*config.POPULATIONS.values(), *populations.values()]):
        raise ValueError("Unsupported population settings; MVP has one worker per population")
    if any(owner not in config.POPULATIONS for owner in config.COMPOSITION_EXPORTS.values()):
        raise ValueError("Unknown population in composition exports")
    # A nominally impossible seed batch cannot be repaired by regeneration.
    if config.SEPARATE_POPULATIONS and config.SURROGATE_EVOL and config.USAGE_EVOL:
        validate_pairing(config.POPULATIONS["surrogate"]["start_population_size"] if config.SURROGATE_EVOL else 1,
                         config.POPULATIONS["usage"]["start_population_size"] if config.USAGE_EVOL else 1,
                         config.matches, config.kappa)
    if any(not 1 <= settings["num_elites"] < settings["start_population_size"]
           for settings in populations.values()):
        raise ValueError("Infeasible population: starting population must exceed the elite minimum")
    root = Path(checkpoints).resolve()
    root.mkdir(parents=True, exist_ok=True)
    workers = {name: PopulationWorker(name, settings, root)
               for name, settings in populations.items()}
    checkpoint, start, _ = legacy.load_checkpoint(folder_name=str(root), global_path=str(root))
    start = start if checkpoint else 1
    if checkpoint:
        if checkpoint.get("mode") != "coevolution":
            raise ValueError("Use a separate checkpoint directory for coevolution")
        if not configuration_compatible(checkpoint["configuration"]):
            raise ValueError("Coevolution configuration differs from checkpoint; use a new directory")
        for name, worker in workers.items():
            worker.restore(checkpoint["workers"][name])
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_state"])
    pending_path = root / "pending_attempt.pkl"
    pending = None
    if pending_path.exists():
        with pending_path.open("rb") as file:
            pending = pickle.load(file)
        if pending["generation"] < start:
            pending = None  # A completed generation checkpoint supersedes this attempt.
        else:
            if not configuration_compatible(pending["configuration"]):
                raise ValueError("Coevolution configuration differs from pending attempt; use a new directory")
            start = pending["generation"]
            for name, worker in workers.items():
                worker.restore(pending["workers"][name])
            random.setstate(pending["random_state"])
            np.random.set_state(pending["numpy_state"])
    # Same per-worker generation/era bound as run_improved.py; no migration.
    stop = (legacy.num_generations if legacy.migration_gen == 0 else
            ((start + legacy.migration_gen - 1) // legacy.migration_gen) * legacy.migration_gen + 1)
    for generation in range(start, stop):
        attempt = pending["attempt"] if pending else 0
        accumulated_records = pending["records"] if pending else []
        if not pending:
            for worker in workers.values():
                worker.successful = []
        pending = None
        while True:
            attempt += 1
            candidate_ids = {}
            for name, worker in workers.items():
                try:
                    candidate_ids[name] = [ind[0] for ind in worker.candidates(
                        generation, regenerate=attempt > 1)]
                finally:
                    save_intermediate(root, workers, generation, attempt, f"candidates_{name}")
            counts = {name: len(ids) for name, ids in candidate_ids.items()}
            print(f"Generation {generation}, attempt {attempt}: unique candidates={counts}", flush=True)
            try:
                if config.SEPARATE_POPULATIONS and config.SURROGATE_EVOL and config.USAGE_EVOL:
                    validate_pairing(counts["surrogate"], counts["usage"], config.matches, config.kappa)
                elif not all(counts.values()):
                    raise ValueError("No joint candidates")
            except ValueError as exc:
                # A valid configured batch can become infeasible after source deduplication.
                print(f"{exc}; retaining successes and refilling candidates before evaluation.", flush=True)
                continue
            # Validate actual, deduplicated pools before submitting any evaluation jobs.
            if not config.SEPARATE_POPULATIONS:
                pairs = [(gene, gene) for gene in candidate_ids["paired"]]
            elif not (config.SURROGATE_EVOL and config.USAGE_EVOL):
                # The frozen side contains exactly its configured seed. Every
                # evolving candidate must see it, regardless of matching budgets.
                pairs = [(surrogate, usage) for surrogate in candidate_ids["surrogate"]
                         for usage in candidate_ids["usage"]]
            else:
                pairs = random_uniform_coverage(candidate_ids["surrogate"], candidate_ids["usage"],
                                                config.matches, config.kappa)
            print(f"Scheduling {len(pairs)} distinct matches", flush=True)
            if config.SEPARATE_POPULATIONS and distinct_mode():
                evaluate_individuals(workers, generation, attempt, root)
                evaluate_individuals(workers, generation, attempt, root, population="usage")
            records = evaluate_matches(workers, pairs, generation, attempt, root)
            accumulated_records.extend(record for record in records if record.success)
            active_records = [record for record in accumulated_records
                              if all(gene in candidate_ids[name] for name, gene in record.source_ids.items())]
            rank_pairs(active_records, candidate_ids)
            ranks, objectives = configured_objectives(active_records, candidate_ids,
                {name: worker.individual_metrics for name, worker in workers.items()}, config)
            statistics = rank_statistics(ranks, accumulated_records)
            passing = barrier_passes(ranks, {name: settings for name, settings in populations.items()
                                          if not config.SEPARATE_POPULATIONS or getattr(config, name.upper() + "_EVOL")})
            for name, worker in workers.items():
                worker.retain_successes(ranks[name])
            save_intermediate(root, workers, generation, attempt, "ranked")
            report = {"generation": generation, "attempt": attempt, "advances": passing,
                      "matches_cap": config.matches, "attempted_matches": len(pairs),
                      "matches": [record.as_dict() for record in records],
                      "ranks": ranks, "statistics": statistics,
                      "individual_metrics": copy.deepcopy(workers["surrogate"].individual_metrics) if config.SEPARATE_POPULATIONS else {},
                      "usage_individual_metrics": copy.deepcopy(workers["usage"].individual_metrics) if config.SEPARATE_POPULATIONS else {},
                      "configuration": configuration_snapshot()}
            report["candidate_objectives"] = objectives
            report_path = root / "history" / f"generation_{generation}_attempt_{records[0].match_id}.json"
            report_path.parent.mkdir(exist_ok=True)
            # Legacy invalid fitness uses infinities; Python JSON preserves them.
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            temporary = pending_path.with_suffix(".tmp")
            with temporary.open("wb") as file:
                pickle.dump({"generation": generation, "attempt": attempt,
                             "configuration": configuration_snapshot(),
                             "workers": {name: worker.snapshot() for name, worker in workers.items()},
                             "records": accumulated_records,
                             "random_state": random.getstate(), "numpy_state": np.random.get_state()}, file)
            temporary.replace(pending_path)
            print(f"Generation {generation}, attempt {attempt}: {statistics}; advances={passing}", flush=True)
            if passing:
                break
            print("Cumulative per-population minimum not met; retaining successful individuals "
                  "and refilling remaining slots at the same generation barrier.", flush=True)
        for name, worker in workers.items():
            if not config.SEPARATE_POPULATIONS or getattr(config, name.upper() + "_EVOL"):
                worker.select(generation, ranks[name])
        save_intermediate(root, workers, generation, attempt, "selected")
        legacy.save_checkpoint(generation, folder_name=str(root), global_path=None,
                               checkpoint_data={
                                   "mode": "coevolution", "configuration": configuration_snapshot(),
                                   "workers": {name: worker.snapshot() for name, worker in workers.items()},
                                   "random_state": random.getstate(), "numpy_state": np.random.get_state(),
                                   "history": str(report_path),
                               })
        pending_path.unlink(missing_ok=True)
    print("-- End of Era --")


def configuration_compatible(saved):
    """Allow lowering elite/success minima without discarding existing work."""
    if not saved:
        return False
    saved = copy.deepcopy(saved)
    current = configuration_snapshot()
    for name, settings in current["populations"].items():
        previous = saved.get("populations", {}).get(name, {})
        if previous.get("num_elites", 0) < settings["num_elites"]:
            return False
        previous["num_elites"] = settings["num_elites"]
    return saved == current


def configuration_snapshot():
    snapshot = {"populations": copy.deepcopy(config.POPULATIONS),
            "objectives": tuple(config.PAIRED_METRICS), "search_budget": SEARCH_BUDGET,
            "matches": config.matches, "kappa": config.kappa,
            "exports": copy.deepcopy(config.COMPOSITION_EXPORTS)}
    snapshot["experiment"] = {key: copy.deepcopy(getattr(config, key)) for key in
        ("SEPARATE_POPULATIONS", "SURROGATE_EVOL", "USAGE_EVOL", "SURROGATE_METRICS", "USAGE_METRICS", "PAIRED_METRICS", "numEvols", "METRIC_AGGREGATION")}
    if not config.SEPARATE_POPULATIONS:
        snapshot["paired_population"] = copy.deepcopy(config.PAIRED_POPULATION)
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True, help="Dedicated coevolution run directory")
    run(parser.parse_args().checkpoints)
