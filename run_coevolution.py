"""Two-population (surrogate + usage) coevolution driver.

This module reuses the existing LLM-GE machinery from ``run_improved`` (imported
as ``legacy``): individual creation, crossover, mutation, job submission and
result polling are the unchanged legacy primitives.  What is new here is only
the joint loop around them:

1. every population produces a pool of unique candidate sources,
2. candidates are paired (``random_uniform_coverage``), composed into one
   ``model_coev_*`` module per pair and evaluated with ``eval.py --coevolution``,
3. pair results ``(kendall_tau, best_accuracy)`` are Pareto ranked and projected
   back onto the individuals (independent score),
4. generations only advance once each population has *cumulatively* more than
   ``num_elites`` successful individuals; otherwise failed slots are
   regenerated and another evaluation attempt is made,
5. selection (elites + parents) is written to a checkpoint.

Run directly (this is what ``islands_wrapper.spawn_coevolution_islands`` does)::

    uv run python run_coevolution.py --checkpoints DIR
"""
import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import pickle
import random
import time
import uuid
from pathlib import Path

from deap import tools

import run_improved as legacy
import src.cfg.surrogate_coev as config
from src.coevolution import (INVALID_SEARCH_FITNESS, PairRecord, RankProxy,
                             SearchFitness, barrier_passes,
                             random_uniform_coverage, rank_pairs,
                             rank_statistics)
from src.source_composition import compose_sources

SUPPORTED_POPULATIONS = {"surrogate", "usage"}
MAX_FILL_ROUNDS = 3  # extra creation rounds when dedup/failed LLM jobs shrink a pool


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def legacy_globals(**values):
    """Temporarily override module globals of run_improved (restored on exit)."""
    saved = {key: getattr(legacy, key) for key in values}
    try:
        for key, value in values.items():
            setattr(legacy, key, value)
        yield
    finally:
        for key, value in saved.items():
            setattr(legacy, key, value)


def _dump(path, data):
    """Atomically pickle ``data`` to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as file:
        pickle.dump(data, file)
    os.replace(tmp, path)


def _load(path):
    with Path(path).open("rb") as file:
        return pickle.load(file)


def _fresh(individual):
    """New individual with the same genes and an *invalid* fitness."""
    return legacy.creator.Individual(list(individual))


def _sha(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _intermediate(root, generation):
    return Path(root) / "intermediate" / f"generation_{generation}"


def _seed_path(settings):
    seed = Path(settings["seed_file"])
    return seed if seed.is_absolute() else Path(legacy.ROOT_DIR) / seed


def _jobs_dir(root):
    return str(Path(root) / "jobs")


# --------------------------------------------------------------------------- #
# Configuration validation / compatibility
# --------------------------------------------------------------------------- #
def validate_supported(populations=None):
    populations = config.POPULATIONS if populations is None else populations
    if set(populations) != SUPPORTED_POPULATIONS:
        raise ValueError("MVP coevolution supports exactly the populations "
                         f"{sorted(SUPPORTED_POPULATIONS)}; got {sorted(populations)}")
    for name, settings in populations.items():
        if settings.get("num_islands", 1) != 1:
            raise ValueError(f"MVP coevolution supports one island per population; "
                             f"{name} requested {settings['num_islands']}")


def configuration_snapshot():
    """Plain-data copy of every setting that shapes the search."""
    return {
        "populations": copy.deepcopy(config.POPULATIONS),
        "matches": config.matches,
        "kappa": config.kappa,
        "composition_exports": copy.deepcopy(config.COMPOSITION_EXPORTS),
    }


def configuration_compatible(saved):
    """A saved run may resume only if nothing changed, except ``num_elites`` may drop."""
    if not saved:
        return False
    current = configuration_snapshot()
    if set(saved) != set(current):
        return False
    for key in current:
        if key != "populations" and saved[key] != current[key]:
            return False
    if set(saved["populations"]) != set(current["populations"]):
        return False
    for name, old in saved["populations"].items():
        old, new = dict(old), dict(current["populations"][name])
        old_elites, new_elites = old.pop("num_elites"), new.pop("num_elites")
        if old != new or new_elites > old_elites:
            return False
    return True


# --------------------------------------------------------------------------- #
# Population worker
# --------------------------------------------------------------------------- #
class PopulationWorker:
    """One evolving population: candidate generation, dedup and selection."""

    def __init__(self, name, settings, root):
        self.name = name
        self.settings = settings
        self.root = Path(root)
        self.population = []   # candidate pool during a generation, survivors afterwards
        self.parents = []      # mating pool for the next generation
        self.elites = []       # carried unchanged into the next generation
        self.successful = []   # gene ids that succeeded cumulatively this generation
        self.hashes = {}       # sha256(source text) -> set of gene ids owning it
        self._bootstrap_hashes()

    # -- sources / dedup ---------------------------------------------------- #
    @property
    def variants(self):
        return self.root / self.name / "variants"

    def source(self, gene):
        """Path of the generated source for ``gene`` (parent directory exists)."""
        path = self.variants / f"{legacy.MODEL}_{gene}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _bootstrap_hashes(self):
        """Register hashes of already-existing MODEL_*.py files (older runs)."""
        if not self.variants.is_dir():
            return
        prefix = f"{legacy.MODEL}_"
        files = sorted(self.variants.glob(f"{prefix}*.py"),
                       key=lambda path: (path.stat().st_mtime_ns, path.name))
        for path in files:
            digest = _sha(path)
            if digest is not None and digest not in self.hashes:
                self.hashes[digest] = {path.stem[len(prefix):]}

    def remove_historical_duplicates(self, individuals, carried_ids):
        """Drop individuals whose source text was already seen for this population.

        ``carried_ids`` (elites/kept successes) are always preserved.  Genes whose
        file is missing (failed LLM job) are dropped unless carried.
        """
        carried_ids = set(carried_ids)
        kept, seen_ids, batch_hashes = [], set(), set()
        for individual in individuals:
            gene = individual[0]
            if gene in seen_ids:
                continue
            digest = _sha(self.source(gene))
            carried = gene in carried_ids
            if digest is None:
                if carried:
                    kept.append(individual)
                    seen_ids.add(gene)
                else:
                    print(f"\t☠ No source file for {self.name} gene {gene}; dropping", flush=True)
                continue
            owners = self.hashes.get(digest)
            if not carried:
                if (owners and gene not in owners) or digest in batch_hashes:
                    print(f"\t‣ Removing duplicate {self.name} source for gene {gene}", flush=True)
                    continue
            kept.append(individual)
            seen_ids.add(gene)
            batch_hashes.add(digest)
            self.hashes.setdefault(digest, set()).add(gene)
        return kept

    # -- persistence ---------------------------------------------------------- #
    def snapshot(self):
        return {
            "name": self.name,
            "population": [_fresh(ind) for ind in self.population],
            "parents": [_fresh(ind) for ind in self.parents],
            "elites": [_fresh(ind) for ind in self.elites],
            "successful": list(self.successful),
            "hashes": {digest: sorted(genes) for digest, genes in self.hashes.items()},
        }

    def restore(self, snapshot):
        self.population = [_fresh(ind) for ind in snapshot.get("population", [])]
        self.parents = [_fresh(ind) for ind in snapshot.get("parents", [])]
        self.elites = [_fresh(ind) for ind in snapshot.get("elites", [])]
        self.successful = list(snapshot.get("successful", []))
        # Older checkpoints have no registry; always add anything newer on disk.
        self.hashes = {digest: set(genes)
                       for digest, genes in snapshot.get("hashes", {}).items()}
        self._bootstrap_hashes()

    # -- candidate generation -------------------------------------------------- #
    def _context(self, generation, creating):
        elites = self.elites
        return legacy_globals(
            VARIANT_DIR=str(self.variants),
            SEED_NETWORK=str(_seed_path(self.settings)),
            PROMPT_GLOB=legacy.resolve_prompt_glob(self.settings.get("prompt_group")),
            GENERATION=generation,
            OUTPUT_DIR=_jobs_dir(self.root),
            GEN_COUNT=-1 if creating else generation,
            TOP_N_GENES=elites or None,
            PROB_EOT=legacy.PROB_EOT if elites else 0,
        )

    def _create(self, count, generation):
        self.variants.mkdir(parents=True, exist_ok=True)
        model = self.settings["llm_model"]
        with self._context(generation, creating=True):
            created = [legacy.toolbox.individual(llm_model=model) for _ in range(count)]
            legacy.delayed_creation_check(created)
        return created

    def _mutate_from_parents(self, count, generation):
        sources = self.parents or self.elites
        model = self.settings["llm_model"]
        with self._context(generation, creating=False):
            children = [legacy.toolbox.clone(random.choice(sources)) for _ in range(count)]
            children = self._mutate(children, model)
        return children

    def _mutate(self, individuals, model):
        for i, mutant in enumerate(individuals):
            if random.random() < self.settings["mutation_probability"]:
                individuals[i] = legacy.toolbox.mutate(individual=mutant, llm_model=model)
                del individuals[i].fitness.values
        return legacy.delayed_mutate_check(individuals)

    def _offspring(self, generation):
        model = self.settings["llm_model"]
        with self._context(generation, creating=False):
            offspring = [legacy.toolbox.clone(parent) for parent in self.parents]
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < self.settings["crossover_probability"]:
                    offspring[i], offspring[i + 1] = legacy.toolbox.mate(
                        offspring[i], offspring[i + 1], llm_model=model)
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values
            offspring = legacy.delayed_mate_check(offspring)
            offspring = self._mutate(offspring, model)
        return offspring

    def _make(self, count, generation):
        if self.parents or self.elites:
            return self._mutate_from_parents(count, generation)
        return self._create(count, generation)

    def _target_size(self):
        if self.parents or self.elites:
            return len(self.parents) + len(self.elites)
        return self.settings["start_population_size"]

    def candidates(self, generation, regenerate=False):
        """Unique candidate pool for the generation (also stored as ``population``).

        First call per generation builds the pool (seed creation, or offspring of
        the selected parents plus carried elites).  With ``regenerate=True`` the
        cumulative successes are kept and only the failed slots are refilled.
        """
        target = self._target_size()
        if regenerate:
            successes = set(self.successful)
            pool = [ind for ind in self.population if ind[0] in successes]
            pool = self.remove_historical_duplicates(pool, successes)
        else:
            self.successful = []
            if self.parents:
                elites = [_fresh(ind) for ind in self.elites]
                pool = self._offspring(generation) + elites
                pool = self.remove_historical_duplicates(pool, {ind[0] for ind in elites})
            else:
                pool = []
        for _ in range(MAX_FILL_ROUNDS):
            need = target - len(pool)
            if need <= 0:
                break
            batch = self._make(need, generation)
            pool = self.remove_historical_duplicates(pool + batch, {ind[0] for ind in pool})
        self.population = pool
        return pool

    # -- selection --------------------------------------------------------------- #
    def select(self, ranks):
        """Choose elites and parents from individuals that succeeded in some pair."""
        passing = [ind for ind in self.population if ranks.get(ind[0]) is not None]
        proxies = [RankProxy(ind, ranks[ind[0]]) for ind in passing]
        random.shuffle(proxies)  # random tie-breaking between equal ranks
        elite_count = min(self.settings["num_elites"], len(proxies))
        elites = tools.selBest(proxies, elite_count)
        limit = min(max(2, min(self.settings["population_size"], len(proxies))), len(proxies))
        chosen = tools.selTournament(proxies, limit, tournsize=2) if proxies else []
        parents, seen = [], set()
        for proxy in chosen + tools.selBest(proxies, len(proxies)):
            if proxy[0] not in seen and len(parents) < limit:
                seen.add(proxy[0])
                parents.append(proxy)
        self.population = [_fresh(ind) for ind in passing]
        self.elites = [_fresh(ind) for ind in elites]
        self.parents = [_fresh(ind) for ind in parents]


# --------------------------------------------------------------------------- #
# Joint evaluation
# --------------------------------------------------------------------------- #
def _search_values(info):
    """(tau, best_acc) from legacy GLOBAL_DATA, or the invalid objective vector."""
    fitness = (info or {}).get("fitness")
    try:
        values = tuple(float(value) for value in fitness)
    except (TypeError, ValueError):
        return INVALID_SEARCH_FITNESS
    if (info.get("status") != "completed" or len(values) != len(INVALID_SEARCH_FITNESS)
            or not all(math.isfinite(value) for value in values)):
        return INVALID_SEARCH_FITNESS
    return values


def evaluate_matches(workers, pairs, generation, attempt, root):
    """Compose, submit and collect every pair; returns one PairRecord per pair."""
    root = Path(root)
    names = list(workers)
    match_dir = root / "matches" / "variants"
    match_dir.mkdir(parents=True, exist_ok=True)
    inter = _intermediate(root, generation)
    run_id = uuid.uuid4().hex
    exports = {export: names.index(owner)
               for export, owner in config.COMPOSITION_EXPORTS.items()}

    matches = {}
    for index, pair in enumerate(pairs):
        gene = f"coev_{run_id}_g{generation}_a{attempt}_m{index}"
        source_ids = dict(zip(names, pair))
        compose_sources([workers[name].source(source_ids[name]) for name in names],
                        exports, match_dir / f"{legacy.MODEL}_{gene}.py")
        legacy.GLOBAL_DATA[gene] = {
            "sub_flag": True, "job_id": None, "status": "subbed file", "fitness": None,
            "start_time": time.time(), "source_ids": source_ids,
            "generation": generation, "attempt": attempt,
        }
        matches[gene] = source_ids
    _dump(inter / f"a{attempt}_matches_composed.pkl",
          {"generation": generation, "attempt": attempt, "matches": matches})

    individuals = []
    for gene in matches:
        individual = legacy.creator.Individual([gene])
        individual.fitness.values = legacy.PLACEHOLDER_FITNESS
        individuals.append(individual)

    records = None
    eval_runline = legacy.EVAL_RUNLINE
    if "--coevolution" not in eval_runline:
        eval_runline += " --coevolution"
    try:
        with legacy_globals(VARIANT_DIR=str(match_dir), EVAL_RUNLINE=eval_runline,
                            GENERATION=generation, OUTPUT_DIR=_jobs_dir(root)):
            legacy.check_and_update_fitness(individuals)
        records = [PairRecord(gene, source_ids, _search_values(legacy.GLOBAL_DATA.get(gene)),
                              SearchFitness, INVALID_SEARCH_FITNESS)
                   for gene, source_ids in matches.items()]
        return records
    finally:
        _dump(inter / f"a{attempt}_evaluation.pkl", {
            "generation": generation, "attempt": attempt, "completed": records is not None,
            "matches": {gene: {"source_ids": ids, "job": copy.deepcopy(legacy.GLOBAL_DATA.get(gene))}
                        for gene, ids in matches.items()},
            "records": [record.as_dict() for record in records] if records else None,
        })


# --------------------------------------------------------------------------- #
# Persistence of shared state
# --------------------------------------------------------------------------- #
def _save_global(root, generation, phase, workers):
    legacy.GLOBAL_DATA_HIST.update(legacy.GLOBAL_DATA.copy())
    _dump(Path(root) / "global_data" / f"global_gen_{generation}.pkl", {
        "phase": phase,
        "generation": generation,
        "GLOBAL_DATA": legacy.GLOBAL_DATA,
        "GLOBAL_DATA_HIST": legacy.GLOBAL_DATA_HIST,
        "GLOBAL_DATA_ANCESTRY": legacy.GLOBAL_DATA_ANCESTRY,
        "workers": {name: worker.snapshot() for name, worker in workers.items()},
    })


def _load_latest_global(root):
    """Load the newest global file into the (identity-preserved) legacy dicts."""
    folder = Path(root) / "global_data"
    if not folder.is_dir():
        return None
    files = [(legacy.extract_generation(path.name, "global"), path)
             for path in folder.iterdir()]
    files = [(gen, path) for gen, path in files if gen is not None]
    if not files:
        return None
    data = _load(max(files, key=lambda item: item[0])[1])
    legacy.GLOBAL_DATA.update(data.get("GLOBAL_DATA", {}))
    legacy.GLOBAL_DATA_HIST.update(data.get("GLOBAL_DATA_HIST", {}))
    legacy.GLOBAL_DATA_ANCESTRY.update(data.get("GLOBAL_DATA_ANCESTRY", {}))
    return data


def _save_pending(root, generation, attempt, records, workers):
    _dump(Path(root) / "pending_attempt.pkl", {
        "generation": generation, "attempt": attempt, "records": records,
        "workers": {name: worker.snapshot() for name, worker in workers.items()},
        "configuration": configuration_snapshot(),
    })


def _save_checkpoint(root, generation, workers):
    _dump(Path(root) / f"checkpoint_gen_{generation}.pkl", {
        "generation": generation,
        "configuration": configuration_snapshot(),
        "workers": {name: worker.snapshot() for name, worker in workers.items()},
    })


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def _generation_bounds(start_gen):
    """First/last (exclusive) generation of this era, from the legacy controls."""
    migration = legacy.migration_gen
    last = legacy.num_generations + 1
    if migration != 0:
        last = min(last, ((start_gen + migration - 1) // migration) * migration + 1)
    return start_gen, last


def _run_generation(root, workers, generation, pending):
    inter = _intermediate(root, generation)
    inter.mkdir(parents=True, exist_ok=True)
    max_attempts = max(1, getattr(legacy, "max_gen_attempts", 5))
    names = list(workers)
    records, attempt, regenerate = [], 0, False
    if pending is not None:
        records, attempt, regenerate = pending["records"], pending["attempt"], True
    legacy.LINKED_GENES.clear()
    legacy.box_print(f"COEVOLUTION GENERATION: {generation}", new_line_end=False)

    while True:
        # Build/refill pools; a pool smaller than kappa cannot be paired at all.
        for round_index in range(max_attempts):
            for worker in workers.values():
                worker.population = worker.candidates(generation, regenerate=regenerate)
            if min(len(worker.population) for worker in workers.values()) >= config.kappa:
                break
            regenerate = True
        else:
            raise RuntimeError("Could not regenerate enough unique candidates for pairing")
        attempt += 1
        if attempt > max_attempts:
            raise RuntimeError(f"Generation {generation}: independent-score barrier not "
                               f"reached after {max_attempts} attempts")
        for name, worker in workers.items():
            _dump(inter / f"a{attempt}_{name}_candidates.pkl", worker.snapshot())

        pools = [[ind[0] for ind in workers[name].population] for name in names]
        pairs = random_uniform_coverage(pools[0], pools[1], config.matches, config.kappa)
        print(f"Attempt {attempt}: evaluating {len(pairs)} matches "
              f"({' x '.join(str(len(pool)) for pool in pools)} candidates)", flush=True)
        try:
            records.extend(evaluate_matches(workers, pairs, generation, attempt, root))
        except BaseException:
            _save_global(root, generation, "evaluation", workers)
            raise

        candidate_ids = {name: list(dict.fromkeys(
            pools[i] + [record.source_ids[name] for record in records]))
            for i, name in enumerate(names)}
        ranks = rank_pairs(records, candidate_ids)
        for name, worker in workers.items():
            worker.successful = [gene for gene, rank in ranks[name].items() if rank is not None]
        passed = barrier_passes(ranks, config.POPULATIONS)
        history = Path(root) / "history"
        history.mkdir(parents=True, exist_ok=True)
        report = {
            "generation": generation, "attempt": attempt, "matches_cap": config.matches,
            "attempted_matches": len(pairs), "kappa": config.kappa, "barrier_passed": passed,
            "pools": {name: len(pool) for name, pool in zip(names, pools)},
            "statistics": rank_statistics(ranks, records),
        }
        (history / f"generation_{generation}_attempt_{attempt}.json").write_text(
            json.dumps(report, indent=2, default=str))
        _dump(inter / f"a{attempt}_ranks.pkl",
              {"ranks": ranks, "records": [record.as_dict() for record in records]})
        _save_global(root, generation, "evaluation", workers)

        if passed:
            break
        print(f"Barrier not reached after attempt {attempt}; regenerating failed slots",
              flush=True)
        _save_pending(root, generation, attempt, records, workers)
        regenerate = True

    for name, worker in workers.items():
        worker.select(ranks[name])
    _dump(inter / "selection.pkl", {name: worker.snapshot() for name, worker in workers.items()})
    _save_global(root, generation, "selected", workers)
    _save_checkpoint(root, generation, workers)
    pending_file = Path(root) / "pending_attempt.pkl"
    if pending_file.exists():
        pending_file.unlink()
    print(f"Finished coevolution generation {generation} in {attempt} attempt(s)", flush=True)


def _run(root):
    checkpoint, start_gen, _ = legacy.load_checkpoint(str(root), global_path=str(root))
    workers = {name: PopulationWorker(name, settings, root)
               for name, settings in config.POPULATIONS.items()}
    if checkpoint:
        if not configuration_compatible(checkpoint.get("configuration")):
            raise ValueError("Checkpoint configuration is incompatible with the current one")
        for name, worker in workers.items():
            worker.restore(checkpoint["workers"][name])
    else:
        start_gen = 1
    _load_latest_global(root)

    pending = None
    pending_file = root / "pending_attempt.pkl"
    if pending_file.exists():
        saved = _load(pending_file)
        if saved["generation"] == start_gen and configuration_compatible(saved["configuration"]):
            for name, worker in workers.items():
                worker.restore(saved["workers"][name])
            pending = saved
        else:
            pending_file.unlink()

    first, last = _generation_bounds(start_gen)
    for generation in range(first, last):
        _run_generation(root, workers, generation, pending)
        pending = None
    print("-- End of Era --", flush=True)


def run(root):
    """Run one era (``migration_gen`` generations) of two-population coevolution."""
    validate_supported()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Isolated bookkeeping dicts; the caller's legacy globals are restored on exit.
    with legacy_globals(GLOBAL_DATA={}, GLOBAL_DATA_HIST={}, LINKED_GENES={},
                        GLOBAL_DATA_ANCESTRY={legacy.MODEL: {"GENES": [legacy.MODEL],
                                                              "MUTATE_TYPE": ["CREATED"]}}):
        _run(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-population coevolution")
    parser.add_argument("--checkpoints", type=str, required=True, help="Run directory")
    run(parser.parse_args().checkpoints)
