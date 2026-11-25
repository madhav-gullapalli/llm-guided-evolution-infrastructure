#!/usr/bin/env python3
"""Offline query diversity / locality from BANANAS candidate or trajectory logs.

NAS-Bench-101 uses edit distance for locality; BANANAS mutates by changing one
operation or edge (local acquisition). Here Hamming distance on NAS-Bench-201
``op_indices`` (length 6) is the edit-distance proxy.

Prefer ``candidate_log_*`` rows with ``selected=True`` (actually queried). If
only ``trajectory_*`` exists, every row is treated as a queried architecture.

Examples::

    uv run python sota/Surrogate/analysis_tools/query_diversity.py \\
        --gene-dir sota/Surrogate/run_nb201/xXxBcdUPD7dkWKBm0CTuqipcipE

    uv run python sota/Surrogate/analysis_tools/query_diversity.py \\
        --run surrogate_coevolution_run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_NB201 = REPO_ROOT / "sota" / "Surrogate" / "run_nb201"


def hamming(a, b) -> int:
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError(f"op_indices length mismatch: {len(a)} vs {len(b)}")
    return sum(x != y for x, y in zip(a, b))


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def find_logs(gene_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Return (candidate_log, trajectory) paths under a gene/match output dir."""
    candidates = sorted(gene_dir.glob("**/candidate_log_*.json"))
    trajectories = sorted(gene_dir.glob("**/trajectory_*.json"))
    # Prefer newest under nasbench201 if multiple
    cand = candidates[-1] if candidates else None
    traj = trajectories[-1] if trajectories else None
    return cand, traj


def queried_ops_from_candidate_log(rows: list[dict]) -> list[list[int]]:
    selected = [row for row in rows if row.get("selected")]
    if not selected:
        selected = rows
    ops = []
    for row in selected:
        indices = row.get("op_indices")
        if indices is None:
            continue
        ops.append(list(map(int, indices)))
    return ops


def queried_ops_from_trajectory(rows: list[dict]) -> list[list[int]]:
    ops = []
    for row in rows:
        indices = row.get("op_indices")
        if indices is None:
            continue
        ops.append(list(map(int, indices)))
    return ops


def load_queried_ops(gene_dir: Path) -> tuple[list[list[int]], str]:
    cand_path, traj_path = find_logs(gene_dir)
    if cand_path is not None:
        ops = queried_ops_from_candidate_log(load_json(cand_path))
        if ops:
            return ops, str(cand_path)
    if traj_path is not None:
        ops = queried_ops_from_trajectory(load_json(traj_path))
        if ops:
            return ops, str(traj_path)
    return [], ""


def summarize_ops(ops: list[list[int]], num_init: int = 0) -> dict:
    """Diversity scalars and per-step locality curves for one search."""
    if not ops:
        return {}
    keys = [tuple(op) for op in ops]
    unique = len(set(keys))
    step_dist = [None]
    for i in range(1, len(ops)):
        step_dist.append(hamming(ops[i - 1], ops[i]))

    # Distance to incumbent (best true accuracy so far unknown here) — use
    # first-seen best by order: after init, incumbent = earliest among unique
    # with... we only have ops. Use running "best index" as first architecture
    # then update when we see a new one only for locality-to-previous-best-query
    # BANANAS sense: distance to the last newly improved — without accuracy we
    # use distance to the first arch of the run as a weak baseline AND distance
    # to the most recent unique? Better: distance to ops[0] after start, and
    # distance to running mode of last k? Simplest BANANAS proxy without acc:
    # distance to ops[max(0, i-1)] already in step_dist; plus distance to the
    # initial sample centroid — skip. Use distance to first post-init arch as
    # "anchor" and also compute mean pairwise.

    pairwise = []
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            pairwise.append(hamming(ops[i], ops[j]))

    # Guided phase only (after random init) when num_init known
    guided = ops[num_init:] if num_init and len(ops) > num_init else ops
    guided_step = []
    for i in range(1, len(guided)):
        guided_step.append(hamming(guided[i - 1], guided[i]))

    # Distance to incumbent assuming accuracy improves only when we keep the
    # chronologically first occurrence of each unique — without accuracy, use
    # distance to the best-so-far *first* architecture and update incumbent to
    # current whenever step moves (weak). Prefer: incumbent = ops[0], then for
    # each new point distance to ops[0] is not BANANAS. 
    # With trajectory we often have true_accuracy — reload if present via ops only.
    return {
        "n_queries": len(ops),
        "n_unique": unique,
        "unique_fraction": unique / len(ops),
        "mean_step_hamming": float(np.mean([d for d in step_dist[1:] if d is not None])) if len(ops) > 1 else None,
        "mean_pairwise_hamming": float(np.mean(pairwise)) if pairwise else None,
        "mean_guided_step_hamming": float(np.mean(guided_step)) if guided_step else None,
        "frac_step_le_1": (
            float(np.mean([d <= 1 for d in step_dist[1:] if d is not None])) if len(ops) > 1 else None
        ),
        "step_hamming": step_dist,
        "num_init": num_init,
    }


def summarize_with_accuracy(rows: list[dict], *, selected_only: bool = True, num_init: int = 0) -> dict:
    """Like summarize_ops but uses true_accuracy to track the validation incumbent."""
    if selected_only:
        rows = [row for row in rows if row.get("selected", True)]
    usable = [row for row in rows if row.get("op_indices") is not None]
    if not usable:
        return {}
    ops = [list(map(int, row["op_indices"])) for row in usable]
    base = summarize_ops(ops, num_init=num_init)
    accs = [row.get("true_accuracy") for row in usable]
    if any(a is None for a in accs):
        return base
    incumbent_dist = []
    best_i = 0
    best_acc = accs[0]
    for i, acc in enumerate(accs):
        if i == 0:
            incumbent_dist.append(0)
            continue
        # Distance to the incumbent *before* this query; then update if better.
        incumbent_dist.append(hamming(ops[i], ops[best_i]))
        if acc > best_acc:
            best_acc = acc
            best_i = i
    base["mean_dist_to_incumbent"] = float(np.mean(incumbent_dist[1:])) if len(incumbent_dist) > 1 else None
    base["dist_to_incumbent"] = incumbent_dist
    base["best_accuracy"] = float(max(accs))
    return base


def summarize_gene_dir(gene_dir: Path, num_init: int = 0) -> dict:
    cand_path, traj_path = find_logs(gene_dir)
    source = ""
    summary: dict = {"gene_dir": str(gene_dir)}
    if cand_path is not None:
        rows = load_json(cand_path)
        summary.update(summarize_with_accuracy(rows, selected_only=True, num_init=num_init))
        source = str(cand_path)
    elif traj_path is not None:
        rows = load_json(traj_path)
        # trajectory rows are all evaluated; treat as selected
        for row in rows:
            row.setdefault("selected", True)
        summary.update(summarize_with_accuracy(rows, selected_only=True, num_init=num_init))
        source = str(traj_path)
    summary["source"] = source
    return summary


def coev_match_dirs(run_dir: Path, run_root: Path = RUN_NB201) -> list[tuple[int, str, Path]]:
    """(generation, match_id, gene_dir) for matches that have logs."""
    history = run_dir / "history"
    found = []
    if not history.is_dir():
        return found
    for path in sorted(history.glob("generation_*_attempt_*.json")):
        report = json.loads(path.read_text())
        generation = int(report.get("generation") or 0)
        for match in report.get("matches") or []:
            match_id = match.get("match_id")
            if not match_id:
                continue
            gene_dir = run_root / match_id
            if not gene_dir.is_dir():
                continue
            # Include failed matches that still wrote query logs (e.g. post-search
            # crash after trajectory/candidate_log were persisted).
            cand, traj = find_logs(gene_dir)
            if cand or traj:
                found.append((generation, match_id, gene_dir))
    return found


def summarize_coevolution_run(run_dir: Path, run_root: Path = RUN_NB201, num_init: int = 20) -> dict:
    rows = []
    by_gen: dict[int, list[dict]] = {}
    for generation, match_id, gene_dir in coev_match_dirs(run_dir, run_root):
        summary = summarize_gene_dir(gene_dir, num_init=num_init)
        if not summary.get("n_queries"):
            continue
        summary["generation"] = generation
        summary["match_id"] = match_id
        rows.append(summary)
        by_gen.setdefault(generation, []).append(summary)
    gen_table = []
    for generation in sorted(by_gen):
        block = by_gen[generation]
        def median(key):
            vals = [b[key] for b in block if b.get(key) is not None]
            return float(np.median(vals)) if vals else None
        gen_table.append({
            "generation": generation,
            "n_matches": len(block),
            "median_unique_fraction": median("unique_fraction"),
            "median_step_hamming": median("mean_step_hamming"),
            "median_pairwise_hamming": median("mean_pairwise_hamming"),
            "median_dist_to_incumbent": median("mean_dist_to_incumbent"),
            "median_frac_step_le_1": median("frac_step_le_1"),
            "median_best_accuracy": median("best_accuracy"),
        })
    return {"matches": rows, "by_generation": gen_table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-dir", type=Path, help="Single gene/match run_nb201 folder")
    parser.add_argument("--run", type=Path, help="Coevolution checkpoints root")
    parser.add_argument("--run-root", type=Path, default=RUN_NB201)
    parser.add_argument("--num-init", type=int, default=20)
    args = parser.parse_args()
    if args.gene_dir:
        summary = summarize_gene_dir(args.gene_dir, num_init=args.num_init)
        print(json.dumps(summary, indent=2))
        return
    if args.run:
        report = summarize_coevolution_run(args.run, args.run_root, num_init=args.num_init)
        print(f"run={args.run}  matches_with_logs={len(report['matches'])}")
        if not report["matches"]:
            print("No candidate_log/trajectory found. Re-run coev with query logging enabled.")
            return
        print(
            f"{'gen':>4} {'n':>4} {'uniq_frac':>9} {'step_ham':>8} "
            f"{'pair_ham':>8} {'inc_dist':>8} {'frac≤1':>7} {'med_acc':>7}"
        )
        for row in report["by_generation"]:
            print(
                f"{row['generation']:>4} {row['n_matches']:>4} "
                f"{(row['median_unique_fraction'] or float('nan')):9.3f} "
                f"{(row['median_step_hamming'] or float('nan')):8.3f} "
                f"{(row['median_pairwise_hamming'] or float('nan')):8.3f} "
                f"{(row['median_dist_to_incumbent'] or float('nan')):8.3f} "
                f"{(row['median_frac_step_le_1'] or float('nan')):7.3f} "
                f"{(row['median_best_accuracy'] or float('nan')):7.2f}"
            )
        return
    raise SystemExit("Pass --gene-dir or --run")


if __name__ == "__main__":
    main()
