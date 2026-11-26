#!/usr/bin/env python3
"""Summarize a coevolution run: assembly fitness, search regret, coev diagnostics.

History JSON from ``run_coevolution.py`` supplies pair objectives and Pareto /
barrier stats (Potter assembly fitness; Wiegand / Panait collaboration). When
``sota/Surrogate/run_nb201/<match_id>/**/errors.json`` exists, this also reports
NAS-Bench-style **anytime validation regret** and COLE **evals to within 1% of
opt**, and contrasts that bar with **top-1% of the space** (percentile).

Example::

    uv run python sota/Surrogate/analysis_tools/summarize_coevolution.py \\
        --run surrogate_coevolution_run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from query_diversity import summarize_coevolution_run as diversity_by_run  # noqa: E402
from search_regret import (  # noqa: E402
    RUN_NB201,
    aggregate_search_rows,
    collect_match_search,
    near_opt_threshold,
    space_optimum,
    top_percentile_accuracy,
)

LABELS = {
    "best_individual_found": "acc",
    "ktau": "tau",
    "ktau_top_X_percent": "top10",
    "time_to_find_top_X_percent": "steps",
}
SEARCH_OBJECTIVES = {"best_individual_found", "time_to_find_top_X_percent"}
SURROGATE_OBJECTIVES = {"ktau", "ktau_top_X_percent"}
# Accuracy is recorded to about 0.01; smaller gaps are the same score.
CHANGE = {"best_individual_found": 0.01, "time_to_find_top_X_percent": 0.5}
DEFAULT_CHANGE = 1e-4


def load_reports(run_dir: Path) -> list[dict]:
    history = run_dir / "history"
    if not history.is_dir():
        raise SystemExit(f"No history/ under {run_dir} (not a coevolution run?)")
    reports = []
    for path in sorted(history.glob("generation_*_attempt_*.json")):
        reports.append(json.loads(path.read_text()))
    if not reports:
        raise SystemExit(f"No generation_*_attempt_*.json in {history}")
    return reports


def label(name: str) -> str:
    return LABELS.get(name, name)


def gene_key(match: dict):
    ids = match.get("source_ids") or {}
    if "paired" in ids:
        return ids["paired"]
    if not ids:
        return match.get("match_id")
    return tuple(sorted(ids.items()))


def successful_rows(report: dict) -> tuple[list[str], list[dict]]:
    names = list((report.get("configuration") or {}).get("objectives") or [])
    rows = []
    for match in report.get("matches") or []:
        if not match.get("success"):
            continue
        values = match.get("objectives") or []
        if not names:
            names = [f"obj{i}" for i in range(len(values))]
        if len(values) != len(names):
            continue
        row = dict(zip(names, (float(v) for v in values)))
        row["pareto_level"] = match.get("pareto_level")
        row["gene"] = gene_key(match)
        rows.append(row)
    return names, rows


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def summarize_attempt(report: dict) -> dict:
    names, rows = successful_rows(report)
    populations = (report.get("statistics") or {}).get("populations") or {}
    summary = {
        "generation": report.get("generation"),
        "attempt": report.get("attempt"),
        "tried": report.get("attempted_matches", len(report.get("matches") or [])),
        "objectives": names,
        "rows": rows,
        "populations": populations,
    }
    for name in names:
        values = [row[name] for row in rows]
        summary[name] = None if not values else {
            "best": max(values), "median": median(values), "worst": min(values),
        }
    return summary


def best_so_far(attempts: list[dict], metric: str = "best_individual_found") -> list[tuple]:
    """Running maximum of each attempt's best value, in report order."""
    carried = None
    series = []
    for attempt in attempts:
        block = attempt.get(metric)
        current = None if block is None else block["best"]
        if current is not None and (carried is None or current > carried):
            carried = current
        series.append((attempt["generation"], carried))
    return series


def rank0_spread(attempt: dict) -> dict[str, tuple[float, float] | None]:
    spread = {}
    for name in attempt["objectives"]:
        values = [row[name] for row in attempt["rows"] if row["pareto_level"] == 0]
        spread[name] = None if not values else (min(values), max(values))
    return spread


def rescore_changes(attempts: list[dict]) -> list[dict]:
    """Genes evaluated in more than one attempt whose objectives moved."""
    seen: dict = {}
    changes = []
    for attempt in attempts:
        for row in attempt["rows"]:
            previous = seen.get(row["gene"])
            if previous is not None:
                moved = {}
                for name in attempt["objectives"]:
                    if name not in previous:
                        continue
                    gap = abs(row[name] - previous[name])
                    if gap > CHANGE.get(name, DEFAULT_CHANGE):
                        moved[name] = (previous[name], row[name])
                if moved:
                    changes.append({
                        "gene": row["gene"],
                        "from_generation": previous["generation"],
                        "to_generation": attempt["generation"],
                        "moved": moved,
                    })
            seen[row["gene"]] = {name: row[name] for name in attempt["objectives"]}
            seen[row["gene"]]["generation"] = attempt["generation"]
    return changes


def _parent_id(entry: dict, gene: str):
    genes = entry.get("GENES") or []
    if len(genes) < 2 or genes[-1] != gene:
        return None
    parent = genes[-2]
    if not isinstance(parent, str) or parent.startswith("P:"):
        return None
    return parent


def source_comparisons(run_dir: Path, reports: list[dict]) -> dict:
    """Compare each gene file with the seed and, when ancestry says so, its parent."""
    latest = reports[-1].get("configuration") or {}
    if latest.get("experiment", {}).get("SEPARATE_POPULATIONS"):
        populations = latest.get("populations") or {}
    else:
        populations = {"paired": latest.get("paired_population") or {}}
    ancestry = _ancestry(run_dir)
    first_seen = {}
    for report in reports:
        for match in report.get("matches") or []:
            if not match.get("success"):
                continue
            ids = match.get("source_ids") or {}
            generation = report.get("generation")
            if "paired" in ids:
                first_seen.setdefault(("paired", ids["paired"]), generation)
            else:
                for population, gene in ids.items():
                    first_seen.setdefault((population, gene), generation)
    compared = []
    for name, settings in populations.items():
        seed_file = settings.get("seed_file")
        seed_path = Path(seed_file) if seed_file else None
        if seed_path is not None and not seed_path.is_file():
            seed_path = run_dir.parent / seed_file
        seed_hash = _digest(seed_path)
        variant_dir = run_dir / name / "variants"
        genes = [(gene, generation) for (population, gene), generation in first_seen.items()
                 if population == name]
        for gene, generation in sorted(genes, key=lambda item: (item[1] is None, item[1], item[0])):
            digest = _digest(variant_dir / f"model_{gene}.py")
            if digest is None:
                continue
            entry = ancestry.get(gene) or {}
            parent = _parent_id(entry, gene)
            parent_hash = _digest(variant_dir / f"model_{parent}.py") if parent else None
            compared.append({
                "population": name,
                "gene": gene,
                "generation": generation,
                "mutation": (entry.get("MUTATE_TYPE") or [None])[-1],
                "differs_from_seed": None if seed_hash is None else digest != seed_hash,
                "same_as_parent": None if parent_hash is None else digest == parent_hash,
            })
    return {
        "genes": compared,
        "differ_from_seed": sum(row["differs_from_seed"] is True for row in compared),
        "same_as_parent": sum(row["same_as_parent"] is True for row in compared),
        "parent_known": sum(row["same_as_parent"] is not None for row in compared),
        "files": len(compared),
    }


def _digest(path: Path | None):
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ancestry(run_dir: Path) -> dict:
    paths = sorted((run_dir / "global_data").glob("global_gen_*.pkl"))
    if not paths:
        return {}
    try:
        with paths[-1].open("rb") as handle:
            data = pickle.load(handle)
    except Exception:
        return {}
    return data.get("GLOBAL_DATA_ANCESTRY") or {}


def _fmt(name: str, value: float) -> str:
    if name == "best_individual_found":
        return f"{value:.2f}"
    if name == "time_to_find_top_X_percent":
        return f"{value:.0f}"
    return f"{value:.3f}"


def _triple(name: str, block: dict | None) -> str:
    if not block:
        return "—"
    return "/".join(_fmt(name, block[part]) for part in ("best", "median", "worst"))


def _match_ids(report: dict) -> list[str]:
    return [m["match_id"] for m in report.get("matches") or []
            if m.get("success") and m.get("match_id")]


def search_from_objectives(attempts: list[dict], optimum: float | None) -> list[str]:
    """Generation-level regret from pair best_individual_found (no errors.json)."""
    if optimum is None or "best_individual_found" not in (
            attempts[0]["objectives"] if attempts else []):
        return []
    near = near_opt_threshold(optimum)
    top1 = top_percentile_accuracy("cifar100", 1.0)
    lines = [
        "Search from pair objectives (assembly fitness → regret):",
        f"  optimum={optimum:.4f}  within-1%-of-opt bar={near:.4f}  "
        f"top-1% space bar={top1:.4f}" if top1 is not None else
        f"  optimum={optimum:.4f}  within-1%-of-opt bar={near:.4f}",
        f"  note: within-1%-of-opt ≠ top-1% of space "
        f"(bars differ by {near - top1:.2f} acc pts)." if top1 is not None else "",
        f"{'gen':>4}  {'best_acc':>8}  {'best_reg':>8}  {'med_reg':>8}  "
        f"{'frac≥near':>9}  {'frac≥top1%':>10}  {'so_far_reg':>10}",
    ]
    lines = [line for line in lines if line != ""]
    carried_acc = None
    for attempt in attempts:
        block = attempt.get("best_individual_found")
        if not block:
            continue
        accs = [row["best_individual_found"] for row in attempt["rows"]]
        regrets = [optimum - a for a in accs]
        best_acc = max(accs)
        if carried_acc is None or best_acc > carried_acc:
            carried_acc = best_acc
        frac_near = sum(a >= near - 1e-3 for a in accs) / len(accs)
        frac_top = (sum(a >= top1 - 1e-3 for a in accs) / len(accs)) if top1 is not None else None
        lines.append(
            f"{attempt['generation']!s:>4}  {best_acc:8.2f}  {min(regrets):8.3f}  "
            f"{median(regrets):8.3f}  {frac_near:9.2f}  "
            f"{(frac_top if frac_top is not None else float('nan')):10.2f}  "
            f"{optimum - carried_acc:10.3f}"
        )
    return lines


def diversity_section(run_dir: Path, run_root: Path) -> list[str]:
    """Query locality by generation when candidate_log/trajectory files exist."""
    report = diversity_by_run(run_dir, run_root)
    lines = ["Query diversity (Hamming on NB201 op_indices; BANANAS-style locality):"]
    if not report["by_generation"]:
        lines.append(
            "  no candidate_log/trajectory for matches yet — "
            "new coev evals save these via enable_query_logging."
        )
        return lines
    lines.append(
        f"{'gen':>4} {'n':>4} {'uniq_frac':>9} {'step_ham':>8} "
        f"{'pair_ham':>8} {'inc_dist':>8} {'frac≤1':>7} {'med_acc':>7}"
    )
    for row in report["by_generation"]:
        lines.append(
            f"{row['generation']:>4} {row['n_matches']:>4} "
            f"{(row['median_unique_fraction'] or float('nan')):9.3f} "
            f"{(row['median_step_hamming'] or float('nan')):8.3f} "
            f"{(row['median_pairwise_hamming'] or float('nan')):8.3f} "
            f"{(row['median_dist_to_incumbent'] or float('nan')):8.3f} "
            f"{(row['median_frac_step_le_1'] or float('nan')):7.3f} "
            f"{(row['median_best_accuracy'] or float('nan')):7.2f}"
        )
    lines.append(
        "  step Hamming = locality to previous query (can spike); "
        "pairwise Hamming = set-level diversity (smoother, TEG-NAS-style). "
        "Read with regret: low Hamming + falling regret ⇒ focused exploitation; "
        "low Hamming + flat regret ⇒ stuck; high Hamming + falling regret ⇒ diverse success."
    )
    return lines


def search_from_trajectories(reports: list[dict], run_root: Path) -> list[str]:
    """Anytime / near-opt stats from errors.json when match folders exist."""
    lines = ["Search from BANANAS trajectories (errors.json):"]
    any_rows = False
    header = (f"{'gen':>4}  {'n_traj':>6}  {'med_final_reg':>13}  {'med_reg@init':>12}  "
              f"{'frac_near':>9}  {'med_evals_near':>14}  {'med_evals_top1':>14}")
    body = []
    for report in reports:
        ids = _match_ids(report)
        rows = collect_match_search(ids, run_root=run_root)
        if not rows:
            body.append(f"{report.get('generation')!s:>4}  {'—':>6}  (no errors.json for matches)")
            continue
        any_rows = True
        agg = aggregate_search_rows(rows)
        body.append(
            f"{report.get('generation')!s:>4}  {agg['n_trajectories']:6d}  "
            f"{agg['median_final_regret']:13.3f}  "
            f"{(agg['median_regret_after_init'] if agg['median_regret_after_init'] is not None else float('nan')):12.3f}  "
            f"{agg['frac_within_1pct_opt']:9.2f}  "
            f"{(agg['median_evals_to_within_1pct_opt'] if agg['median_evals_to_within_1pct_opt'] is not None else float('nan')):14.1f}  "
            f"{(agg['median_evals_to_top_1pct'] if agg['median_evals_to_top_1pct'] is not None else float('nan')):14.1f}"
        )
    if not any_rows:
        return [
            "Search from BANANAS trajectories (errors.json):",
            f"  none found under {run_root} — pair-objective regret above still applies.",
        ]
    return lines + [header] + body


def meeting_scoreboard(attempts: list[dict], reports: list[dict], run_root: Path) -> list[str]:
    """Compact slide-oriented block: search / surrogate / coev."""
    names = next((a["objectives"] for a in attempts if a["objectives"]), [])
    optimum = space_optimum("cifar100")
    lines = [
        "=== Meeting scoreboard ===",
        "Search (NAS-Bench-101 / BANANAS / DE-NAS): anytime regret + near-opt hit time.",
        "Surrogate (predictor papers): Kendall τ / top-10% τ on held-out ranks.",
        "Coevolution (Potter / Wiegand / Panait): assembly fitness, barrier, κ/matches.",
    ]
    search_names = [n for n in names if n in SEARCH_OBJECTIVES]
    surr_names = [n for n in names if n in SURROGATE_OBJECTIVES]
    lines.append(f"  Logged pair objectives — search: {search_names or '—'}; "
                 f"surrogate: {surr_names or '—'}")
    if optimum is not None and any(a.get("best_individual_found") for a in attempts):
        carried = [v for _, v in best_so_far(attempts) if v is not None]
        best = max(carried)
        lines.append(f"  Best assembly accuracy {best:.2f}  →  regret {optimum - best:.3f} "
                     f"(opt {optimum:.2f})")
        near = near_opt_threshold(optimum)
        all_acc = [row["best_individual_found"] for a in attempts for row in a["rows"]]
        lines.append(f"  Matches ≥ within-1%-of-opt ({near:.2f}): "
                     f"{sum(a >= near - 1e-3 for a in all_acc)}/{len(all_acc)}")
    all_ids = [mid for report in reports for mid in _match_ids(report)]
    traj = collect_match_search(all_ids, run_root=run_root)
    if traj:
        agg = aggregate_search_rows(traj)
        lines.append(
            f"  Trajectories with errors.json: n={agg['n_trajectories']}  "
            f"median final regret={agg['median_final_regret']:.3f}  "
            f"frac within-1%-of-opt={agg['frac_within_1pct_opt']:.2f}  "
            f"median evals-to-near-opt="
            f"{agg['median_evals_to_within_1pct_opt'] if agg['median_evals_to_within_1pct_opt'] is not None else '—'}"
        )
        if agg.get("median_regret_after_init") is not None:
            lines.append(
                f"  Median regret after random init ({traj[0].get('num_init', '?')} arches): "
                f"{agg['median_regret_after_init']:.3f} "
                f"(if this ≈ final regret, later search bought little)"
            )
    last = attempts[-1]["populations"] if attempts else {}
    if "paired" in last:
        block = last["paired"]
        lines.append(f"  Coev last gen: passing={block.get('passing')}  "
                     f"rank_0={block.get('rank_0')}  ranks={block.get('rank_counts')}")
    if any(name in last for name in ("surrogate", "usage")):
        for name in ("surrogate", "usage"):
            block = last.get(name) or {}
            if block:
                lines.append(f"  Coev {name}: passing={block.get('passing')}  "
                             f"rank_0={block.get('rank_0')}")
    else:
        lines.append("  Attribution note: joint run only logs paired stats — "
                     "freeze-one-side / separate populations still needed to split "
                     "search vs surrogate credit.")
    cfg = (reports[-1].get("configuration") or {}) if reports else {}
    lines.append(f"  Collaboration: matches={cfg.get('matches')}  kappa={cfg.get('kappa')}")
    return lines


def render(run_dir: Path, reports: list[dict], run_root: Path = RUN_NB201) -> str:
    attempts = [summarize_attempt(report) for report in reports]
    names = next((attempt["objectives"] for attempt in attempts if attempt["objectives"]), [])
    optimum = space_optimum("cifar100")
    lines = meeting_scoreboard(attempts, reports, run_root)
    lines.append("")
    lines.append(
        f"run={run_dir}  attempts={len(attempts)}  objectives={','.join(names) or 'none'}"
    )
    lines.append("best/median/worst are over successful matches. so_far carries best accuracy.")
    header = f"{'gen':>4} {'att':>4} {'tried':>6} {'pass':>5} {'r0':>4}"
    for name in names:
        header += f"  {label(name) + ' b/m/w':>16}"
    if "best_individual_found" in names:
        header += f"  {'so_far':>7}"
        if optimum is not None:
            header += f"  {'regret':>7}"
    lines.append(header)
    carried = dict(best_so_far(attempts))
    for attempt in attempts:
        paired = attempt["populations"].get("paired") or {}
        passing = paired.get("passing")
        rank0 = paired.get("rank_0")
        if passing is None:
            passing = sum(row is not None for row in (attempt["rows"]))
            rank0 = sum(row["pareto_level"] == 0 for row in attempt["rows"])
        row = (f"{attempt['generation']!s:>4} {attempt['attempt']!s:>4} "
               f"{attempt['tried']!s:>6} {passing!s:>5} {rank0!s:>4}")
        for name in names:
            row += f"  {_triple(name, attempt.get(name)):>16}"
        if "best_individual_found" in names:
            so_far = carried.get(attempt["generation"])
            row += f"  {_fmt('best_individual_found', so_far) if so_far is not None else '—':>7}"
            if optimum is not None and so_far is not None:
                row += f"  {optimum - so_far:7.3f}"
        lines.append(row)
    lines.append("")
    lines.extend(search_from_objectives(attempts, optimum))
    lines.append("")
    lines.extend(search_from_trajectories(reports, run_root))
    lines.append("")
    lines.extend(diversity_section(run_dir, run_root))
    extra = [name for name in ("surrogate", "usage") if any(name in a["populations"] for a in attempts)]
    if extra:
        lines.append("")
        lines.append("Separate populations (passing / rank 0):")
        for attempt in attempts:
            parts = []
            for name in extra:
                block = attempt["populations"].get(name) or {}
                parts.append(f"{name} {block.get('passing', 0)}/{block.get('rank_0', 0)}")
            lines.append(f"  gen {attempt['generation']}: " + "  ".join(parts))
    lines.append("")
    lines.append("Rank-0 range (min–max). A tight range means the front is one point.")
    for attempt in attempts:
        parts = []
        for name, span in rank0_spread(attempt).items():
            if span is None:
                parts.append(f"{label(name)} —")
            else:
                parts.append(f"{label(name)} {_fmt(name, span[0])}–{_fmt(name, span[1])}")
        lines.append(f"  gen {attempt['generation']}: " + "  ".join(parts))
    reached = [(gen, value) for gen, value in best_so_far(attempts) if value is not None]
    if reached:
        best = max(value for _, value in reached)
        first = next(gen for gen, value in reached if abs(value - best) <= 1e-6)
        lines.append("")
        lines.append(f"Best-so-far accuracy {_fmt('best_individual_found', best)} "
                     f"first reached at generation {first}"
                     + (f" (regret {optimum - best:.3f})." if optimum is not None else "."))
    scored = {}
    for attempt in attempts:
        for row in attempt["rows"]:
            scored.setdefault(row["gene"], set()).add(attempt["generation"])
    repeated = [gene for gene, gens in scored.items() if len(gens) > 1]
    changes = rescore_changes(attempts)
    lines.append("")
    lines.append(f"Re-score: {len(repeated)} genes evaluated in more than one generation, "
                 f"{len(changes)} changed.")
    for change in changes:
        moved = ", ".join(
            f"{label(name)} {_fmt(name, before)}->{_fmt(name, after)}"
            for name, (before, after) in change["moved"].items())
        gene = change["gene"]
        shown = gene if isinstance(gene, str) else dict(gene)
        lines.append(f"  {shown} gen {change['from_generation']}->{change['to_generation']}: {moved}")
    sources = source_comparisons(run_dir, reports)
    lines.append("")
    if not sources["files"]:
        lines.append("Source check: no variant files matched the evaluated genes.")
    else:
        lines.append(
            f"Source check: {sources['differ_from_seed']}/{sources['files']} genes differ from the seed. "
            f"{sources['same_as_parent']}/{sources['parent_known']} with a parent file match that parent."
        )
        copied = [row for row in sources["genes"] if row["same_as_parent"] is True]
        for row in copied[:8]:
            lines.append(f"  {row['gene']} gen {row['generation']} {row['mutation']} identical to parent")
        if len(copied) > 8:
            lines.append(f"  … {len(copied) - 8} more identical to their parent")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Coevolution checkpoints root")
    parser.add_argument(
        "--run-root", type=Path, default=RUN_NB201,
        help="Directory with <match_id>/nasbench201/.../errors.json",
    )
    args = parser.parse_args()
    run_dir = Path(args.run)
    print(render(run_dir, load_reports(run_dir), run_root=args.run_root))


if __name__ == "__main__":
    main()
