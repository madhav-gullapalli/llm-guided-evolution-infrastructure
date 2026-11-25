#!/usr/bin/env python3
"""Plot anytime validation regret vs #architecture evaluations.

NAS-Bench-101 / DE-NAS style: regret(t) = optimum − best validation accuracy
in the first t queries. Lower is better. Use ``--match-run`` only for shared
axis limits; the x-axis follows each primary series' own length.

Examples::

    uv run python sota/Surrogate/analysis_tools/plot_search_regret.py \\
        --run surrogate_coevolution_run

    uv run python sota/Surrogate/analysis_tools/plot_search_regret.py \\
        --genes coev_xxx_g1_a1_m0:elite seed:bananas_seed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from axis_limits import REGRET_YLIM
from search_regret import (
    RUN_NB201,
    aggregate_search_rows,
    collect_match_search,
    find_errors_json,
    median_regret_curve,
    space_optimum,
    summarize_errors_file,
)


def _coev_match_ids(run_dir: Path) -> list[str]:
    history = run_dir / "history"
    ids = []
    if not history.is_dir():
        return ids
    for path in sorted(history.glob("generation_*_attempt_*.json")):
        report = json.loads(path.read_text())
        for match in report.get("matches") or []:
            if match.get("success") and match.get("match_id"):
                ids.append(match["match_id"])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Coevolution run root (uses history match ids)")
    parser.add_argument(
        "--genes", nargs="*", default=[],
        help="match_id or gene_id, optionally gene:label",
    )
    parser.add_argument("--run-root", type=Path, default=RUN_NB201)
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--auto-limits", action="store_true")
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument("--aggregate", action="store_true", default=True,
                        help="For --run, plot median regret over matches (default)")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Plot every match curve lightly instead of the median")
    args = parser.parse_args()

    optimum = space_optimum(args.dataset)
    if optimum is None:
        raise SystemExit(f"No NAS-Bench-201 optimum for dataset={args.dataset}")

    series: list[tuple[str, list[float]]] = []
    if args.run:
        run_dir = args.run
        ids = _coev_match_ids(run_dir)
        rows = collect_match_search(ids, run_root=args.run_root, dataset=args.dataset)
        if not rows:
            raise SystemExit(f"No errors.json under {args.run_root} for matches in {run_dir}")
        if args.no_aggregate:
            for row in rows:
                series.append((row["match_id"][:20], row["regret_curve"]))
        else:
            curve = median_regret_curve(rows)
            if curve:
                agg = aggregate_search_rows(rows)
                label = (f"{run_dir.name} median (n={agg['n_trajectories']}, "
                         f"final={agg['median_final_regret']:.2f})")
                series.append((label, curve))
            # Also overlay the single best final-regret trajectory
            best = min(rows, key=lambda r: r["final_regret"])
            series.append((f"best match final={best['final_regret']:.2f}", best["regret_curve"]))
        out = args.output or (run_dir / "analysis" / "search_regret.png")
    else:
        if not args.genes:
            raise SystemExit("Pass --run or --genes")
        for spec in args.genes:
            gene, label = (spec.split(":", 1) + [None])[:2] if ":" in spec else (spec, spec)
            path = find_errors_json(gene, args.run_root)
            if path is None:
                print(f"skip {gene}: no errors.json")
                continue
            row = summarize_errors_file(path, dataset=args.dataset)
            series.append((label or gene, row["regret_curve"]))
        out = args.output or Path("search_regret.png")

    if not series:
        raise SystemExit("Nothing to plot")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, curve in series:
        xs = np.arange(1, len(curve) + 1)
        ax.plot(xs, curve, label=label, linewidth=2 if "median" in label or "best" in label else 1,
                alpha=0.95 if "median" in label or "best" in label else 0.35)
    ax.axhline(0.0, color="0.5", linestyle=":", linewidth=1)
    near = 0.01 * optimum
    ax.axhline(near, color="0.6", linestyle="--", linewidth=1, label=f"1% of opt ≈ {near:.2f}")
    ax.set_xlabel("# architectures evaluated")
    ax.set_ylabel(f"Validation regret (opt {optimum:.2f} − best-so-far)")
    ax.set_title("Anytime validation regret")
    if args.ylim:
        ax.set_ylim(args.ylim)
    elif not args.auto_limits:
        ax.set_ylim(REGRET_YLIM)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for label, curve in series:
        print(f"  {label}: final_regret={curve[-1]:.3f}  n_evals={len(curve)}")


if __name__ == "__main__":
    # Fix the mistaken forward reference: define before import use in type checkers.
    main()
