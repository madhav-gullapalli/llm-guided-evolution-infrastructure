#!/usr/bin/env python3
"""Plot search outcomes from summarize_run.py: guided accuracy and within-1%-of-optimum reach.

Reads <run>/analysis/summary.csv. Two panels:
  1) Best / median guided_best_acc of genes created each generation (+ random-start baseline)
  2) Fraction of genes that reached within 1% of the NAS-Bench-201 optimum, and median
     evaluation count among those that reached it (COLE's evals_to_within_1pct_opt)

Use --match-run so two runs share identical x/y limits:

  uv run python sota/Surrogate/analysis_tools/plot_search_outcomes.py \\
      --run surrogate_precision_seed1_run --match-run surrogate_mse_seed1_run
  uv run python sota/Surrogate/analysis_tools/plot_search_outcomes.py \\
      --run surrogate_mse_seed1_run --match-run surrogate_precision_seed1_run

Writes <run>/analysis/search_outcomes.png by default.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from axis_limits import ACC_YLIM, REACH_FRAC_YLIM

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_run(path: Path) -> Path:
    return path if path.is_absolute() or path.exists() else REPO_ROOT / path


def load_rows(summary: Path) -> list[dict]:
    with open(summary, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok" and r.get("generation")]
    return [r for r in rows if r.get("guided_best_acc")]


def by_generation(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        out[int(r["generation"])].append(r)
    return out


def padded(vals: list[float], pad_frac: float = 0.08) -> tuple[float, float]:
    lo, hi = min(vals), max(vals)
    pad = max(0.5, pad_frac * (hi - lo) if hi > lo else 1.0)
    return (lo - pad, hi + pad)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--match-run", type=Path, default=None)
    parser.add_argument(
        "--ylim-acc",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help=f"Guided accuracy y-limits (default chart-normalized {ACC_YLIM[0]}–{ACC_YLIM[1]})",
    )
    parser.add_argument(
        "--auto-limits",
        action="store_true",
        help="Autoscale accuracy from data / --match-run",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    summary = args.summary or run_dir / "analysis" / "summary.csv"
    if not summary.exists():
        raise SystemExit(f"{summary} not found; run summarize_run.py --run {args.run} --seed 1 first")
    rows = load_rows(summary)
    if not rows:
        raise SystemExit(f"No genes with guided_best_acc in {summary}")

    match_rows: list[dict] = []
    if args.match_run is not None:
        match_summary = resolve_run(args.match_run) / "analysis" / "summary.csv"
        if not match_summary.exists():
            raise SystemExit(f"{match_summary} not found for --match-run")
        match_rows = load_rows(match_summary)

    by_gen = by_generation(rows)
    gens = sorted(by_gen)
    all_gens = sorted({int(r["generation"]) for r in rows + match_rows}) if match_rows else gens
    xlim = (min(all_gens), max(all_gens))

    acc_vals = [float(r["guided_best_acc"]) for r in rows + match_rows]
    for r in rows + match_rows:
        if r.get("random_start_best_acc"):
            acc_vals.append(float(r["random_start_best_acc"]))
    if args.ylim_acc is not None:
        ylim_acc = tuple(args.ylim_acc)
    elif args.auto_limits:
        ylim_acc = padded(acc_vals)
    else:
        ylim_acc = ACC_YLIM

    fig, (ax_acc, ax_opt) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    best = [max(float(r["guided_best_acc"]) for r in by_gen[g]) for g in gens]
    med = [float(np.median([float(r["guided_best_acc"]) for r in by_gen[g]])) for g in gens]
    ax_acc.plot(gens, best, "-o", color="tab:green", label="guided best acc: best of gen")
    ax_acc.plot(gens, med, ":s", color="tab:green", alpha=0.75, label="guided best acc: median of gen")
    rs = {r["random_start_best_acc"] for r in rows if r.get("random_start_best_acc")}
    if len(rs) == 1:
        ax_acc.axhline(float(rs.pop()), color="gray", linestyle="--", linewidth=1, label="random-start best acc")
    ax_acc.set_ylim(*ylim_acc)
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))

    fracs, med_evals, eval_gens = [], [], []
    for g in gens:
        grp = by_gen[g]
        reached = [float(r["evals_to_within_1pct_opt"]) for r in grp if r.get("evals_to_within_1pct_opt")]
        fracs.append(len(reached) / len(grp))
        if reached:
            eval_gens.append(g)
            med_evals.append(float(np.median(reached)))
    ax_opt.plot(gens, fracs, "-o", color="tab:purple", label="fraction reaching within 1% of optimum")
    ax_opt.set_ylim(*REACH_FRAC_YLIM)
    ax_opt.set_ylabel("Fraction of genes")
    ax_opt.grid(True, linestyle="--", alpha=0.5)

    ax_evals = ax_opt.twinx()
    if med_evals:
        ax_evals.plot(eval_gens, med_evals, ":D", color="tab:brown", alpha=0.85, label="median evals to within 1% (reachers)")
    ax_evals.set_ylabel("Evals to within 1% of opt")
    # Match twin y across runs using union of reachers' eval counts
    eval_all = [
        float(r["evals_to_within_1pct_opt"])
        for r in rows + match_rows
        if r.get("evals_to_within_1pct_opt")
    ]
    if eval_all:
        ax_evals.set_ylim(*padded(eval_all, pad_frac=0.1))

    lines_l, labels_l = ax_opt.get_legend_handles_labels()
    lines_r, labels_r = ax_evals.get_legend_handles_labels()
    ax_opt.legend(lines_l + lines_r, labels_l + labels_r, fontsize=8, loc="center left", bbox_to_anchor=(1.12, 0.5))

    for g in gens:
        ax_opt.annotate(
            f"n={len(by_gen[g])}",
            (g, 0),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
            color="gray",
            xycoords=("data", "axes fraction"),
        )

    ax_opt.set_xlim(*xlim)
    ax_opt.set_xticks(gens)
    ax_opt.set_xlabel("LLM-GE generation (genes created in that generation)")

    title = f"Search outcomes by generation — {run_dir.name}"
    ax_acc.set_title(title)
    fig.tight_layout()

    output = args.output or run_dir / "analysis" / "search_outcomes.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output}")


if __name__ == "__main__":
    main()
