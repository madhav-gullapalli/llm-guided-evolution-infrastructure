#!/usr/bin/env python3
"""Plot Kendall tau and top-10% tau across LLM-GE generations, against the parameter-count baseline.

Reads <run>/analysis/summary.csv from summarize_run.py. For each generation, draws the best and
median tau of the successfully evaluated genes created in that generation (elites carried over
from earlier generations are not repeated), for overall tau and top-10% tau, plus dashed lines
for a no-learning baseline that ranks architectures by parameter count. Lines above the baseline
mean the surrogates learned something beyond "bigger is better".

Example (from repo root):
  uv run python sota/Surrogate/analysis_tools/plot_tau_trend.py --run surrogate_islands_run
Writes <run>/analysis/tau_trend.png by default.
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

REPO_ROOT = Path(__file__).resolve().parents[3]

SERIES = [
    ("kendall_tau", "param_count_tau", "Kendall τ (all test architectures)", "tab:blue"),
    ("top10_tau", "param_count_top10_tau", "Top-10% τ (best architectures only)", "tab:orange"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Run folder, e.g. surrogate_islands_run")
    parser.add_argument("--summary", type=Path, default=None, help="Default: <run>/analysis/summary.csv")
    parser.add_argument("--output", type=Path, default=None, help="Default: <run>/analysis/tau_trend.png")
    args = parser.parse_args()

    run_dir = args.run if args.run.is_absolute() or args.run.exists() else REPO_ROOT / args.run
    summary = args.summary or run_dir / "analysis" / "summary.csv"
    if not summary.exists():
        raise SystemExit(f"{summary} not found; run summarize_run.py --run {args.run} first")
    with open(summary, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok" and r.get("top10_tau") and r["generation"]]
    if not rows:
        raise SystemExit(f"No genes with top10_tau in {summary}; re-run summarize_run.py to add the column")

    by_gen: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_gen[int(r["generation"])].append(r)
    gens = sorted(by_gen)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col, base_col, label, color in SERIES:
        values = [[float(r[col]) for r in by_gen[g]] for g in gens]
        ax.plot(gens, [max(v) for v in values], "-o", color=color, label=f"{label}: best")
        ax.plot(gens, [float(np.median(v)) for v in values], ":s", color=color, alpha=0.7, label=f"{label}: median")
        baselines = {r[base_col] for r in rows if r[base_col]}
        if len(baselines) == 1:
            ax.axhline(float(baselines.pop()), color=color, linestyle="--", linewidth=1,
                       label=f"{label}: parameter-count baseline")

    for g in gens:
        ax.annotate(f"n={len(by_gen[g])}", (g, 0), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, color="gray", xycoords=("data", "axes fraction"))
    ax.set_xticks(gens)
    ax.set_xlabel("LLM-GE generation (genes created in that generation)")
    ax.set_ylabel("Kendall τ on held-out test architectures")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(f"Surrogate ranking quality by generation — {run_dir.name}")
    fig.tight_layout()

    output = args.output or run_dir / "analysis" / "tau_trend.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output}")


if __name__ == "__main__":
    main()
