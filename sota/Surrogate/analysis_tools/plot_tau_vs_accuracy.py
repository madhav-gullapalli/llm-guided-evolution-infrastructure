#!/usr/bin/env python3
"""Plot surrogate Kendall tau against the accuracy of the architectures the surrogate found.

Reads <run>/analysis/summary.csv from summarize_run.py (one point per successfully evaluated gene).
Left: tau vs guided_best_acc. Right: tau vs guided_mean_acc. Each panel reports Kendall and
Spearman correlation across genes; marker size shows how many genes share the same point.

Use --match-run so two runs share identical x/y limits (easier slide comparisons):

  uv run python sota/Surrogate/analysis_tools/plot_tau_vs_accuracy.py \\
      --run surrogate_islands_run --match-run surrogate_precision_run
  uv run python sota/Surrogate/analysis_tools/plot_tau_vs_accuracy.py \\
      --run surrogate_precision_run --match-run surrogate_islands_run

Writes <run>/analysis/tau_vs_accuracy.png by default.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr

from axis_limits import ACC_MEAN_YLIM, ACC_YLIM, TAU_XLIM

REPO_ROOT = Path(__file__).resolve().parents[3]

PANELS = [
    ("guided_best_acc", "Best accuracy among surrogate-guided picks (%)"),
    ("guided_mean_acc", "Mean accuracy of surrogate-guided picks (%)"),
]


def resolve_run(path: Path) -> Path:
    return path if path.is_absolute() or path.exists() else REPO_ROOT / path


def load_rows(summary: Path) -> list[dict]:
    with open(summary, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    return [r for r in rows if r["kendall_tau"] and r["guided_best_acc"] and r["guided_mean_acc"]]


def correlation_text(x: np.ndarray, y: np.ndarray, min_distinct: int = 5) -> str:
    distinct = len({(round(a, 4), round(b, 4)) for a, b in zip(x, y)})
    counts = f"n = {len(x)} genes ({distinct} distinct scores)"
    if distinct < min_distinct:
        return f"{counts}\ntoo few distinct points for correlation"
    # Genes with identical scores (unchanged code) count once each, so p-values are optimistic.
    k, k_p = kendalltau(x, y)
    s, s_p = spearmanr(x, y)
    return f"Kendall = {k:.2f} (p={k_p:.3f})\nSpearman = {s:.2f} (p={s_p:.3f})\n{counts}"


def padded_limits(vals: list[float], pad_frac: float = 0.05) -> tuple[float, float]:
    lo, hi = min(vals), max(vals)
    pad = max(1e-3, pad_frac * (hi - lo) if hi > lo else abs(hi) * pad_frac or 0.05)
    return (lo - pad, hi + pad)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Run folder, e.g. surrogate_islands_run")
    parser.add_argument("--summary", type=Path, default=None, help="Default: <run>/analysis/summary.csv")
    parser.add_argument("--color-by", default="island", choices=["island", "mode", "generation", "mutation"])
    parser.add_argument(
        "--match-run",
        type=Path,
        default=None,
        help="Another run; axis limits are the union of both summaries so plots compare cleanly",
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help=f"τ x-limits (default chart-normalized {TAU_XLIM[0]}–{TAU_XLIM[1]})",
    )
    parser.add_argument(
        "--ylim-best",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help=f"Y limits for guided best acc (default {ACC_YLIM[0]}–{ACC_YLIM[1]})",
    )
    parser.add_argument(
        "--ylim-mean",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help=f"Y limits for guided mean acc (default {ACC_MEAN_YLIM[0]}–{ACC_MEAN_YLIM[1]})",
    )
    parser.add_argument(
        "--auto-limits",
        action="store_true",
        help="Autoscale axes from data / --match-run (disables chart-normalized defaults)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Default: <run>/analysis/tau_vs_accuracy.png")
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    summary = args.summary or run_dir / "analysis" / "summary.csv"
    if not summary.exists():
        raise SystemExit(f"{summary} not found; run summarize_run.py --run {args.run} first")
    rows = load_rows(summary)
    if not rows:
        raise SystemExit(f"No successfully evaluated genes in {summary}")

    match_rows: list[dict] = []
    if args.match_run is not None:
        match_dir = resolve_run(args.match_run)
        match_summary = match_dir / "analysis" / "summary.csv"
        if not match_summary.exists():
            raise SystemExit(f"{match_summary} not found for --match-run")
        match_rows = load_rows(match_summary)

    limit_rows = rows + match_rows
    if args.auto_limits:
        shared_xlim = padded_limits([float(r["kendall_tau"]) for r in limit_rows])
        shared_ylim = {
            "guided_best_acc": padded_limits([float(r["guided_best_acc"]) for r in limit_rows]),
            "guided_mean_acc": padded_limits([float(r["guided_mean_acc"]) for r in limit_rows]),
        }
    else:
        shared_xlim = TAU_XLIM
        shared_ylim = {
            "guided_best_acc": ACC_YLIM,
            "guided_mean_acc": ACC_MEAN_YLIM,
        }

    groups = sorted({r[args.color_by] or "unknown" for r in rows})
    colors = dict(zip(groups, plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))))
    random_start = {float(r["random_start_best_acc"]) for r in rows if r["random_start_best_acc"]}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (col, ylabel) in zip(axes, PANELS):
        tau = np.array([float(r["kendall_tau"]) for r in rows])
        acc = np.array([float(r[col]) for r in rows])

        counts = Counter((round(t, 4), round(a, 4), r[args.color_by] or "unknown") for t, a, r in zip(tau, acc, rows))
        for group in groups:
            pts = [(t, a, n) for (t, a, g), n in counts.items() if g == group]
            if not pts:
                continue
            t, a, n = map(np.array, zip(*pts))
            ax.scatter(t, a, s=40 + 40 * (n - 1), color=colors[group], alpha=0.7,
                       edgecolor="black", linewidth=0.5, label=group)

        if col == "guided_best_acc" and len(random_start) == 1:
            ax.axhline(next(iter(random_start)), color="gray", linestyle="--", linewidth=1,
                       label="best of random start")

        ax.text(0.03, 0.97, correlation_text(tau, acc), transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.set_xlabel("Surrogate Kendall τ (gene fitness)")
        ax.set_xlim(0, 1)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)

        ax.set_xlim(*(args.xlim if args.xlim is not None else shared_xlim))
        if col == "guided_best_acc":
            ax.set_ylim(*(args.ylim_best if args.ylim_best is not None else shared_ylim[col]))
        else:
            ax.set_ylim(*(args.ylim_mean if args.ylim_mean is not None else shared_ylim[col]))

    legend = axes[0].legend(title=args.color_by, fontsize=8, loc="lower right")
    for handle in getattr(legend, "legend_handles", None) or legend.legendHandles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([40])
    title = f"Kendall τ vs accuracy found — {run_dir.name}  (marker size = genes at that point)"
    fig.suptitle(title)
    fig.tight_layout()

    output = args.output or run_dir / "analysis" / "tau_vs_accuracy.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output}")


if __name__ == "__main__":
    main()
