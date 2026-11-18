#!/usr/bin/env python3
"""Plot Kendall tau and top-10% tau across LLM-GE generations, against the parameter-count baseline.

Reads <run>/analysis/summary.csv from summarize_run.py. For each generation, draws the best and
median tau of the successfully evaluated genes created in that generation (elites carried over
from earlier generations are not repeated), for overall tau and top-10% tau, plus dashed lines
for a no-learning baseline that ranks architectures by parameter count. Lines above the baseline
mean the surrogates learned something beyond "bigger is better".

Use --match-run so two runs share identical x/y limits (easier slide comparisons).
Use --split-panels so overall τ and top-10% τ get separate y-scales (top-10% stays readable):

  uv run python sota/Surrogate/analysis_tools/plot_tau_trend.py \\
      --run surrogate_precision_seed1_run --match-run surrogate_mse_seed1_run --split-panels
  uv run python sota/Surrogate/analysis_tools/plot_tau_trend.py \\
      --run surrogate_mse_seed1_run --match-run surrogate_precision_seed1_run --split-panels

Writes <run>/analysis/tau_trend.png (or tau_trend_split.png with --split-panels).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from axis_limits import TAU_YLIM, TOP10_TAU_YLIM

REPO_ROOT = Path(__file__).resolve().parents[3]

SERIES = [
    ("kendall_tau", "param_count_tau", "Kendall τ (all test architectures)", "tab:blue"),
    ("top10_tau", "param_count_top10_tau", "Top-10% τ (best architectures only)", "tab:orange"),
]


def resolve_run(path: Path) -> Path:
    return path if path.is_absolute() or path.exists() else REPO_ROOT / path


def load_rows(summary: Path) -> list[dict]:
    with open(summary, newline="") as f:
        return [
            r
            for r in csv.DictReader(f)
            if r["status"] == "ok" and r.get("top10_tau") and r["generation"]
        ]


def by_generation(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        out[int(r["generation"])].append(r)
    return out


def series_limits(
    rows_list: list[list[dict]],
    series: list[tuple] | None = None,
) -> tuple[float, float]:
    """Y limits from best/median-of-generation curves (not raw gene outliers)."""
    series = series or SERIES
    vals: list[float] = []
    for rows in rows_list:
        by_gen = by_generation(rows)
        for col, base_col, *_ in series:
            for g_rows in by_gen.values():
                col_vals = [float(r[col]) for r in g_rows if r.get(col)]
                if col_vals:
                    vals.append(max(col_vals))
                    vals.append(float(np.median(col_vals)))
            vals.extend(float(r[base_col]) for r in rows if r.get(base_col))
    if not vals:
        return (0.0, 1.0)
    lo, hi = min(vals), max(vals)
    pad = max(0.02, 0.08 * (hi - lo) if hi > lo else 0.05)
    return (lo - pad, hi + pad)


def plot_series_on_ax(ax, by_gen, gens, rows, series_spec, show_n: bool) -> None:
    col, base_col, label, color = series_spec
    values = [[float(r[col]) for r in by_gen[g] if r.get(col)] for g in gens]
    ax.plot(gens, [max(v) for v in values], "-o", color=color, label=f"{label}: best")
    ax.plot(
        gens,
        [float(np.median(v)) for v in values],
        ":s",
        color=color,
        alpha=0.7,
        label=f"{label}: median",
    )
    baselines = {r[base_col] for r in rows if r.get(base_col)}
    if len(baselines) == 1:
        ax.axhline(
            float(baselines.pop()),
            color=color,
            linestyle="--",
            linewidth=1,
            label=f"{label}: parameter-count baseline",
        )
    if show_n:
        for g in gens:
            ax.annotate(
                f"n={len(by_gen[g])}",
                (g, 0),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
                color="gray",
                xycoords=("data", "axes fraction"),
            )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Run folder, e.g. surrogate_islands_run")
    parser.add_argument("--summary", type=Path, default=None, help="Default: <run>/analysis/summary.csv")
    parser.add_argument(
        "--match-run",
        type=Path,
        default=None,
        help="Another run; axis limits are the union of both summaries so plots compare cleanly",
    )
    parser.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"))
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help=f"Overall τ y-limits (default chart-normalized {TAU_YLIM[0]}–{TAU_YLIM[1]})",
    )
    parser.add_argument(
        "--ylim-top10",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help=f"Top-10%% τ y-limits with --split-panels (default {TOP10_TAU_YLIM[0]}–{TOP10_TAU_YLIM[1]})",
    )
    parser.add_argument(
        "--auto-limits",
        action="store_true",
        help="Autoscale y from data (disables chart-normalized defaults)",
    )
    parser.add_argument(
        "--split-panels",
        action="store_true",
        help="Separate panels for overall τ and top-10%% τ (readable top-10%% scale; matched per panel)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Default: <run>/analysis/tau_trend.png")
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    summary = args.summary or run_dir / "analysis" / "summary.csv"
    if not summary.exists():
        raise SystemExit(f"{summary} not found; run summarize_run.py --run {args.run} first")
    rows = load_rows(summary)
    if not rows:
        raise SystemExit(f"No genes with top10_tau in {summary}; re-run summarize_run.py to add the column")

    match_rows: list[dict] = []
    if args.match_run is not None:
        match_dir = resolve_run(args.match_run)
        match_summary = match_dir / "analysis" / "summary.csv"
        if not match_summary.exists():
            raise SystemExit(f"{match_summary} not found for --match-run")
        match_rows = load_rows(match_summary)

    all_rows = [rows] + ([match_rows] if match_rows else [])
    by_gen = by_generation(rows)
    gens = sorted(by_gen)

    if args.xlim is not None:
        xlim = tuple(args.xlim)
    elif match_rows:
        all_gens = sorted({int(r["generation"]) for r in rows + match_rows})
        xlim = (min(all_gens), max(all_gens))
    else:
        xlim = (min(gens), max(gens))

    title = f"Surrogate ranking quality by generation — {run_dir.name}"

    if args.split_panels:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for ax, series_spec in zip(axes, SERIES):
            plot_series_on_ax(ax, by_gen, gens, rows, series_spec, show_n=(ax is axes[-1]))
            ax.set_xlim(*xlim)
            ax.set_xticks(gens)
            if series_spec is SERIES[0]:
                if args.ylim is not None:
                    ax.set_ylim(*args.ylim)
                elif args.auto_limits:
                    ax.set_ylim(*series_limits(all_rows, series=[series_spec]))
                else:
                    ax.set_ylim(*TAU_YLIM)
            else:
                if args.ylim_top10 is not None:
                    ax.set_ylim(*args.ylim_top10)
                elif args.auto_limits:
                    ax.set_ylim(*series_limits(all_rows, series=[series_spec]))
                else:
                    ax.set_ylim(*TOP10_TAU_YLIM)
            ax.set_ylabel("Kendall τ (held-out test)")
        axes[0].set_title(title)
        axes[-1].set_xlabel("LLM-GE generation (genes created in that generation)")
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for series_spec in SERIES:
            plot_series_on_ax(ax, by_gen, gens, rows, series_spec, show_n=True)
        ax.set_xlim(*xlim)
        ax.set_xticks(gens)
        if args.ylim is not None:
            ax.set_ylim(*args.ylim)
        elif args.auto_limits:
            ax.set_ylim(*series_limits(all_rows))
        else:
            ax.set_ylim(*TAU_YLIM)
        ax.set_xlabel("LLM-GE generation (genes created in that generation)")
        ax.set_ylabel("Kendall τ on held-out test architectures")
        ax.set_title(title)

    paired = any(r.get("mode") == "COEVOLUTION" for r in rows)
    for axis in fig.axes:
        axis.set_ylabel("Kendall τ (decimal)")
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    fig.axes[-1].set_xlabel("Coevolution generation (evaluated pairs)" if paired else
                            "LLM-GE generation (genes created in that generation)")
    fig.tight_layout()

    output = args.output or run_dir / "analysis" / ("tau_trend_split.png" if args.split_panels else "tau_trend.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output}")


if __name__ == "__main__":
    main()
