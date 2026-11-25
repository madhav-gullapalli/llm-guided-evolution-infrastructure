#!/usr/bin/env python3
"""Plot query locality / diversity next to search quality.

  uv run python sota/Surrogate/analysis_tools/plot_query_diversity.py \\
      --gene-dir sota/Surrogate/run_nb201/<gene>

  uv run python sota/Surrogate/analysis_tools/plot_query_diversity.py \\
      --run surrogate_coevolution_run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from query_diversity import (  # noqa: E402
    RUN_NB201,
    find_logs,
    load_json,
    summarize_coevolution_run,
    summarize_with_accuracy,
)
from search_regret import load_errors, regret_curve, space_optimum


def plot_gene(gene_dir: Path, output: Path, num_init: int = 20) -> None:
    cand, traj = find_logs(gene_dir)
    if cand is None and traj is None:
        raise SystemExit(f"No candidate_log/trajectory under {gene_dir}")
    if cand is not None:
        rows = load_json(cand)
    else:
        rows = load_json(traj)
        for row in rows:
            row.setdefault("selected", True)
    summary = summarize_with_accuracy(rows, selected_only=True, num_init=num_init)
    step = summary.get("step_hamming") or []
    inc = summary.get("dist_to_incumbent") or []
    xs = np.arange(1, len(step) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    ax = axes[0]
    if len(step) > 1:
        ax.plot(xs[1:], [d for d in step[1:]], "o-", markersize=3, label="Hamming to previous")
    if len(inc) > 1:
        ax.plot(xs[1:], inc[1:], "s--", markersize=3, alpha=0.8, label="Hamming to val incumbent")
    if num_init:
        ax.axvline(num_init, color="0.5", ls=":", label=f"num_init={num_init}")
    ax.set_ylabel("Edit distance (Hamming)")
    ax.set_title(
        f"Query locality  unique={summary.get('n_unique')}/{summary.get('n_queries')}  "
        f"frac≤1={summary.get('frac_step_le_1')}"
    )
    ax.set_ylim(-0.2, 6.2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    err_path = None
    paths = sorted(gene_dir.glob("**/errors.json"))
    if paths:
        err_path = paths[0]
    if err_path is not None:
        config, valid = load_errors(err_path)
        opt = space_optimum(config.get("dataset") or "cifar100")
        if opt is not None and valid:
            regrets = regret_curve(valid, opt)
            ax.plot(np.arange(1, len(regrets) + 1), regrets, label=f"validation regret (opt {opt:.2f})")
            ax.axhline(0.01 * opt, color="0.6", ls="--", label="within-1%-of-opt")
    ax.set_xlabel("# architectures evaluated")
    ax.set_ylabel("Validation regret")
    ax.set_title("Search quality (same trajectory)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("step_hamming", "dist_to_incumbent")}, indent=2))


def plot_run(run_dir: Path, output: Path, run_root: Path, num_init: int) -> None:
    report = summarize_coevolution_run(run_dir, run_root, num_init=num_init)
    if not report["by_generation"]:
        raise SystemExit(
            f"No diversity logs for {run_dir}. "
            "Re-run coevolution after enable_query_logging (candidate_log/trajectory)."
        )
    gens = [r["generation"] for r in report["by_generation"]]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True)

    # Step Hamming is spiky (locality to previous query). Pairwise mean is the
    # set-level diversity summary (TEG-NAS-style) and is usually smoother.
    ax = axes[0]
    ax.plot(gens, [r["median_step_hamming"] for r in report["by_generation"]], "o-", label="median step Hamming")
    ax.plot(
        gens,
        [r["median_pairwise_hamming"] for r in report["by_generation"]],
        "D-",
        label="median pairwise Hamming",
    )
    ax.plot(gens, [r["median_dist_to_incumbent"] for r in report["by_generation"]], "s--", label="median dist to incumbent")
    ax.set_ylabel("Hamming (0–6)")
    ax.set_ylim(-0.2, 6.2)
    ax.set_title(f"Query diversity by generation — {run_dir.name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(gens, [r["median_unique_fraction"] for r in report["by_generation"]], "^-", color="C2", label="median unique fraction")
    ax.plot(gens, [r["median_frac_step_le_1"] for r in report["by_generation"]], "v--", color="C4", label="median frac step ≤1")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(gens, [r["median_best_accuracy"] for r in report["by_generation"]], "o-", color="C3", label="median best acc")
    ax.set_xlabel("Coevolution generation")
    ax.set_ylabel("Best accuracy (median over matches)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")
    for row in report["by_generation"]:
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-dir", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--run-root", type=Path, default=RUN_NB201)
    parser.add_argument("--num-init", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.gene_dir:
        out = args.output or (args.gene_dir / "analysis" / "query_diversity.png")
        # gene_dir may be run_nb201/<id>/... nested; write beside first errors parent
        if args.output is None:
            cand, traj = find_logs(args.gene_dir)
            base = (cand or traj).parent if (cand or traj) else args.gene_dir
            out = base / "query_diversity.png"
        plot_gene(args.gene_dir, out, num_init=args.num_init)
        return
    if args.run:
        out = args.output or (args.run / "analysis" / "query_diversity_by_generation.png")
        plot_run(args.run, out, args.run_root, args.num_init)
        return
    raise SystemExit("Pass --gene-dir or --run")


if __name__ == "__main__":
    main()
