#!/usr/bin/env python3
"""Plot BANANAS search potency: best-found accuracy vs #architectures evaluated.

Adapted from COLE ``results_paper.py`` for LLM-GE gene folders under ``run_nb201/``.

Example:
  python plot_search_trajectory.py \\
    --run-root ../run_nb201 \\
    --genes seed:seed_mlp xXxG6eiSkPigHllPGknB63fm57V:best \\
            xXxKQwdEO2RqgA5YJT4nfGbLzuo:bad_pca \\
    --output ../results/analysis_tools/search_trajectory.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


OPTIMAL = {
    ("nasbench201", "cifar100"): 73.49,
    ("nasbench201", "cifar10"): 91.61,
    ("nasbench201", "ImageNet16-120"): 47.31,
}

YLIMS = {
    ("nasbench201", "cifar100"): (62.0, 74.0),
    ("nasbench201", "cifar10"): (84.0, 92.0),
    ("nasbench201", "ImageNet16-120"): (44.0, 48.0),
}


def parse_gene_specs(specs: Iterable[str]) -> list[tuple[str, str]]:
    """Parse ``gene_id`` or ``gene_id:label`` entries."""
    out = []
    for spec in specs:
        if ":" in spec:
            gene_id, label = spec.split(":", 1)
        else:
            gene_id, label = spec, spec
        out.append((gene_id.strip(), label.strip()))
    return out


def find_errors_json(
    run_root: Path,
    gene_id: str,
    dataset: str,
    seed: Optional[int],
    predictor_substr: Optional[str],
) -> list[Path]:
    gene_dir = run_root / gene_id
    if not gene_dir.is_dir():
        return []
    pattern = f"nasbench201/{dataset}/**/errors.json"
    paths = sorted(gene_dir.glob(pattern))
    if seed is not None:
        paths = [p for p in paths if p.parent.name == str(seed)]
    if predictor_substr:
        paths = [p for p in paths if predictor_substr in str(p)]
    return paths


def load_trajectory(errors_path: Path, metric: str = "valid_acc") -> np.ndarray:
    with open(errors_path) as f:
        data = json.load(f)
    results = data[1]
    y = np.asarray(results[metric], dtype=np.float64)
    return np.maximum.accumulate(y)


def aggregate_curves(ys: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(len(y) for y in ys)
    padded = []
    for y in ys:
        if len(y) < max_len:
            pad = np.full(max_len - len(y), y[-1], dtype=np.float64)
            padded.append(np.concatenate([y, pad]))
        else:
            padded.append(y[:max_len])
    arr = np.vstack(padded)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    n = arr.shape[0]
    sem = std / np.sqrt(n)
    ci_mult = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
    ci = ci_mult * sem
    return mean, std, ci, arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parents[1] / "run_nb201")
    parser.add_argument(
        "--genes",
        nargs="+",
        required=True,
        help="gene_id or gene_id:label (compare multiple genes on one plot)",
    )
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--space", default="nasbench201")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed folder filter (e.g. 242)")
    parser.add_argument(
        "--predictor-substr",
        default="CustomMLP",
        help="Filter path by substring (default CustomMLP). Use '' to keep all.",
    )
    parser.add_argument("--metric", default="valid_acc", choices=["valid_acc", "test_acc"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/analysis_tools/search_trajectory.png",
    )
    args = parser.parse_args()

    gene_specs = parse_gene_specs(args.genes)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(gene_specs), 1)))
    markers = ["s", "^", "o", "D", "v", "P", "X"]

    fig, ax = plt.subplots(figsize=(10, 6))
    stats_lines: list[str] = []
    optimal = OPTIMAL.get((args.space, args.dataset))
    ylims = YLIMS.get((args.space, args.dataset))

    for i, (gene_id, label) in enumerate(gene_specs):
        pred_filter = args.predictor_substr or None
        paths = find_errors_json(args.run_root, gene_id, args.dataset, args.seed, pred_filter)
        if not paths:
            print(f"[WARN] no errors.json for gene={gene_id}")
            continue
        curves = [load_trajectory(p, args.metric) for p in paths]
        mean, std, ci, raw = aggregate_curves(curves)
        x = np.arange(len(mean))
        color = colors[i]
        marker = markers[i % len(markers)]
        ax.plot(
            x,
            mean,
            label=f"{label} (n={len(curves)})",
            color=color,
            marker=marker,
            markersize=5,
            markevery=max(1, len(x) // 20),
            linewidth=2,
            alpha=0.9,
        )
        if len(curves) > 1:
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.2)

        final = mean[-1]
        stats_lines.append(f"{label}: final_best={final:.3f}%  files={len(paths)}")
        if optimal is not None:
            for pct in (0.98, 0.99, 0.995):
                thr = optimal * pct
                hit = np.where(mean >= thr)[0]
                msg = f"{hit[0]}" if len(hit) else "never"
                stats_lines.append(f"  iters to {pct*100:.1f}% of optimal ({thr:.2f}%): {msg}")
        print(f"Loaded {len(curves)} trajectory(ies) for {label} from {paths[0].parent}")

    if optimal is not None:
        ax.axhline(
            optimal,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Optimal Architecture ({optimal}%)",
        )
    if ylims is not None:
        ax.set_ylim(*ylims)

    ax.set_title(f"NAS Trajectory: {args.dataset.upper()} ({args.space})")
    ax.set_xlabel("Number of Architectures Evaluated")
    ax.set_ylabel(f"Best-so-far {args.metric.replace('_', ' ')} (%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    txt_path = args.output.with_suffix(".txt")
    txt_path.write_text("\n".join(stats_lines) + "\n")
    print(f"Plot saved to {args.output}")
    print(f"Stats saved to {txt_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
