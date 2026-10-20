#!/usr/bin/env python3
"""Plot surrogate Kendall τ / MSE over BANANAS search steps.

Adapted from COLE ``analyze_surrogate_results.py`` for LLM-GE ``run_nb201/<gene>/`` layout.
Supports comparing multiple genes on one figure (each gene usually has one trial).

Example:
  python plot_surrogate_metrics.py \\
    --run-root ../run_nb201 \\
    --genes seed:seed xXxG6eiSkPigHllPGknB63fm57V:best \\
            xXxKQwdEO2RqgA5YJT4nfGbLzuo:bad_pca \\
    --output ../results/analysis_tools/surrogate_tau_mse.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_gene_specs(specs: Iterable[str]) -> list[tuple[str, str, Optional[str]]]:
    """Parse ``gene_id``, ``gene_id:label``, or ``gene_id:label@PredictorSubstr``."""
    out = []
    for spec in specs:
        predictor = None
        if "@" in spec:
            spec, predictor = spec.rsplit("@", 1)
            predictor = predictor.strip() or None
        if ":" in spec:
            gene_id, label = spec.split(":", 1)
        else:
            gene_id, label = spec, spec
        out.append((gene_id.strip(), label.strip(), predictor))
    return out


def find_metric_files(
    run_root: Path,
    gene_id: str,
    dataset: str,
    seed: Optional[int],
    predictor_substr: Optional[str],
) -> list[Path]:
    gene_dir = run_root / gene_id
    if not gene_dir.is_dir():
        return []
    paths = sorted(gene_dir.glob(f"nasbench201/{dataset}/**/surrogate_metrics_*.json"))
    if seed is not None:
        paths = [p for p in paths if f"/{seed}/" in str(p).replace("\\", "/") or p.parent.name == str(seed)]
    if predictor_substr:
        paths = [p for p in paths if predictor_substr in str(p)]
    return paths


def load_metrics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = json.loads(path.read_text())
    epochs = np.asarray(data.get("epochs", list(range(len(data["kendall_tau"])))), dtype=np.float64)
    tau = np.asarray(data["kendall_tau"], dtype=np.float64)
    mse = np.asarray(data.get("mse", []), dtype=np.float64)
    meta = {
        "experiment_id": data.get("experiment_id"),
        "base_predictor": data.get("base_predictor"),
        "gene_id": path.parts[-6] if len(path.parts) >= 6 else path.parent.name,
    }
    return epochs, tau, mse, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parents[1] / "run_nb201")
    parser.add_argument("--genes", nargs="+", required=True, help="gene_id or gene_id:label")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--predictor-substr",
        default="CustomMLP",
        help="Filter path by substring (default CustomMLP). Use '' to keep all.",
    )
    parser.add_argument("--k", type=int, default=10, help="Architectures evaluated per BANANAS epoch (for x-axis)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/analysis_tools/surrogate_tau_mse.png",
    )
    args = parser.parse_args()

    gene_specs = parse_gene_specs(args.genes)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(gene_specs), 1)))
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, (ax_tau, ax_mse) = plt.subplots(1, 2, figsize=(14, 5))
    summary: list[str] = []

    for i, (gene_id, label, per_gene_pred) in enumerate(gene_specs):
        pred_filter = per_gene_pred if per_gene_pred is not None else (args.predictor_substr or None)
        files = find_metric_files(
            args.run_root, gene_id, args.dataset, args.seed, pred_filter
        )
        if not files:
            print(f"[WARN] no surrogate_metrics for gene={gene_id}")
            continue

        tau_trials = []
        mse_trials = []
        epochs_ref = None
        for f in files:
            epochs, tau, mse, meta = load_metrics(f)
            epochs_ref = epochs
            tau_trials.append(tau)
            if len(mse):
                mse_trials.append(mse)
            print(f"Loaded {f} base={meta.get('base_predictor')} tau_final={tau[-1]:.4f}")

        # Pad to common length
        max_len = max(len(t) for t in tau_trials)
        tau_pad = np.vstack(
            [np.pad(t, (0, max_len - len(t)), constant_values=t[-1]) for t in tau_trials]
        )
        tau_mean = tau_pad.mean(axis=0)
        x = (epochs_ref if epochs_ref is not None else np.arange(max_len))
        if len(x) < max_len:
            x = np.arange(max_len)
        # Map epoch index -> cumulative architectures evaluated after init.
        # BANANAS logs one metric point per retrain (after num_init).
        x_queries = (x + 1) * args.k  # approximate; sufficient for relative comparison

        color = colors[i]
        marker = markers[i % len(markers)]
        ax_tau.plot(
            x_queries,
            tau_mean,
            label=f"{label} (n={len(tau_trials)})",
            color=color,
            marker=marker,
            markevery=max(1, len(x_queries) // 10),
            linewidth=2,
        )
        summary.append(f"{label}: final_tau={tau_mean[-1]:.4f} n_files={len(files)}")

        if mse_trials:
            mse_pad = np.vstack(
                [np.pad(m, (0, max_len - len(m)), constant_values=m[-1]) for m in mse_trials]
            )
            mse_mean = mse_pad.mean(axis=0)
            ax_mse.plot(
                x_queries,
                mse_mean,
                label=f"{label} (n={len(mse_trials)})",
                color=color,
                marker=marker,
                markevery=max(1, len(x_queries) // 10),
                linewidth=2,
            )
            summary.append(f"{label}: final_mse={mse_mean[-1]:.4f}")

    ax_tau.set_title(f"Surrogate Kendall τ ({args.dataset})")
    ax_tau.set_xlabel("Architectures evaluated (approx, k×epoch)")
    ax_tau.set_ylabel("Kendall τ")
    ax_tau.grid(True, linestyle="--", alpha=0.5)
    ax_tau.legend()

    ax_mse.set_title(f"Surrogate MSE ({args.dataset})")
    ax_mse.set_xlabel("Architectures evaluated (approx, k×epoch)")
    ax_mse.set_ylabel("MSE")
    ax_mse.grid(True, linestyle="--", alpha=0.5)
    ax_mse.legend()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    args.output.with_suffix(".txt").write_text("\n".join(summary) + "\n")
    print(f"Plot saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
