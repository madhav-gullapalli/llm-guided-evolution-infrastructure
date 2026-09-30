"""Prediction-table and calibration plots for surrogate analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREDICTION_COLUMNS = [
    "gene_id",
    "parent_gene",
    "outer_generation",
    "inner_epoch",
    "seed",
    "trial",
    "split",
    "arch_id",
    "op_indices",
    "true_accuracy",
    "predicted_accuracy",
    "residual",
    "true_rank",
    "predicted_rank",
    "selected",
]


def _op_indices_from_arch(arch: Any) -> Optional[list]:
    if arch is None:
        return None
    if hasattr(arch, "get_op_indices"):
        try:
            return [int(x) for x in arch.get_op_indices()]
        except Exception:
            pass
    if getattr(arch, "op_indices", None) is not None:
        return [int(x) for x in arch.op_indices]
    return None


def _arch_id(op_indices: Optional[Sequence[int]], fallback: Any = None) -> str:
    if op_indices is not None:
        return "-".join(str(int(x)) for x in op_indices)
    if fallback is None:
        return ""
    return str(fallback)


def add_ranks_and_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Add residual and ranks. Rank 1 is the best (highest) accuracy."""
    out = df.copy()
    out["residual"] = out["predicted_accuracy"] - out["true_accuracy"]
    grouped = []
    group_cols = [c for c in ("gene_id", "seed", "trial", "inner_epoch", "split") if c in out.columns]
    for _, group in out.groupby(group_cols, dropna=False) if group_cols else [(None, out)]:
        g = group.copy()
        g["true_rank"] = g["true_accuracy"].rank(ascending=False, method="min").astype(int)
        g["predicted_rank"] = g["predicted_accuracy"].rank(ascending=False, method="min").astype(int)
        grouped.append(g)
    return pd.concat(grouped, ignore_index=True) if grouped else out


def records_from_test_set(
    arches: Sequence[Any],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    inner_epoch: int,
    gene_id: str = "seed",
    parent_gene: Optional[str] = None,
    outer_generation: Optional[int] = None,
    seed: Optional[int] = None,
    trial: Optional[int] = None,
) -> list[dict]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_pred.ndim == 2:
        y_pred = np.mean(y_pred, axis=0)
    y_pred = y_pred.reshape(-1)
    records = []
    for arch, true_acc, pred_acc in zip(arches, y_true, y_pred):
        op_indices = _op_indices_from_arch(arch)
        records.append(
            {
                "gene_id": gene_id,
                "parent_gene": parent_gene,
                "outer_generation": outer_generation,
                "inner_epoch": int(inner_epoch),
                "seed": seed,
                "trial": trial,
                "split": "test",
                "arch_id": _arch_id(op_indices),
                "op_indices": op_indices,
                "true_accuracy": float(true_acc),
                "predicted_accuracy": float(pred_acc),
                "selected": False,
            }
        )
    return records


def records_from_candidate_log(
    candidate_log: Iterable[Mapping[str, Any]],
    *,
    gene_id: str = "seed",
    parent_gene: Optional[str] = None,
    outer_generation: Optional[int] = None,
    seed: Optional[int] = None,
    trial: Optional[int] = None,
) -> list[dict]:
    records = []
    for row in candidate_log:
        op_indices = row.get("op_indices")
        true_acc = row.get("true_accuracy")
        pred_acc = row.get("predicted_accuracy")
        if true_acc is None or pred_acc is None:
            continue
        records.append(
            {
                "gene_id": gene_id,
                "parent_gene": parent_gene,
                "outer_generation": outer_generation,
                "inner_epoch": int(row.get("generation", row.get("inner_epoch", -1))),
                "seed": seed,
                "trial": trial,
                "split": "candidate",
                "arch_id": _arch_id(op_indices, row.get("arch_index")),
                "op_indices": op_indices,
                "true_accuracy": float(true_acc),
                "predicted_accuracy": float(pred_acc),
                "selected": bool(row.get("selected", False)),
            }
        )
    return records


def build_prediction_table(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(records))
    if df.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    for col in PREDICTION_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = add_ranks_and_residuals(df)
    return df[PREDICTION_COLUMNS]


def plot_predicted_vs_actual(df: pd.DataFrame, output_path: Path, title: str = "") -> Path:
    """Two-panel plot: predicted vs actual, and residual vs actual."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.dropna(subset=["true_accuracy", "predicted_accuracy"]).copy()
    if plot_df.empty:
        raise ValueError("No finite true/predicted accuracies to plot")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    y = plot_df["true_accuracy"].to_numpy()
    yhat = plot_df["predicted_accuracy"].to_numpy()
    residual = plot_df["residual"].to_numpy()

    lo = float(min(y.min(), yhat.min()))
    hi = float(max(y.max(), yhat.max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)

    axes[0].scatter(y, yhat, alpha=0.45, s=18, edgecolors="none")
    axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1, label="y = x")
    axes[0].set_xlabel("True accuracy")
    axes[0].set_ylabel("Predicted accuracy")
    axes[0].set_title("Predicted vs actual")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y, residual, alpha=0.45, s=18, edgecolors="none")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("True accuracy")
    axes[1].set_ylabel("Residual (predicted − true)")
    axes[1].set_title("Residual vs actual")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title or "Surrogate calibration")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_prediction_analysis(
    records: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    stem: str = "prediction_table",
    title: str = "",
) -> dict:
    """Write the per-architecture CSV and the calibration/residual plot."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_prediction_table(records)
    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)

    plot_df = df
    if "inner_epoch" in df.columns and df["inner_epoch"].notna().any():
        last_epoch = df["inner_epoch"].max()
        latest = df[df["inner_epoch"] == last_epoch]
        if not latest.empty:
            plot_df = latest
    if "split" in plot_df.columns and (plot_df["split"] == "test").any():
        plot_df = plot_df[plot_df["split"] == "test"]

    plot_path = out_dir / f"{stem}_predicted_vs_actual.png"
    if plot_df.empty:
        plot_df = df
    plot_predicted_vs_actual(plot_df, plot_path, title=title)
    return {"csv": str(csv_path), "plot": str(plot_path), "n_rows": int(len(df))}
