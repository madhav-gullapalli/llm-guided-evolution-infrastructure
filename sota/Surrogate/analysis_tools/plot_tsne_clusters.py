#!/usr/bin/env python3
"""Fixed-projection t-SNE of NB201 embeddings, colored by gene residuals.

Rebuild of cole-dev ``visualize_data.py`` for LLM-GE:
  - embeddings from compact cache ``data/embedding_cache/nb201_*.npz``
  - residuals / predictions from ``run_nb201/<gene>/.../analysis/predictions_*.csv``

Also runs DBSCAN on the 2D projection and writes reverse cluster maps +
high/low residual cluster feature comparison (op-index based).

Example:
  python plot_tsne_clusters.py \\
    --gene xXxG6eiSkPigHllPGknB63fm57V \\
    --output-dir ../results/analysis_tools/tsne_best
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# NAS-Bench-201 op-index string conversion (mirrors naslib conversions).
EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
OP_NAMES_NB201 = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]


def convert_op_indices_to_str(op_indices) -> str:
    edge_op_dict = {edge: OP_NAMES_NB201[int(op)] for edge, op in zip(EDGE_LIST, op_indices)}
    op_edge_list = [
        "{}~{}".format(edge_op_dict[(i, j)], i - 1)
        for i, j in sorted(edge_op_dict, key=lambda x: x[1])
    ]
    return "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)


def parse_op_indices(val) -> Optional[list[int]]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, list):
        return [int(x) for x in val]
    s = str(val).strip()
    if not s:
        return None
    try:
        parsed = ast.literal_eval(s)
        return [int(x) for x in parsed]
    except Exception:
        nums = re.findall(r"-?\d+", s)
        return [int(x) for x in nums] if nums else None


def load_embedding_cache(cache_path: Path) -> tuple[np.ndarray, list[str]]:
    data = np.load(cache_path, allow_pickle=True)
    arch_strings = [str(x) for x in data["arch_strings"].tolist()]
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    return embeddings, arch_strings


def load_prediction_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    if "inner_epoch" in df.columns and df["inner_epoch"].notna().any():
        last = df["inner_epoch"].max()
        df = df[df["inner_epoch"] == last].copy()
    df["op_indices_parsed"] = df["op_indices"].apply(parse_op_indices)
    df = df[df["op_indices_parsed"].notna()].copy()
    df["arch_string"] = df["op_indices_parsed"].apply(convert_op_indices_to_str)
    if "residual" not in df.columns:
        df["residual"] = df["predicted_accuracy"] - df["true_accuracy"]
    df["abs_residual"] = df["residual"].abs()
    return df


def find_prediction_csv(run_root: Path, gene_id: str, dataset: str) -> Path:
    matches = sorted(
        (run_root / gene_id).glob(f"nasbench201/{dataset}/**/analysis/predictions_*.csv")
    )
    if not matches:
        raise FileNotFoundError(f"No predictions CSV under {run_root / gene_id}")
    return matches[-1]


def draw_scatter(
    xy: np.ndarray,
    values: np.ndarray,
    out_path: Path,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    discrete: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    if discrete:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap="tab20", s=12, alpha=0.7)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, s=12, alpha=0.55)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


OP_FEATURE_NAMES = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]


def op_count_features(op_indices: list[int]) -> dict:
    feats = {name: 0 for name in OP_FEATURE_NAMES}
    for op in op_indices:
        feats[OP_FEATURE_NAMES[int(op)]] += 1
    feats["n_ops"] = len(op_indices)
    return feats


def compare_clusters(
    df: pd.DataFrame,
    cluster_col: str,
    out_path: Path,
    n_high: int = 5,
    n_low: int = 5,
) -> None:
    """Compare op-mix of high- vs low-mean-|residual| clusters."""
    stats = (
        df.groupby(cluster_col)["abs_residual"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_abs_residual", "count": "n"})
    )
    stats = stats[stats.index >= 0]  # drop DBSCAN noise (-1)
    if stats.empty:
        out_path.write_text("No non-noise clusters.\n")
        return

    high = stats.sort_values("mean_abs_residual", ascending=False).head(n_high)
    remaining = stats.drop(index=high.index, errors="ignore")
    low = remaining.sort_values("mean_abs_residual", ascending=True).head(n_low)

    lines = ["# Cluster residual summary", stats.sort_values("mean_abs_residual", ascending=False).to_string(), ""]

    def group_feats(ids):
        rows = df[df[cluster_col].isin(ids)]
        feats = [op_count_features(ops) for ops in rows["op_indices_parsed"]]
        if not feats:
            return {}
        keys = feats[0].keys()
        return {k: float(np.mean([f[k] for f in feats])) for k in keys}

    high_ids = list(high.index)
    low_ids = list(low.index)
    high_f = group_feats(high_ids)
    low_f = group_feats(low_ids)

    lines.append(f"High-|residual| clusters: {high_ids}")
    lines.append(f"Low-|residual| clusters: {low_ids}")
    lines.append("")
    lines.append(f"{'feature':<18} {'high_mean':>12} {'low_mean':>12}")
    for k in high_f:
        lines.append(f"{k:<18} {high_f[k]:12.3f} {low_f.get(k, float('nan')):12.3f}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {out_path}")


def write_reverse_mapping(df: pd.DataFrame, cluster_col: str, out_path: Path) -> None:
    with open(out_path, "w") as f:
        for cid, group in df.groupby(cluster_col):
            f.write(f"Cluster {int(cid)}:\n")
            for arch in group["arch_string"].tolist():
                f.write(f"  {arch}\n")
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", required=True, help="Gene id under run_nb201/")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "run_nb201",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data/embedding_cache/nb201_0d95b1493a31ec76.npz",
    )
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/analysis_tools/tsne_<gene>",
    )
    parser.add_argument("--max-points", type=int, default=4000, help="Subsample for faster projection")
    parser.add_argument(
        "--method",
        choices=["pca", "tsne"],
        default="pca",
        help="2D projection method (pca is fast/default; tsne is slower)",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--dbscan-eps", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.output_dir or (
        Path(__file__).resolve().parents[1] / "results/analysis_tools" / f"tsne_{args.gene}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cache {args.cache}", flush=True)
    embeddings, arch_strings = load_embedding_cache(args.cache)
    arch_to_idx = {a: i for i, a in enumerate(arch_strings)}

    csv_path = find_prediction_csv(args.run_root, args.gene, args.dataset)
    print(f"Loading predictions {csv_path}", flush=True)
    pred = load_prediction_table(csv_path)

    idxs = []
    keep_rows = []
    for _, row in pred.iterrows():
        i = arch_to_idx.get(row["arch_string"])
        if i is not None:
            idxs.append(i)
            keep_rows.append(row)
    if not idxs:
        raise RuntimeError("No prediction rows matched embedding cache arch_strings")

    joined = pd.DataFrame(keep_rows).reset_index(drop=True)
    X = embeddings[np.asarray(idxs)]
    # Free full cache from memory ASAP
    del embeddings
    print(f"Joined {len(joined)} / {len(pred)} prediction rows to cache ({X.shape[1]}-D)", flush=True)

    rng = np.random.default_rng(args.seed)
    if len(joined) > args.max_points:
        sel = rng.choice(len(joined), size=args.max_points, replace=False)
        joined = joined.iloc[sel].reset_index(drop=True)
        X = X[sel]
        print(f"Subsampled to {len(joined)} points for t-SNE", flush=True)

    X_scaled = StandardScaler().fit_transform(X)
    from sklearn.decomposition import PCA

    if args.method == "pca":
        print("PCA to 2-D...", flush=True)
        xy = PCA(n_components=2, random_state=args.seed).fit_transform(X_scaled)
        axis_name = "PCA"
    else:
        pca_dim = min(50, X_scaled.shape[0] - 1, X_scaled.shape[1])
        print(f"PCA to {pca_dim}-D then t-SNE (this can take several minutes)...", flush=True)
        X_pca = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(X_scaled)
        xy = TSNE(
            n_components=2,
            random_state=args.seed,
            perplexity=min(args.perplexity, max(5, (len(joined) - 1) / 3)),
            init="pca",
            learning_rate="auto",
        ).fit_transform(X_pca)
        axis_name = "t-SNE"

    joined["tsne_x"] = xy[:, 0]
    joined["tsne_y"] = xy[:, 1]

    # Persist coordinates for reuse
    coords_path = out_dir / "tsne_coords.csv"
    joined[
        [
            "arch_string",
            "true_accuracy",
            "predicted_accuracy",
            "residual",
            "abs_residual",
            "tsne_x",
            "tsne_y",
        ]
    ].to_csv(coords_path, index=False)
    print(f"Saved {coords_path}")

    draw_scatter(
        xy,
        joined["true_accuracy"].to_numpy(),
        out_dir / f"{args.method}_true_accuracy.png",
        f"{args.gene}: true accuracy ({axis_name})",
        "true accuracy",
        cmap="viridis",
    )
    draw_scatter(
        xy,
        joined["predicted_accuracy"].to_numpy(),
        out_dir / f"{args.method}_predicted_accuracy.png",
        f"{args.gene}: predicted accuracy ({axis_name})",
        "predicted accuracy",
        cmap="viridis",
    )
    draw_scatter(
        xy,
        joined["abs_residual"].to_numpy(),
        out_dir / f"{args.method}_abs_residual.png",
        f"{args.gene}: |residual| ({axis_name})",
        "|predicted − true|",
        cmap="Reds",
    )
    draw_scatter(
        xy,
        joined["residual"].to_numpy(),
        out_dir / f"{args.method}_residual.png",
        f"{args.gene}: residual (pred − true) ({axis_name})",
        "residual",
        cmap="coolwarm",
    )

    # Scale DBSCAN eps to projection spread if user left default
    span = float(np.ptp(xy[:, 0]))
    eps = args.dbscan_eps
    if args.method == "pca" and args.dbscan_eps == 3.0:
        eps = max(0.5, 0.08 * span)
    print(f"DBSCAN eps={eps}", flush=True)
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(xy)
    joined["cluster"] = labels
    n_clusters = len(set(labels) - {-1})
    print(f"Found {n_clusters} clusters (+ noise={(labels == -1).sum()})", flush=True)

    draw_scatter(
        xy,
        labels.astype(float),
        out_dir / f"{args.method}_clusters.png",
        f"{args.gene}: DBSCAN clusters (n={n_clusters}) ({axis_name})",
        "cluster id",
        discrete=True,
    )

    write_reverse_mapping(joined, "cluster", out_dir / "reverse_mapping.txt")
    compare_clusters(joined, "cluster", out_dir / "cluster_residual_comparison.txt")

    # Also dump per-cluster mean residual table
    cluster_table = (
        joined.groupby("cluster")
        .agg(
            n=("arch_string", "count"),
            mean_abs_residual=("abs_residual", "mean"),
            mean_true=("true_accuracy", "mean"),
            mean_pred=("predicted_accuracy", "mean"),
        )
        .sort_values("mean_abs_residual", ascending=False)
    )
    cluster_table.to_csv(out_dir / "cluster_stats.csv")
    print(f"Saved {out_dir / 'cluster_stats.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
