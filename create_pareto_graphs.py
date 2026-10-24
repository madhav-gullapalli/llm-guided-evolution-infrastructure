import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = "./sota/Surrogate/results/"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS = ["Kendall_Tau", "Top_Individual_Found"]
KT_MAX = 1
MSE_MAX = 500


def load_data(directory, metric="top-individual", max_age_hours=72):
    """
    Load Kendall Tau and the selected second objective from result files.

    Coevolution files contain {final_kendall_tau},{best_accuracy}; legacy MSE
    runs contain {final_kendall_tau},{final_mse}. Select the metric to match
    the input directory. Any third column (legacy runtime) is ignored.
    """
    data = []
    cutoff = time.time() - max_age_hours * 3600 if max_age_hours is not None else None
    stale = 0
    for fname in os.listdir(directory):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(directory, fname)
        try:
            if cutoff is not None and os.path.getmtime(path) < cutoff:
                stale += 1
                continue
            with open(path, "r") as f:
                content = f.read().strip()
            if not content:
                continue
            values = [float(x) for x in content.split(",")]
            if len(values) < len(METRICS):
                print(f"Skipping {fname}: expected at least {len(METRICS)} values")
                continue
            if not np.isfinite(values[:len(METRICS)]).all():
                print(f"Skipping {fname}: non-finite objectives")
                continue
            if metric != "mse" and not 0 <= values[1] <= 100:
                print(f"Skipping {fname}: accuracy outside 0–100; possibly legacy MSE")
                continue
            data.append(values[:len(METRICS)])
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    print(f"Loaded {len(data)} results; skipped {stale} results older than {max_age_hours} hours.")
    return pd.DataFrame(data, columns=["Kendall_Tau", "MSE" if metric == "mse" else "Top_Individual_Found"])


def get_pareto_front_2d(x_values, y_values, x_max=True, y_max=False):
    pts = np.vstack((x_values, y_values)).T
    n_points = pts.shape[0]
    is_optimal = np.zeros(n_points, dtype=bool)

    x_sort = -pts[:, 0] if x_max else pts[:, 0]
    y_sort = -pts[:, 1] if y_max else pts[:, 1]
    idx = np.lexsort((y_sort, x_sort))
    sorted_pts = pts[idx]

    current_best_y = -np.inf if y_max else np.inf
    for i, (_, y) in enumerate(sorted_pts):
        if y_max:
            if y > current_best_y:
                is_optimal[idx[i]] = True
                current_best_y = y
        elif y < current_best_y:
            is_optimal[idx[i]] = True
            current_best_y = y

    return is_optimal


def plot_pareto(df, output_dir, trivial_points=None, metric="top-individual"):
    x_col = "Kendall_Tau"
    is_mse = metric == "mse"
    y_col = "MSE" if is_mse else "Top_Individual_Found"

    plot_df = df[[x_col, y_col]].copy()
    if trivial_points:
        triv_df = pd.DataFrame(trivial_points, columns=[x_col, y_col])
        combined_df = pd.concat([plot_df, triv_df], ignore_index=True)
    else:
        combined_df = plot_df

    mask = get_pareto_front_2d(
        combined_df[x_col].values,
        combined_df[y_col].values,
        x_max=True,
        y_max=not is_mse,
    )
    pareto_df = combined_df[mask].sort_values(by=x_col)

    plt.figure(figsize=(10, 6))
    plt.xlim(0, KT_MAX)
    if is_mse:
        plt.ylim(0, MSE_MAX)

    plt.scatter(
        df[x_col],
        df[y_col],
        c="royalblue",
        alpha=0.25,
        s=30,
        label="All Experimental Models",
        zorder=1,
    )
    plt.step(
        pareto_df[x_col],
        pareto_df[y_col],
        where="pre",
        color="crimson",
        lw=2.5,
        label="Pareto Front",
        zorder=3,
    )
    plt.scatter(
        pareto_df[x_col],
        pareto_df[y_col],
        color="crimson",
        edgecolor="black",
        s=70,
        label="Pareto Optimal Points",
        zorder=4,
    )

    if trivial_points:
        t_x, t_y = zip(*trivial_points)
        plt.scatter(
            t_x,
            t_y,
            color="black",
            marker="x",
            s=120,
            linewidths=2,
            label="Trivial Anchors",
            zorder=5,
        )

    plt.xlabel("Kendall Tau (Higher is Better)")
    plt.ylabel("Mean Squared Error (Lower is Better)" if is_mse else
               "Top Individual Found — Validation Accuracy (Higher is Better)")
    plt.title("Pareto Efficiency: Kendall Tau vs. " + ("MSE" if is_mse else "Top Individual Found"))
    plt.legend(loc="best", frameon=True, shadow=True)
    plt.grid(True, linestyle=":", alpha=0.6)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "pareto_kt_vs_mse.png" if is_mse else
                             "pareto_kt_vs_top_individual.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph generated: {save_path}")


def plot_pareto_kt_mse(df, output_dir, trivial_points=None):
    """Plot legacy MSE results explicitly."""
    return plot_pareto(df, output_dir, trivial_points, metric="mse")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot surrogate Pareto objectives.")
    parser.add_argument("--metric", choices=["top-individual", "mse"], default="top-individual",
                        help="Second objective stored in the result files (default: top-individual).")
    parser.add_argument("--max-age-hours", type=float, default=72,
                        help="Maximum result file age in hours (default: 72).")
    args = parser.parse_args()
    if args.max_age_hours <= 0:
        parser.error("--max-age-hours must be positive")
    df_results = load_data(RESULTS_DIR, metric=args.metric, max_age_hours=args.max_age_hours)

    if df_results.empty:
        print("Data directory empty or inaccessible.")
    else:
        plot_pareto(
            df_results,
            OUTPUT_DIR,
            trivial_points=[(0, 0), (1, 200)] if args.metric == "mse" else None,
            metric=args.metric,
        )
        print("Pareto graph has been generated.")
