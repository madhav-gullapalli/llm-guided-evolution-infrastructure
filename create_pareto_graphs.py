import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = "./sota/Surrogate/results/"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS = ["Kendall_Tau", "MSE"]
KT_MAX = 1
MSE_MAX = 500


def load_data(directory):
    """
    Load surrogate result files as Kendall Tau and MSE pairs.

    New result files are written as {final_kendall_tau},{final_mse}. Older
    files may still include runtime as a third column; that value is ignored.
    """
    data = []
    for fname in os.listdir(directory):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(directory, fname)
        try:
            with open(path, "r") as f:
                content = f.read().strip()
            if not content:
                continue
            values = [float(x) for x in content.split(",")]
            if len(values) < len(METRICS):
                print(f"Skipping {fname}: expected at least {len(METRICS)} values")
                continue
            data.append(values[:len(METRICS)])
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    return pd.DataFrame(data, columns=METRICS)


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


def plot_pareto_kt_mse(df, output_dir, trivial_points=None):
    x_col = "Kendall_Tau"
    y_col = "MSE"

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
        y_max=False,
    )
    pareto_df = combined_df[mask].sort_values(by=x_col)

    plt.figure(figsize=(10, 6))
    plt.xlim(0, KT_MAX)
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
    plt.ylabel("Mean Squared Error (Lower is Better)")
    plt.title("Pareto Efficiency: Kendall Tau vs. MSE")
    plt.legend(loc="best", frameon=True, shadow=True)
    plt.grid(True, linestyle=":", alpha=0.6)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "pareto_kt_vs_mse.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph generated: {save_path}")


if __name__ == "__main__":
    df_results = load_data(RESULTS_DIR)

    if df_results.empty:
        print("Data directory empty or inaccessible.")
    else:
        plot_pareto_kt_mse(
            df_results,
            OUTPUT_DIR,
            trivial_points=[(0, 0), (1, 200)],
        )
        print("Pareto graph has been generated.")
