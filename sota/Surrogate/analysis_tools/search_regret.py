#!/usr/bin/env python3
"""NAS-Bench-style anytime validation regret helpers.

Immediate / anytime regret follows Ying et al. (NAS-Bench-101): at budget t,
regret(t) = f(A*) − max_{i≤t} validation accuracy. DE-NAS and BANANAS report
the same curve (or its flip: best error so far) averaged over seeds.

Near-opt hit time uses COLE's bar: first eval with accuracy ≥ 0.99 × optimum.
That is not the same as time_to_find_top_X_percent (percentile rank cut).
"""
from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SURROGATE_DIR = REPO_ROOT / "sota" / "Surrogate"
RUN_NB201 = SURROGATE_DIR / "run_nb201"
DATA_DIR = SURROGATE_DIR / "naslib" / "data"

# dataset → (pickle name, accuracy field path under each cell)
NB201_DATA = {
    "cifar100": ("nb201_cifar100_full_training.pickle", "cifar100"),
    "cifar10": ("nb201_cifar10_full_training.pickle", "cifar10-valid"),
    "cifar10-valid": ("nb201_cifar10_full_training.pickle", "cifar10-valid"),
}

OPTIMUM_FRACTION = 0.99


@lru_cache(maxsize=8)
def space_optimum(dataset: str = "cifar100") -> Optional[float]:
    """Best final validation accuracy in the local NAS-Bench-201 pickle."""
    entry = NB201_DATA.get(dataset)
    if entry is None:
        return None
    path = DATA_DIR / entry[0]
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        data = pickle.load(handle)
    key = entry[1]
    return max(cell[key]["eval_acc1es"][-1] for cell in data.values())


@lru_cache(maxsize=8)
def top_percentile_accuracy(dataset: str = "cifar100", percent: float = 1.0) -> Optional[float]:
    """Accuracy at the (100 − percent) percentile (top ``percent``% of the space)."""
    entry = NB201_DATA.get(dataset)
    if entry is None:
        return None
    path = DATA_DIR / entry[0]
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        data = pickle.load(handle)
    key = entry[1]
    accs = [cell[key]["eval_acc1es"][-1] for cell in data.values()]
    return float(np.percentile(accs, 100.0 - percent))


def near_opt_threshold(optimum: float, fraction: float = OPTIMUM_FRACTION) -> float:
    return fraction * optimum


def best_so_far_curve(valid_acc) -> np.ndarray:
    return np.maximum.accumulate(np.asarray(valid_acc, dtype=float))


def regret_curve(valid_acc, optimum: float) -> np.ndarray:
    return optimum - best_so_far_curve(valid_acc)


def evals_to_threshold(valid_acc, threshold: float, tol: float = 1e-3) -> Optional[int]:
    """1-based index of the first evaluation that reaches ``threshold``."""
    for index, accuracy in enumerate(np.asarray(valid_acc, dtype=float), 1):
        if accuracy >= threshold - tol:
            return index
    return None


def load_errors(path: Path) -> tuple[dict, list[float]]:
    config, curve = json.loads(path.read_text())
    return config, list(curve.get("valid_acc") or [])


def find_errors_json(match_or_gene: str, run_root: Path = RUN_NB201) -> Optional[Path]:
    """Locate ``errors.json`` for a coev match id or solo gene folder."""
    gene_dir = Path(run_root) / match_or_gene
    if not gene_dir.is_dir():
        return None
    paths = sorted(gene_dir.glob("nasbench201/*/**/errors.json"))
    return paths[0] if paths else None


def summarize_trajectory(
    valid_acc,
    optimum: float,
    *,
    near_fraction: float = OPTIMUM_FRACTION,
    top_percent: Optional[float] = None,
    top_threshold: Optional[float] = None,
    num_init: int = 0,
) -> dict:
    """Scalar search summary for one BANANAS / coev evaluation trajectory."""
    acc = np.asarray(valid_acc, dtype=float)
    if acc.size == 0:
        return {}
    best = float(acc.max())
    regrets = regret_curve(acc, optimum)
    near_bar = near_opt_threshold(optimum, near_fraction)
    row = {
        "n_evals": int(acc.size),
        "num_init": int(num_init),
        "best_accuracy": best,
        "final_regret": float(regrets[-1]),
        "regret_after_init": float(regrets[num_init - 1]) if num_init and acc.size >= num_init else None,
        "optimum": float(optimum),
        "near_opt_threshold": float(near_bar),
        "evals_to_within_1pct_opt": evals_to_threshold(acc, near_bar),
        "within_1pct_opt": bool(best >= near_bar - 1e-3),
        "found_optimum": bool(best >= optimum - 1e-3),
        "regret_curve": regrets.tolist(),
        "best_so_far": best_so_far_curve(acc).tolist(),
    }
    if top_threshold is None and top_percent is not None:
        # Caller may pass dataset-specific percentile separately.
        top_threshold = None
    if top_threshold is not None:
        row["top_percent_threshold"] = float(top_threshold)
        row["evals_to_top_percent"] = evals_to_threshold(acc, top_threshold)
        row["reached_top_percent"] = bool(best >= top_threshold - 1e-3)
    if num_init and acc.size >= num_init:
        row["random_start_best_acc"] = float(acc[:num_init].max())
    return row


def summarize_errors_file(path: Path, dataset: str = "cifar100") -> dict:
    config, valid = load_errors(path)
    ds = config.get("dataset") or dataset
    optimum = space_optimum(ds)
    if optimum is None:
        return {"path": str(path), "error": f"no optimum for dataset {ds}"}
    num_init = int((config.get("search") or {}).get("num_init") or 0)
    top_thr = top_percentile_accuracy(ds, 1.0)
    row = summarize_trajectory(
        valid, optimum, num_init=num_init, top_threshold=top_thr)
    row["path"] = str(path)
    row["dataset"] = ds
    row["seed"] = (config.get("search") or {}).get("seed", config.get("seed"))
    return row


def collect_match_search(
    match_ids: list[str],
    *,
    run_root: Path = RUN_NB201,
    dataset: str = "cifar100",
) -> list[dict]:
    """Load search summaries for every match/gene that has an errors.json."""
    rows = []
    for match_id in match_ids:
        path = find_errors_json(match_id, run_root)
        if path is None:
            continue
        row = summarize_errors_file(path, dataset=dataset)
        row["match_id"] = match_id
        rows.append(row)
    return rows


def aggregate_search_rows(rows: list[dict]) -> dict:
    """Median / best search stats over many match trajectories."""
    if not rows:
        return {}
    finals = [r["final_regret"] for r in rows if "final_regret" in r]
    after_init = [r["regret_after_init"] for r in rows if r.get("regret_after_init") is not None]
    within = [r for r in rows if r.get("within_1pct_opt")]
    evals = [r["evals_to_within_1pct_opt"] for r in rows if r.get("evals_to_within_1pct_opt") is not None]
    top_evals = [r["evals_to_top_percent"] for r in rows if r.get("evals_to_top_percent") is not None]
    best_acc = [r["best_accuracy"] for r in rows if "best_accuracy" in r]
    return {
        "n_trajectories": len(rows),
        "best_accuracy": max(best_acc) if best_acc else None,
        "best_final_regret": min(finals) if finals else None,
        "median_final_regret": float(np.median(finals)) if finals else None,
        "median_regret_after_init": float(np.median(after_init)) if after_init else None,
        "frac_within_1pct_opt": len(within) / len(rows),
        "median_evals_to_within_1pct_opt": float(np.median(evals)) if evals else None,
        "median_evals_to_top_1pct": float(np.median(top_evals)) if top_evals else None,
        "n_hit_within_1pct": len(within),
        "n_hit_top_1pct_traj": sum(1 for r in rows if r.get("reached_top_percent")),
    }


def median_regret_curve(rows: list[dict]) -> Optional[list[float]]:
    """Pointwise median of per-match regret curves (truncated to shortest length)."""
    curves = [r["regret_curve"] for r in rows if r.get("regret_curve")]
    if not curves:
        return None
    length = min(len(c) for c in curves)
    stacked = np.asarray([c[:length] for c in curves], dtype=float)
    return np.median(stacked, axis=0).tolist()
