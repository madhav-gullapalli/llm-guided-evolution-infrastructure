#!/usr/bin/env python3
"""Write one CSV row per gene for an LLM-GE islands run.

Columns:
  gene, island, generation, parent, mutation, mode, family, status,
                                       mode USE_missing_class: LLM dropped SurrogateUsageOptimizer,
                                       so eval.py ran stock BANANAS for that gene
  kendall_tau, mse                     from results/<gene>_results.txt (tau,precision,mse or legacy tau,mse);
                                       evolution fitness is (tau, precision_top10); mse is tie-break only
  top10_tau                            Kendall tau on only the truly best 10% of the held-out test
                                       architectures (can the surrogate tell good from great?)
  precision_top10                      Precision@top-10% (Ning et al. / GATES): fraction of the
                                       true top 10% that the surrogate also ranks in its top 10%
  param_count_tau, param_count_top10_tau, param_count_precision_top10
                                       the same metrics for a no-learning baseline that ranks
                                       architectures by parameter count; a surrogate below these
                                       learned nothing beyond "bigger is better"
  random_start_best_acc                best true accuracy among the random initial
                                       architectures (same for every gene with a fixed seed)
  guided_best_acc, guided_mean_acc     best / mean true accuracy of the surrogate-guided picks
  guided_evals_to_best                 evaluation number (random start included, e.g. 94 of 100)
                                       at which guided_best_acc was first reached
  evals_to_within_1pct_opt             COLE's metric: first evaluation (random start included) whose
                                       accuracy is within 1% of the best architecture in the whole
                                       NAS-Bench-201 space (blank if never reached)
  found_optimum                        the search evaluated the best architecture in the space
  beat_random_start                    guided picks found something better than the random start
  same_score_as_parent                 identical (tau, mse) to a parent -> mutation likely had no effect
  output_dir                           per-gene folder under run_nb201/

Example (from repo root):
  uv run python sota/Surrogate/analysis_tools/summarize_run.py --run surrogate_islands_run
Writes <run>/analysis/summary.csv by default.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import kendalltau

SURROGATE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SURROGATE_DIR.parents[1]

COLUMNS = [
    "gene", "island", "generation", "parent", "mutation", "mode", "family", "status",
    "kendall_tau", "mse", "top10_tau", "precision_top10",
    "param_count_tau", "param_count_top10_tau", "param_count_precision_top10",
    "random_start_best_acc", "guided_best_acc", "guided_mean_acc",
    "guided_evals_to_best", "evals_to_within_1pct_opt", "found_optimum", "beat_random_start",
    "same_score_as_parent", "output_dir",
]

TOP_FRACTION = 0.10
OPTIMUM_FRACTION = 0.99
# NASLib maps both validation and test accuracy to eval_acc1es, so the optimum must come from the same field.
NB201_DATA = {
    "cifar10": ("nb201_cifar10_full_training.pickle", "cifar10-valid"),
    "cifar100": ("nb201_cifar100_full_training.pickle", "cifar100"),
    "ImageNet16-120": ("nb201_ImageNet16_full_training.pickle", "ImageNet16-120"),
}
# NAS-Bench-201 op indices (naslib OP_NAMES): 2 = ReLUConvBN3x3, 3 = ReLUConvBN1x1.
CONV3X3, CONV1X1 = 2, 3

GEN_RE = re.compile(r"_gen_(\d+)\.pkl$")
SCRIPT_RE = re.compile(r"Bash script saved to titanic_test/\d+/(xXx[A-Za-z0-9]+)")
ISLAND_RE = re.compile(r"Population data saved as \S*/(island_[^/\s]+)/checkpoint_gen_")


def gen_number(path: Path) -> int:
    m = GEN_RE.search(path.name)
    return int(m.group(1)) if m else -1


def gene_from_token(token: str) -> str:
    """Ancestry entries are gene ids or 'P:<other_parent>-C:<child>' for crossover."""
    return token.split("-C:", 1)[1] if token.startswith("P:") and "-C:" in token else token


def parents_from_ancestry(gene: str, entry: dict) -> list[str]:
    genes = entry.get("GENES", [])
    if len(genes) < 2:
        return []
    parents = [gene_from_token(genes[-2])]
    last = genes[-1]
    if last.startswith("P:") and "-C:" in last:
        parents.append(last[2:].split("-C:", 1)[0])
    return [p for p in parents if p != gene]


def load_global_data(run_dir: Path) -> tuple[dict, dict, dict]:
    """Return (fitness by gene, ancestry by gene, first generation each gene was recorded)."""
    hist: dict = {}
    ancestry: dict = {}
    first_gen: dict = {}
    for path in sorted((run_dir / "global_data").glob("global_gen_*.pkl"), key=gen_number):
        with open(path, "rb") as f:
            data = pickle.load(f)
        gen = gen_number(path)
        for key in ("GLOBAL_DATA_HIST", "GLOBAL_DATA"):
            for gene, info in data.get(key, {}).items():
                hist[gene] = info
                first_gen.setdefault(gene, gen)
        for gene, entry in data.get("GLOBAL_DATA_ANCESTRY", {}).items():
            ancestry[gene] = entry
            first_gen.setdefault(gene, gen)
    return hist, ancestry, first_gen


def islands_from_checkpoints(run_dir: Path) -> dict:
    out: dict = {}
    for island_dir in sorted(run_dir.glob("island_*")):
        for path in sorted(island_dir.glob("checkpoint_gen_*.pkl"), key=gen_number):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:  # deap is needed to unpickle Individuals
                print(f"[WARN] could not read {path}: {e!r}")
                continue
            members = list(data.get("population", []))
            members += list(getattr(data.get("hof"), "items", []) or [])
            for ind in members:
                if len(ind):
                    out.setdefault(ind[0], island_dir.name)
    return out


def islands_from_logs(log_dir: Path, genes: set) -> dict:
    out: dict = {}
    if not log_dir.is_dir():
        return out
    for path in sorted(log_dir.glob("Report_islands-*.out")):
        text = path.read_text(errors="ignore")
        island = ISLAND_RE.search(text)
        if not island:
            continue
        for gene in SCRIPT_RE.findall(text):
            if gene in genes:
                out.setdefault(gene, island.group(1))
    return out


def read_results(results_dir: Path, gene: str) -> tuple[Optional[float], Optional[float]]:
    """Return (kendall_tau, mse) from results file.

    Supports ``tau,precision,mse`` (current fitness) and legacy ``tau,mse``.
    """
    path = results_dir / f"{gene}_results.txt"
    if not path.exists():
        return None, None
    try:
        parts = [float(x) for x in path.read_text().strip().split(",") if x.strip()]
    except ValueError:
        return None, None
    if len(parts) >= 3:
        tau, _precision, mse = parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        tau, mse = parts[0], parts[1]
    else:
        return None, None
    if not (math.isfinite(tau) and math.isfinite(mse)):
        return None, None
    return tau, mse


def find_output_dir(run_root: Path, gene: str, dataset: str, seed: int) -> Optional[Path]:
    candidates = [p for p in (run_root / gene).glob(f"nasbench201/{dataset}/*/{seed}") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "errors.json").stat().st_mtime if (p / "errors.json").exists() else 0)


def detect_mode(models_dir: Path, gene: str) -> str:
    path = models_dir / f"model_{gene}.py"
    if not path.exists():
        return ""
    return "USE" if "class SurrogateUsageOptimizer" in path.read_text(errors="ignore") else "SURROGATE"


@functools.lru_cache(maxsize=None)
def search_space_optimum(dataset: str) -> Optional[float]:
    """Best final accuracy over all NAS-Bench-201 cells, from the file NASLib queries during search."""
    entry = NB201_DATA.get(dataset)
    if entry is None:
        return None
    path = SURROGATE_DIR / "naslib" / "data" / entry[0]
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return max(cell[entry[1]]["eval_acc1es"][-1] for cell in data.values())


def search_metrics(out_dir: Optional[Path]) -> dict:
    """Accuracy metrics from errors.json (best-so-far over all evals) and trajectory (guided picks)."""
    row: dict = {}
    if out_dir is None:
        return row
    num_init = 0
    errors_path = out_dir / "errors.json"
    if errors_path.exists():
        config, curve = json.loads(errors_path.read_text())
        num_init = int(config.get("search", {}).get("num_init", 0))
        valid = curve.get("valid_acc", [])
        if valid and num_init:
            row["random_start_best_acc"] = max(valid[:num_init])
        row["_n_evals"] = len(valid)
        optimum = search_space_optimum(config.get("dataset", ""))
        if valid and optimum is not None:
            target = OPTIMUM_FRACTION * optimum
            hit = next((i for i, acc in enumerate(valid) if acc >= target - 1e-3), None)
            row["evals_to_within_1pct_opt"] = hit + 1 if hit is not None else None
            row["found_optimum"] = max(valid) >= optimum - 1e-3
    traj_files = sorted(out_dir.glob("trajectory_*.json"))
    if not traj_files:
        return row
    picks = [p for p in json.loads(traj_files[0].read_text()) if p.get("true_accuracy") is not None]
    if not picks:
        return row
    accs = [p["true_accuracy"] for p in picks]
    best = max(accs)
    best_pick = picks[accs.index(best)]
    row["guided_best_acc"] = best
    row["guided_mean_acc"] = sum(accs) / len(accs)
    row["guided_evals_to_best"] = best_pick["generation"] + 1
    rs_best = row.get("random_start_best_acc")
    if rs_best is not None:
        row["beat_random_start"] = best > rs_best
    return row


def param_count_proxy(op_indices: list[int]) -> int:
    """Orders NAS-Bench-201 cells the same way as their parameter count (a 3x3 conv has ~9x a 1x1's weights)."""
    return 9 * op_indices.count(CONV3X3) + op_indices.count(CONV1X1)


def _tau(x, y) -> Optional[float]:
    value = kendalltau(x, y)[0]
    return None if value is None or math.isnan(value) else float(value)


def _precision_at_top(true: np.ndarray, score: np.ndarray, fraction: float = TOP_FRACTION) -> float:
    """Precision@top-K%: |true-top ∩ predicted-top| / K (Ning et al. 2021, GATES)."""
    n = len(true)
    k = max(1, int(round(n * fraction)))
    true_top = set(np.argsort(-true, kind="stable")[:k])
    pred_top = set(np.argsort(-score, kind="stable")[:k])
    return len(true_top & pred_top) / k


def ranking_metrics(out_dir: Optional[Path]) -> dict:
    """Top-10% tau, Precision@top-10%, and the parameter-count baseline on the final retrain's test set."""
    if out_dir is None:
        return {}
    files = sorted((out_dir / "analysis").glob("predictions_*.csv"))
    if not files:
        return {}
    with open(files[0], newline="") as f:
        test = [r for r in csv.DictReader(f) if r["split"] == "test"]
    if not test:
        return {}
    last = max(int(r["inner_epoch"]) for r in test)
    test = [r for r in test if int(r["inner_epoch"]) == last]
    true = np.array([float(r["true_accuracy"]) for r in test])
    pred = np.array([float(r["predicted_accuracy"]) for r in test])
    proxy = np.array([param_count_proxy(json.loads(r["op_indices"])) for r in test])
    top = true >= np.quantile(true, 1 - TOP_FRACTION)
    return {
        "top10_tau": _tau(true[top], pred[top]),
        "precision_top10": _precision_at_top(true, pred),
        "param_count_tau": _tau(true, proxy),
        "param_count_top10_tau": _tau(true[top], proxy[top]),
        "param_count_precision_top10": _precision_at_top(true, proxy),
    }


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Run folder, e.g. surrogate_islands_run")
    parser.add_argument("--output", type=Path, default=None, help="Default: <run>/analysis/summary.csv")
    parser.add_argument("--results-dir", type=Path, default=SURROGATE_DIR / "results")
    parser.add_argument("--run-root", type=Path, default=SURROGATE_DIR / "run_nb201")
    parser.add_argument("--models-dir", type=Path, default=SURROGATE_DIR / "models" / "llmge_models")
    parser.add_argument("--island-logs", type=Path, default=REPO_ROOT / "run_job_outputs" / "islands")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--seed", type=int, default=242)
    args = parser.parse_args()

    run_dir = args.run if args.run.is_absolute() or args.run.exists() else REPO_ROOT / args.run
    if not (run_dir / "global_data").is_dir():
        raise SystemExit(f"No global_data/ in {run_dir}")

    hist, ancestry, first_gen = load_global_data(run_dir)
    genes = (set(hist) | set(ancestry)) - {"model"}

    island_of = islands_from_checkpoints(run_dir)
    missing = genes - set(island_of)
    if missing:
        for gene, island in islands_from_logs(args.island_logs, missing).items():
            island_of.setdefault(gene, island)

    scores = {g: read_results(args.results_dir, g) for g in genes}
    rows = []
    for gene in genes:
        entry = ancestry.get(gene, {"GENES": [gene], "MUTATE_TYPE": ["CREATED"]})
        parents = parents_from_ancestry(gene, entry)
        mutation = entry.get("MUTATE_TYPE", ["CREATED"])[-1]
        generation = 0 if mutation == "CREATED" else first_gen.get(gene, "")

        island = island_of.get(gene, "")
        for p in parents:
            if island:
                break
            island = island_of.get(p, "")

        out_dir = find_output_dir(args.run_root, gene, args.dataset, args.seed)
        tau, mse = scores[gene]
        metrics = search_metrics(out_dir)
        if tau is not None:
            status = "ok"
        elif out_dir is None:
            status = "failed_before_search"
        else:
            status = "failed_in_search"

        same_as_parent = ""
        if tau is not None and parents:
            same_as_parent = any(
                scores.get(p, (None, None)) == (tau, mse) or read_results(args.results_dir, p) == (tau, mse)
                for p in parents
            )

        rows.append({
            "gene": gene,
            "island": island.removeprefix("island_"),
            "generation": generation,
            "parent": ";".join(parents),
            "mutation": mutation,
            "mode": detect_mode(args.models_dir, gene),
            "family": out_dir.parent.name.replace("LLM_NB201_Predictor_", "").replace("_bananas", "") if out_dir else "",
            "status": status,
            "kendall_tau": tau,
            "mse": mse,
            **(ranking_metrics(out_dir) if tau is not None else {}),
            **{k: v for k, v in metrics.items() if not k.startswith("_")},
            "same_score_as_parent": same_as_parent,
            "output_dir": str(out_dir.relative_to(SURROGATE_DIR)) if out_dir else "",
        })

    # In a USE run, a gene without SurrogateUsageOptimizer is scored with stock BANANAS by eval.py.
    modes = [r["mode"] for r in rows if r["mode"]]
    if modes and modes.count("USE") > len(modes) / 2:
        for r in rows:
            if r["mode"] == "SURROGATE":
                r["mode"] = "USE_missing_class"

    rows.sort(key=lambda r: (r["status"] != "ok", -(r["kendall_tau"] or -math.inf), str(r["generation"]), r["gene"]))

    output = args.output or run_dir / "analysis" / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: fmt(row.get(c)) for c in COLUMNS})

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"{len(rows)} genes ({len(ok)} ok, {len(rows) - len(ok)} failed) -> {output}")
    if ok:
        best_tau = ok[0]
        best_acc = max(ok, key=lambda r: r.get("guided_best_acc") or -math.inf)
        print(f"  best tau:          {best_tau['gene']}  tau={best_tau['kendall_tau']:.4f}")
        if best_acc.get("guided_best_acc") is not None:
            print(f"  best guided acc:   {best_acc['gene']}  acc={best_acc['guided_best_acc']:.2f}")
        ranked = [r for r in ok if r.get("top10_tau") is not None]
        if ranked:
            best_top = max(ranked, key=lambda r: r["top10_tau"])
            best_prec = max(
                (r for r in ok if r.get("precision_top10") is not None),
                key=lambda r: r["precision_top10"],
                default=None,
            )
            base = ranked[0]
            print(f"  best top-10% tau:  {best_top['gene']}  top10_tau={best_top['top10_tau']:.4f}")
            if best_prec is not None:
                print(f"  best Precision@top-10%: {best_prec['gene']}  "
                      f"precision_top10={best_prec['precision_top10']:.4f}")
            print(f"  parameter-count baseline (seed {args.seed} test set): "
                  f"tau={base['param_count_tau']:.4f}  top10_tau={base['param_count_top10_tau']:.4f}  "
                  f"precision_top10={base['param_count_precision_top10']:.4f}")
        no_ops = sum(1 for r in ok if r["same_score_as_parent"] is True)
        print(f"  same score as parent: {no_ops}/{len(ok)}")


if __name__ == "__main__":
    main()
