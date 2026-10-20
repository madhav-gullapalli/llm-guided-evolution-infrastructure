#!/usr/bin/env python3
"""Re-evaluate a run's top genes with other BANANAS search seeds to see whether their scores hold up.

Every LLM-GE evaluation uses search seed 242, so a gene's score can partly reflect luck with that
seed. This script evaluates copies of the top genes with extra seeds and summarizes the spread.

  submit   copy each gene to sota/Surrogate/models/seed_reruns/model_<gene>__seed<N>.py and sbatch
           one eval.py job per (gene, seed). The copy has its own gene id, so the evolution's
           results/<gene>_results.txt and run_nb201/<gene>/ are never overwritten.
  collect  gather seed 242 (the original evaluation) plus every finished re-run into
           <run>/analysis/seed_reruns.csv and print mean / std per gene.

Each job is one GPU for up to 30 minutes and counts toward the PACE GPU limit, so avoid submitting
while an evolution run is active.

Examples (from repo root):
  uv run python sota/Surrogate/analysis_tools/seed_rerun.py submit --run surrogate_islands_run --top 5 --seeds 1 2
  uv run python sota/Surrogate/analysis_tools/seed_rerun.py collect --run surrogate_islands_run
"""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import subprocess
from pathlib import Path

import yaml

from summarize_run import (
    REPO_ROOT,
    SURROGATE_DIR,
    find_output_dir,
    fmt,
    ranking_metrics,
    read_results,
    search_metrics,
)

ORIGINAL_SEED = 242
RERUN_DIR = SURROGATE_DIR / "models" / "seed_reruns"
METRICS = ["kendall_tau", "top10_tau", "guided_best_acc", "guided_mean_acc", "evals_to_within_1pct_opt"]
COLUMNS = ["gene", "seed", "status"] + METRICS + ["param_count_tau", "param_count_top10_tau", "output_dir"]


def resolve_run(run: Path) -> Path:
    return run if run.is_absolute() or run.exists() else REPO_ROOT / run


def rerun_id(gene: str, seed: int) -> str:
    return f"{gene}__seed{seed}"


def pick_genes(summary: Path, top: int) -> list[str]:
    """Top genes by tau, skipping genes whose (tau, mse) repeats one already picked."""
    picked, seen = [], set()
    with open(summary, newline="") as f:
        for r in csv.DictReader(f):
            score = (r["kendall_tau"], r["mse"])
            if r["status"] != "ok" or score in seen:
                continue
            seen.add(score)
            picked.append(r["gene"])
            if len(picked) == top:
                break
    return picked


def job_script(gene_id: str, seed: int) -> str:
    with open(REPO_ROOT / "slurm-config" / "slurm_config.yaml") as f:
        template = yaml.safe_load(f)["python_bash_script"]
    runline = (f"uv run python {SURROGATE_DIR / 'eval.py'} --model model_{gene_id} "
               f"--variant_dir {RERUN_DIR} --seed {seed}")
    return template.replace("--job-name=evaluateGene", "--job-name=seedRerun").replace("{}", runline, 1)


def submit(args: argparse.Namespace, run_dir: Path) -> None:
    genes = args.genes or pick_genes(run_dir / "analysis" / "summary.csv", args.top)
    if not genes:
        raise SystemExit("No genes to re-run; run summarize_run.py first or pass --genes")
    jobs_dir = run_dir / "analysis" / "seed_reruns"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    RERUN_DIR.mkdir(parents=True, exist_ok=True)
    log = jobs_dir / "jobs.csv"
    new_log = not log.exists()
    with open(log, "a", newline="") as f:
        writer = csv.writer(f)
        if new_log:
            writer.writerow(["gene", "seed", "job_id"])
        for gene in genes:
            source = args.models_dir / f"model_{gene}.py"
            if not source.exists():
                print(f"[skip] {gene}: {source} not found")
                continue
            for seed in args.seeds:
                gene_id = rerun_id(gene, seed)
                if read_results(args.results_dir, gene_id) != (None, None) and not args.force:
                    print(f"[skip] {gene_id}: already evaluated (use --force to redo)")
                    continue
                shutil.copyfile(source, RERUN_DIR / f"model_{gene_id}.py")
                script = jobs_dir / f"{gene_id}.sh"
                script.write_text(job_script(gene_id, seed))
                if args.dry_run:
                    print(f"[dry-run] sbatch {script}")
                    continue
                result = subprocess.run(["sbatch", str(script)], cwd=REPO_ROOT, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"[fail] {gene_id}: {result.stderr.strip()}")
                    continue
                job_id = result.stdout.strip().split()[-1]
                writer.writerow([gene, seed, job_id])
                print(f"[submitted] {gene_id} -> job {job_id}")


def evaluation_row(gene: str, seed: int, gene_id: str, args: argparse.Namespace) -> dict:
    tau, mse = read_results(args.results_dir, gene_id)
    out_dir = find_output_dir(args.run_root, gene_id, args.dataset, seed)
    row = {"gene": gene, "seed": seed, "kendall_tau": tau}
    if tau is not None:
        row.update(ranking_metrics(out_dir))
        row.update({k: v for k, v in search_metrics(out_dir).items() if k in METRICS})
        row["status"] = "ok"
    else:
        row["status"] = "failed" if out_dir else "pending_or_failed"
    row["output_dir"] = str(out_dir.relative_to(SURROGATE_DIR)) if out_dir else ""
    return row


def spread(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def collect(args: argparse.Namespace, run_dir: Path) -> None:
    log = run_dir / "analysis" / "seed_reruns" / "jobs.csv"
    if not log.exists():
        raise SystemExit(f"{log} not found; run the submit step first")
    with open(log, newline="") as f:
        pairs = sorted({(r["gene"], int(r["seed"])) for r in csv.DictReader(f)})
    genes = list(dict.fromkeys(g for g, _ in pairs))

    rows = []
    for gene in genes:
        rows.append(evaluation_row(gene, ORIGINAL_SEED, gene, args))
        rows += [evaluation_row(gene, seed, rerun_id(gene, seed), args) for g, seed in pairs if g == gene]

    output = run_dir / "analysis" / "seed_reruns.csv"
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: fmt(row.get(c)) for c in COLUMNS})
    print(f"{len(rows)} evaluations ({sum(r['status'] == 'ok' for r in rows)} ok) -> {output}\n")

    print(f"{'gene':30s} {'seeds ok':>8s}  {'tau (seed 242)':>14s}  " + "  ".join(f"{m:>17s}" for m in METRICS))
    print("(evals_to_within_1pct_opt is averaged over the seeds that reached the target; the count is in parentheses)")
    for gene in genes:
        ok = [r for r in rows if r["gene"] == gene and r["status"] == "ok"]
        original = next((r["kendall_tau"] for r in ok if r["seed"] == ORIGINAL_SEED), None)
        cells = [spread([r[m] for r in ok if r.get(m) is not None]) for m in METRICS]
        reached = sum(r.get("evals_to_within_1pct_opt") is not None for r in ok)
        cells[-1] += f" ({reached}/{len(ok)} seeds)"
        print(f"{gene:30s} {len(ok):>8d}  {fmt(original):>14s}  " + "  ".join(f"{c:>17s}" for c in cells))

    baselines = {}
    for r in rows:
        if r["status"] == "ok" and r.get("param_count_tau") is not None:
            baselines[r["seed"]] = (r["param_count_tau"], r["param_count_top10_tau"])
    for seed, (tau, top) in sorted(baselines.items()):
        print(f"parameter-count baseline, seed {seed}: tau={tau:.3f}  top10_tau={top:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["submit", "collect"])
    parser.add_argument("--run", type=Path, required=True, help="Run folder, e.g. surrogate_islands_run")
    parser.add_argument("--top", type=int, default=5, help="submit: number of top genes by tau (distinct scores)")
    parser.add_argument("--genes", nargs="*", default=None, help="submit: explicit gene ids instead of --top")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2], help="submit: extra search seeds")
    parser.add_argument("--dry-run", action="store_true", help="submit: write job scripts without sbatch")
    parser.add_argument("--force", action="store_true", help="submit: re-run seeds that already have results")
    parser.add_argument("--results-dir", type=Path, default=SURROGATE_DIR / "results")
    parser.add_argument("--run-root", type=Path, default=SURROGATE_DIR / "run_nb201")
    parser.add_argument("--models-dir", type=Path, default=SURROGATE_DIR / "models" / "llmge_models")
    parser.add_argument("--dataset", default="cifar100")
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    if args.command == "submit":
        submit(args, run_dir)
    else:
        collect(args, run_dir)


if __name__ == "__main__":
    main()
