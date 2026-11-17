#!/usr/bin/env python3
"""Summarize independent-score / Pareto stats from a coevolution run history.

Reads ``<run>/history/generation_*_attempt_*.json`` written by ``run_coevolution.py``
(each attempt stores ``rank_statistics``: per-population passing counts and rank-0
counts). Prints a compact table proving the independent-score barrier narrative:
surrogate and usage must each accumulate successes — one strong partner cannot
carry a weak one.

Example::

    uv run python sota/Surrogate/analysis_tools/summarize_coevolution.py \\
        --run surrogate_coevolution_run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_reports(run_dir: Path) -> list[dict]:
    history = run_dir / "history"
    if not history.is_dir():
        raise SystemExit(f"No history/ under {run_dir} (not a coevolution run?)")
    reports = []
    for path in sorted(history.glob("generation_*_attempt_*.json")):
        reports.append(json.loads(path.read_text()))
    if not reports:
        raise SystemExit(f"No generation_*_attempt_*.json in {history}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Coevolution checkpoints root")
    args = parser.parse_args()
    run_dir = Path(args.run)
    reports = load_reports(run_dir)

    print(f"run={run_dir}  attempts={len(reports)}")
    print(f"{'gen':>4} {'att':>4} {'cap':>5} {'tried':>6}  "
          f"{'surr_pass':>9} {'surr_r0':>7}  {'use_pass':>8} {'use_r0':>6}")
    for report in reports:
        stats = report.get("statistics", {}).get("populations", {})
        surr = stats.get("surrogate", {})
        usage = stats.get("usage", {})
        print(
            f"{report.get('generation', '?'):>4} "
            f"{report.get('attempt', '?'):>4} "
            f"{report.get('matches_cap', '?'):>5} "
            f"{report.get('attempted_matches', '?'):>6}  "
            f"{surr.get('passing', 0):>9} {surr.get('rank_0', 0):>7}  "
            f"{usage.get('passing', 0):>8} {usage.get('rank_0', 0):>6}"
        )
    last = reports[-1].get("statistics", {}).get("populations", {})
    print("\nIndependent score (last attempt):")
    for name, block in last.items():
        print(f"  {name}: passing={block.get('passing')} "
              f"rank_0={block.get('rank_0')} ranks={block.get('rank_counts')}")


if __name__ == "__main__":
    main()
