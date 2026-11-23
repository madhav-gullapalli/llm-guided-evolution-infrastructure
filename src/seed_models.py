"""Discover initialization seeds without importing or executing them."""
from pathlib import Path


def discover_seed_models(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Seed directory does not exist: {directory}")
    seeds = [str(path.resolve()) for path in sorted(directory.glob('*.py'))
             if path.is_file() and not path.name.startswith('_')]
    if not seeds:
        raise ValueError(f"Seed directory contains no seed models: {directory}")
    return seeds
