"""Usage-independent NASBench-201 surrogate measurement."""
import itertools
import random

import numpy as np
from scipy.stats import kendalltau

TRAIN_SIZE = 100
TEST_SIZE = 1000
SAMPLE_SEED = 242


def architecture_split(train_size=TRAIN_SIZE, test_size=TEST_SIZE, seed=SAMPLE_SEED):
    """Uniform, disjoint samples shared by every candidate and generation."""
    architectures = list(itertools.product(range(5), repeat=6))
    if train_size < 2 or test_size < 2 or train_size + test_size > len(architectures):
        raise ValueError("Invalid NASBench-201 training/held-out sample sizes")
    sample = random.Random(seed).sample(architectures, train_size + test_size)
    return sample[:train_size], sample[train_size:]


def evaluate_surrogate(predictor_factory, architecture_factory, accuracy_query,
                       train_size=TRAIN_SIZE, test_size=TEST_SIZE, seed=SAMPLE_SEED,
                       ensemble_size=3, top_percent=None, require_tau=True):
    """Fit fresh predictors on random training data, then score untouched data.

    Factories keep benchmark loading and evolved predictor code outside the
    sampling/measurement contract. No usage optimizer participates.
    """
    train_specs, test_specs = architecture_split(train_size, test_size, seed)
    train = [architecture_factory(spec) for spec in train_specs]
    test = [architecture_factory(spec) for spec in test_specs]
    ytrain = np.asarray([accuracy_query(arch) for arch in train], dtype=float)
    ytest = np.asarray([accuracy_query(arch) for arch in test], dtype=float)
    if not np.isfinite(ytrain).all() or not np.isfinite(ytest).all():
        raise ValueError("Non-finite benchmark accuracy")
    predictions = []
    for _ in range(ensemble_size):
        predictor = predictor_factory()
        predictor.fit(train, ytrain)
        prediction = np.asarray(predictor.query(test), dtype=float).reshape(-1)
        if prediction.shape != ytest.shape or not np.isfinite(prediction).all():
            raise ValueError("Invalid held-out surrogate predictions")
        predictions.append(prediction)
    mean = np.mean(predictions, axis=0)
    tau = float(kendalltau(ytest, mean).statistic)
    if top_percent is not None:
        from src.experiment_metrics import top_tau
        tau = top_tau(ytest, mean, top_percent)
    if not np.isfinite(tau):
        if require_tau:
            raise ValueError("Undefined held-out Kendall tau")
        tau = 0.0
    return {"kendall_tau": tau, "mse": float(np.mean((ytest - mean) ** 2)),
            "train_size": train_size, "test_size": test_size, "seed": seed,
            "ensemble_size": ensemble_size,
            "train_architectures": train_specs, "test_architectures": test_specs}
