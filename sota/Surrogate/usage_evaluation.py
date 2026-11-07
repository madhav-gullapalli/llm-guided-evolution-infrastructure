"""Mode-4 oracle surrogate and independent usage-efficiency measurement."""
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np


@dataclass
class EvaluationCounts:
    real_evaluations: int = 0
    surrogate_predictions: int = 0

    def as_dict(self):
        total = self.real_evaluations + self.surrogate_predictions
        return {"real_evaluations": self.real_evaluations,
                "surrogate_predictions": self.surrogate_predictions,
                "real_evaluation_fraction": self.real_evaluations / total if total else None}


class PerfectSurrogate:
    """Predict benchmark validation accuracy exactly; fitting is a no-op."""
    def __init__(self, accuracy_query, counts):
        self.accuracy_query = accuracy_query
        self.counts = counts

    def fit(self, xtrain, ytrain, train_info=None, **kwargs):
        return 0.0

    def query(self, architectures, info=None, **kwargs):
        values = np.asarray([self.accuracy_query(arch) for arch in architectures], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite oracle prediction")
        self.counts.surrogate_predictions += len(values)
        return values

    def set_hyperparams(self, params):
        self.hyperparams = params
        return params

    def get_hyperparams(self):
        return getattr(self, "hyperparams", None)

    def set_random_hyperparams(self):
        return self.get_hyperparams()


def evaluate_usage(optimizer_factory, architecture_type, validation_metric,
                   dataset, dataset_api, budget=100):
    """Run usage with an oracle for a fixed number of policy steps.

    Count direct benchmark performance queries inside the usage policy. Oracle
    lookups bypass the counter. Search-space setup, test-set construction and
    trainer reporting are evaluator overhead and excluded from this metric.
    """
    if budget < 1:
        raise ValueError("Oracle usage budget must be positive")
    counts = EvaluationCounts()
    original_query = architecture_type.query

    def oracle_accuracy(arch):
        return original_query(arch, validation_metric, dataset, dataset_api=dataset_api)

    def predictor_factory(**kwargs):
        return PerfectSurrogate(oracle_accuracy, counts)

    optimizer = optimizer_factory(predictor_factory)
    optimizer.adapt_search_space(architecture_type(), dataset_api=dataset_api)

    def counted_query(arch, metric, *args, **kwargs):
        name = getattr(metric, "name", str(metric))
        if name in {"RAW", "ALL"} or name.endswith(("_ACCURACY", "_LOSS")):
            counts.real_evaluations += 1
        return original_query(arch, metric, *args, **kwargs)

    # No Trainer: its repeated incumbent/statistics queries are bookkeeping.
    with patch.object(architecture_type, "query", counted_query):
        optimizer.before_training()
        for epoch in range(budget):
            optimizer.new_epoch(epoch)
        optimizer.after_training()
    return {**counts.as_dict(), "policy_steps": budget}
