"""Coevolution-only observation of ensemble fits and query logging."""
import types

import numpy as np
from scipy.stats import kendalltau


def observe_usage_optimizer(optimizer):
    """Keep evolved new_epoch; measure the existing objectives after ensemble fits.

    Wrapping the ensemble factory/fit observes training without controlling
    candidate generation, acquisition, selection, or architecture evaluation.
    """
    get_ensemble = optimizer._get_ensemble

    def observed_ensemble(self, *args, **kwargs):
        ensemble = get_ensemble(*args, **kwargs)
        fit = ensemble.fit

        def observed_fit(*fit_args, **fit_kwargs):
            result = fit(*fit_args, **fit_kwargs)
            predictions = np.asarray(ensemble.query(self.test_data))
            mean_predictions = np.mean(predictions, axis=0)
            self.paired_targets = list(self.test_accuracies)
            self.paired_predictions = list(mean_predictions)
            tau, _ = kendalltau(self.test_accuracies, mean_predictions)
            mse = np.mean((np.array(self.test_accuracies) - mean_predictions) ** 2)
            self.surrogate_test_metrics.append(tau)
            self.surrogate_test_metrics_mse.append(mse)
            print(f"[Coevolution] Surrogate Test Kendall Tau = {tau:.4f}, MSE = {mse:.6f}")
            return result

        ensemble.fit = observed_fit
        return ensemble

    optimizer._get_ensemble = types.MethodType(observed_ensemble, optimizer)
    return optimizer


def _nb201_index(op_indices):
    index = 0
    for i, op in enumerate(op_indices):
        index += int(op) * (5 ** i)
    return index


def _log_trajectory_point(optimizer, model, epoch):
    if not hasattr(optimizer, "trajectory"):
        optimizer.trajectory = []
    arch = getattr(model, "arch", model)
    op_indices = getattr(arch, "op_indices", None)
    if op_indices is not None:
        op_indices = list(map(int, op_indices))
    predicted = getattr(model, "predicted_accuracy", None)
    if predicted is not None:
        predicted = float(predicted)
    optimizer.trajectory.append({
        "op_indices": op_indices,
        "generation": epoch,
        "predicted_accuracy": predicted,
        "true_accuracy": model.accuracy,
    })


def enable_query_logging(optimizer):
    """Record candidate pools and evaluated architectures for offline diversity.

    Wraps ``_get_best_candidates`` (candidate_log) and records each post-init
    evaluation into ``trajectory``. Does not change which architectures are selected.
    """
    if not hasattr(optimizer, "candidate_log"):
        optimizer.candidate_log = []
    if not hasattr(optimizer, "trajectory"):
        optimizer.trajectory = []

    original_get_best = optimizer._get_best_candidates

    def logged_get_best_candidates(self, candidates, acq_fn):
        selected = original_get_best(candidates, acq_fn)
        selected_ids = {id(candidate) for candidate in selected}
        epoch = int(getattr(self, "current_epoch", -1) or -1)
        dataset_api = getattr(self, "dataset_api", None)
        dataset = getattr(getattr(self, "config", None), "dataset", None)
        for candidate in candidates:
            arch = candidate.arch
            op_indices = list(map(int, arch.op_indices))
            predicted = getattr(candidate, "predicted_accuracy", None)
            if predicted is None and getattr(self, "ensemble", None) is not None:
                try:
                    preds = np.asarray(self.ensemble.query([arch]))
                    predicted = float(np.mean(preds))
                    candidate.predicted_accuracy = predicted
                except Exception:
                    predicted = None
            true_accuracy = None
            if dataset_api is not None and dataset is not None:
                try:
                    from naslib.search_spaces.core.query_metrics import Metric
                    true_accuracy = arch.query(
                        metric=Metric.VAL_ACCURACY, dataset=dataset, dataset_api=dataset_api)
                except Exception:
                    true_accuracy = None
            acquisition = None
            try:
                acquisition = float(np.asarray(acq_fn([arch])).reshape(-1)[0])
            except Exception:
                acquisition = None
            self.candidate_log.append({
                "generation": epoch,
                "arch_index": _nb201_index(op_indices),
                "op_indices": op_indices,
                "predicted_accuracy": float(predicted) if predicted is not None else None,
                "acquisition_value": acquisition,
                "selected": id(candidate) in selected_ids,
                "true_accuracy": float(true_accuracy) if true_accuracy is not None else None,
            })
        return selected

    optimizer._get_best_candidates = types.MethodType(logged_get_best_candidates, optimizer)

    gene_new_epoch = optimizer.new_epoch
    state = {"n_train": 0}

    def measured_new_epoch(epoch):
        optimizer.current_epoch = epoch
        gene_new_epoch(epoch)
        train_data = getattr(optimizer, "train_data", None)
        if not train_data:
            return
        new_models = train_data[state["n_train"]:]
        state["n_train"] = len(train_data)
        num_init = int(getattr(optimizer, "num_init", 0) or 0)
        if epoch >= num_init:
            for model in new_models:
                _log_trajectory_point(optimizer, model, epoch)

    optimizer.new_epoch = measured_new_epoch
    return optimizer
